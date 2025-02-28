import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from tokenizer import Tokenizer

# Uncomment or define your Tokenizer class as needed.
# from tokenizer import Tokenizer

# For better GPU memory management
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------
# Data Preparation
# ---------------------------
MAX_SEQ_LENGTH = 128  # Maximum tokens per sentence

def tokenize_corpus(corpus_path):
    """
    Reads the corpus, tokenizes it, and ensures that no sentence exceeds MAX_SEQ_LENGTH.
    If a sentence is longer, it is split into multiple sentences of length MAX_SEQ_LENGTH.
    Start ("<s>") and end ("</s>") tokens are added to each segment.
    """
    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus = file.read()
    
    tokenizer = Tokenizer(corpus)  # Assuming Tokenizer is defined elsewhere
    tokenized_sentences = tokenizer.tokenize()

    final_tokenized = []
    
    for sentence in tokenized_sentences:
        if len(sentence) > MAX_SEQ_LENGTH:
            # Split into multiple sentences of length MAX_SEQ_LENGTH
            for i in range(0, len(sentence), MAX_SEQ_LENGTH):
                sub_sentence = sentence[i:i + MAX_SEQ_LENGTH]  # Take 128 tokens at a time
                final_tokenized.append(["<s>"] + sub_sentence + ["</s>"])
        else:
            # Keep sentence as is if within length limit
            final_tokenized.append(["<s>"] + sentence + ["</s>"])
    
    # Split dataset: 80% Train, 10% Validation, 10% Test
    train_sentences, temp_sentences = train_test_split(final_tokenized, test_size=0.2, random_state=42)
    val_sentences, test_sentences = train_test_split(temp_sentences, test_size=0.5, random_state=42)
    
    # Build vocabulary from training sentences; add <UNK> token
    vocab = {word: idx for idx, word in enumerate(set(word for sentence in train_sentences for word in sentence))}
    vocab["<UNK>"] = len(vocab)
    
    return train_sentences, val_sentences, test_sentences, vocab

# ---------------------------
# Dataset Definition (Full Sentence)
# ---------------------------
class SentenceDataset(Dataset):
    """
    Dataset that returns entire sentences as sequences of token IDs.
    For each sentence, the input is sentence[:-1] and the target is sentence[1:].
    """
    def __init__(self, tokenized_corpus, vocab):
        self.vocab = vocab
        self.sentences = tokenized_corpus  # Each sentence is a list of tokens
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in sentence[:-1]]
        target_tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in sentence[1:]]
        return (torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(target_tokens, dtype=torch.long),
                " ".join(sentence))

# ---------------------------
# Collate Function for Padding
# ---------------------------
def collate_fn(batch):
    """
    Pads sequences in a batch.
    Each item in the batch is (input_tensor, target_tensor, sentence_string).
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    sentences = [item[2] for item in batch]
    
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, sentences


class LSTMLanguageModel(nn.Module):
    """
    Vanilla LSTM Language Model.
    
    The model embeds input tokens, processes the sequence with an LSTM,
    and uses the final hidden state to predict the next word.
    
    Args:
      vocab_size (int): Size of the vocabulary.
      embed_size (int): Dimensionality of word embeddings.
      hidden_size (int): Size of the LSTM hidden state.
      num_layers (int): Number of LSTM layers.
      dropout_rate (float): Dropout rate.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout_rate=0.5):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        """
        Args:
          x (Tensor): Input tensor of shape (batch_size, seq_length)
          hidden (tuple): Optional initial (hidden, cell) state.
          
        Returns:
          log_probs (Tensor): Log-probabilities with shape (batch_size*seq_length, vocab_size)
          hidden (tuple): Final (hidden, cell) state.
        """
        batch_size = x.size(0)  # Get batch size dynamically

        # Initialize hidden and cell states if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)

        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        output, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_length, hidden_size)
        output = self.layer_norm(output)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch_size, seq_length, vocab_size)
        
        batch_size, seq_len, _ = logits.size()
        logits = logits.contiguous().view(batch_size * seq_len, -1)  # Flatten for loss computation
        log_probs = self.log_softmax(logits)

        return log_probs, hidden


# ---------------------------
# Training Function (Using AMP)
# ---------------------------
def train(model, train_loader, optimizer, criterion, device, epochs):
    """ Trains the RNN/LSTM model using mixed precision (AMP) and ignores padding tokens. """
    model.train()
    scaler = GradScaler()

    for epoch in range(epochs):
        epoch_loss, total_samples = 0, 0
    
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
    
            with autocast():
                log_probs, _ = model(inputs)
                flat_targets = targets.contiguous().view(-1)
                # With reduction="none", loss is a vector of losses per token
                loss = criterion(log_probs, flat_targets)
                mask = flat_targets != 0  # Assuming 0 is the <PAD> index
                masked_loss = loss[mask]   # Now we can index since loss is not a scalar
                if masked_loss.numel() > 0:
                    loss = masked_loss.mean()
                else:
                    loss = torch.tensor(0.0, device=device)
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            epoch_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()
    
        avg_loss = epoch_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    return model

# ---------------------------
# Perplexity Calculation Functions
# ---------------------------
def calculate_nll(model, loader, device):
    """
    Compute the negative log-likelihood (NLL) loss for each sentence.
    For each sentence, flatten predictions and targets and compute mean loss.
    """
    model.eval()
    sentence_losses = []
    sentences_out = []
    
    with torch.no_grad():
        for inputs, targets, sentences in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            log_probs, _ = model(inputs)  # (batch_size * seq_len, vocab_size)
            flat_targets = targets.contiguous().view(-1)
            loss = nn.NLLLoss(reduction="none")(log_probs, flat_targets)
            # Reshape loss to (batch_size, seq_len)
            batch_size, seq_len = inputs.size(0), inputs.size(1)
            loss = loss.view(batch_size, seq_len)
            for i in range(batch_size):
                # Create mask for non-padding tokens (assuming padding value 0)
                mask = targets[i] != 0
                if mask.sum().item() > 0:
                    mean_loss = loss[i][mask].mean().item()
                else:
                    mean_loss = float('inf')
                sentence_losses.append(mean_loss)
            sentences_out.extend(sentences)
    return sentence_losses, sentences_out

def set_perplexity(model, loader, device, file_name):
    """
    Computes and saves sentence-level perplexity.
    Duplicate sentences are merged by averaging their losses.
    """
    nll_losses, sentences = calculate_nll(model, loader, device)
    # Merge duplicate sentences
    loss_dict = defaultdict(list)
    for sent, loss in zip(sentences, nll_losses):
        loss_dict[sent].append(loss)
    
    with open(file_name, "w") as f:
        f.write(f"{'Sentence':<80}{'Perplexity'}\n")
        f.write("=" * 100 + "\n")
        for sent, losses in loss_dict.items():
            avg_loss = np.mean(losses)
            ppl = np.exp(avg_loss)
            f.write(f"{sent:<80}{ppl:.5f}\n")
        overall_ppl = np.exp(np.mean([np.mean(losses) for losses in loss_dict.values()]))
        f.write("=" * 100 + "\n")
        f.write(f"{'Average Perplexity':<80}{overall_ppl:.5f}\n")
        print(f"Average perplexity of {file_name}: {overall_ppl:.5f}")

# ---------------------------
# Run Experiment (Training and Evaluation)
# ---------------------------
def run_experiment(corpus_path, model_type="LSTM"):
    # Tokenize and prepare data
    train_sentences, val_sentences, test_sentences, vocab = tokenize_corpus(corpus_path)
    train_dataset = SentenceDataset(train_sentences, vocab)
    val_dataset = SentenceDataset(val_sentences, vocab)
    test_dataset = SentenceDataset(test_sentences, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True)
    
    vocab_size = len(vocab)
    embed_size = 300
    hidden_size = 256
    
    # Create LSTM model instance
    model = LSTMLanguageModel(vocab_size, embed_size, hidden_size, num_layers=1, dropout_rate=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss(ignore_index=0,reduction="none")  # 0 is the index of <PAD> token in the vocab
    
    print(f"\nTraining {model_type} Model on {corpus_path}...\n")
    gc.collect()
    torch.cuda.empty_cache()
    model = train(model, train_loader, optimizer, criterion, device, epochs=5)
    
    model_save_path = f"trained_model_{os.path.basename(corpus_path)}_{model_type}.pt"
    torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab}, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Calculate perplexity for each set
    train_file = f"{os.path.basename(corpus_path)}_train_perplexity_{model_type}.txt"
    val_file = f"{os.path.basename(corpus_path)}_val_perplexity_{model_type}.txt"
    test_file = f"{os.path.basename(corpus_path)}_test_perplexity_{model_type}.txt"
    
    print("\nCalculating and saving Train Perplexity...")
    set_perplexity(model, train_loader, device, train_file)
    print("\nCalculating and saving Validation Perplexity...")
    set_perplexity(model, val_loader, device, val_file)
    print("\nCalculating and saving Test Perplexity...")
    set_perplexity(model, test_loader, device, test_file)
    print(f"Perplexity scores saved in: {train_file}, {val_file}, {test_file}")
    
    return model_save_path



# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    pride_corpus = "iNLP/Pride_and_Prejudice.txt"
    ulysses_corpus = "iNLP/Ulysses.txt"
    
    saved_models = []
    for corpus in [pride_corpus,ulysses_corpus]:
        saved_models.append(run_experiment(corpus, model_type="LSTM"))

    print("\nSaved models:", saved_models)