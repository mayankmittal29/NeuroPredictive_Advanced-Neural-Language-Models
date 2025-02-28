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

# Optimize GPU usage and reduce fragmentation
torch.backends.cudnn.benchmark = True  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------
# Data Preparation
# ---------------------------
MAX_SEQ_LENGTH = 128  # Maximum tokens per sentence

def tokenize_corpus(corpus_path):
    """
    Reads the corpus, tokenizes it, and splits long sentences.
    If a sentence is longer than MAX_SEQ_LENGTH, it is split into multiple segments.
    Each segment gets a start token "<s>" and an end token "</s>".
    """
    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus = file.read()
    
    # Use your Tokenizer here; if not available, you can use a simple split.
    try:
        tokenizer = Tokenizer(corpus)  # Assumed to be defined elsewhere
        tokenized_sentences = tokenizer.tokenize()
    except Exception:
        tokenized_sentences = [line.split() for line in corpus.splitlines() if line.strip()]
    
    final_tokenized = []
    for sentence in tokenized_sentences:
        if len(sentence) > MAX_SEQ_LENGTH:
            for i in range(0, len(sentence), MAX_SEQ_LENGTH):
                sub_sentence = sentence[i:i + MAX_SEQ_LENGTH]
                final_tokenized.append(["<s>"] + sub_sentence + ["</s>"])
        else:
            final_tokenized.append(["<s>"] + sentence + ["</s>"])
    
    # Split into train (80%), validation (10%), test (10%)
    train_sentences, temp_sentences = train_test_split(final_tokenized, test_size=0.2, random_state=42)
    val_sentences, test_sentences = train_test_split(temp_sentences, test_size=0.5, random_state=42)
    
    # ---------------------------
    # Build Vocabulary with explicit <PAD> token at index 0.
    # ---------------------------
    vocab_set = set(word for sentence in train_sentences for word in sentence)
    vocab = {"<PAD>": 0}  # Reserve index 0 for padding.
    for word in vocab_set:
        if word != "<PAD>":
            vocab[word] = len(vocab)
    # Add <UNK> token if not already in vocab
    if "<UNK>" not in vocab:
        vocab["<UNK>"] = len(vocab)
    
    return train_sentences, val_sentences, test_sentences, vocab

# ---------------------------
# Dataset Definition
# ---------------------------
class SentenceDataset(Dataset):
    """
    Dataset that returns entire sentences as sequences of token IDs.
    For each sentence, input is sentence[:-1] and target is sentence[1:].
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
    Pads sequences in a batch to the same length.
    Each item in the batch is (input_tensor, target_tensor, sentence_string).
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    sentences = [item[2] for item in batch]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, sentences

# ---------------------------
# Vanilla RNN Language Model Definition
# ---------------------------
class RNNLanguageModel(nn.Module):
    """ RNN-based language model for full-sentence prediction. """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout_rate=0.5):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.relu = nn.Tanh()  # Using GELU; you can experiment with other activations.
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        output, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_length, hidden_size)
        output = self.layer_norm(output)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch_size, seq_length, vocab_size)
        batch_size, seq_len, _ = logits.size()
        logits = logits.contiguous().view(batch_size * seq_len, -1)
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
def calculate_nll(model, loader, device) -> tuple[list[float], list[str]]:
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
            log_probs, _ = model(inputs)  # (batch_size * seq_length, vocab_size)
            flat_targets = targets.contiguous().view(-1)
            loss = nn.NLLLoss(reduction="none")(log_probs, flat_targets)
            batch_size, seq_len = inputs.size(0), inputs.size(1)
            loss = loss.view(batch_size, seq_len)
            for i in range(batch_size):
                # Create mask for non-padding tokens (padding is 0, reserved for <PAD>)
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
    Computes and saves the perplexity of sentences in the given DataLoader.
    Duplicate sentences are merged by averaging their losses.
    """
    nll_losses, sentences = calculate_nll(model, loader, device)
    from collections import defaultdict
    sentence_loss_dict = defaultdict(list)
    for sentence, loss in zip(sentences, nll_losses):
        sentence_loss_dict[sentence].append(loss)
    with open(file_name, "w") as f:
        f.write(f"{'Sentence':<80}{'Perplexity'}\n")
        f.write("=" * 100 + "\n")
        for sentence, losses in sentence_loss_dict.items():
            avg_loss = np.mean(losses)
            ppl = np.exp(avg_loss)
            f.write(f"{sentence:<80}{ppl:.5f}\n")
        overall_ppl = np.exp(np.mean([np.mean(losses) for losses in sentence_loss_dict.values()]))
        f.write("=" * 100 + "\n")
        f.write(f"{'Average Perplexity':<80}{overall_ppl:.5f}\n")
        print(f"Average perplexity of {file_name}: {overall_ppl:.5f}")

# ---------------------------
# Run Experiment Function
# ---------------------------
def run_experiment(corpus_path, model_type="RNN"):
    """
    Runs the full training and evaluation pipeline.
    """
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
    hidden_size = 512
    
    # Create RNN Model
    model = RNNLanguageModel(vocab_size, embed_size, hidden_size, num_layers=1, dropout_rate=0.3)
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
    for corpus in [pride_corpus, ulysses_corpus]:
        saved_models.append(run_experiment(corpus, model_type="RNN"))
    
    print("\nSaved models:", saved_models)
