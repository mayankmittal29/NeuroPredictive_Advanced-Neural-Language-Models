import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
import os
def tokenize_corpus(corpus_path, n):
    with open(corpus_path, "r", encoding="utf-8") as file:
        corpus = file.read()

    tokenizer = Tokenizer(corpus)
    tokenized_sentences = tokenizer.tokenize()

    # Add <s> and </s> tokens
    final_tokenized = []
    for sen in tokenized_sentences:
        final_tokenized.append(["<s>"] * (n-1) + sen + ["</s>"])

    # Split dataset into train (80%), val (10%), test (10%)
    train_sentences, temp_sentences = train_test_split(final_tokenized, test_size=0.2, random_state=42)
    val_sentences, test_sentences = train_test_split(temp_sentences, test_size=0.5, random_state=42)

    # Build Vocabulary (Add an <UNK> token for unseen words)
    vocab = {word: idx for idx, word in enumerate(set(word for sentence in train_sentences for word in sentence))}
    vocab["<UNK>"] = len(vocab)  # Add <UNK> token

    return train_sentences, val_sentences, test_sentences, vocab

class NGramDataset(Dataset):
    """ Dataset for N-Gram Language Model """
    def __init__(self, tokenized_corpus, n, vocab):
        self.n = n
        self.vocab = vocab
        self.data = []

        for sentence in tokenized_corpus:
            for i in range(len(sentence) - n):
                context = sentence[i:i + n - 1]
                target = sentence[i + n - 1]
                self.data.append((context, target, sentence))  # Store sentence here!

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target, sentence = self.data[idx]
        context_tensor = torch.tensor([self.vocab.get(word, self.vocab["<UNK>"]) for word in context], dtype=torch.long)
        target_tensor = torch.tensor(self.vocab.get(target, self.vocab["<UNK>"]), dtype=torch.long)
        return context_tensor, target_tensor, " ".join(sentence)  # Return full sentence as string



class FFNNLanguageModel(nn.Module):
    """ Feed Forward Neural Network Language Model """
    def __init__(self, vocab_size, embed_size, hidden_size, n,dropout_rate):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear((n-1) * embed_size, hidden_size)
        self.relu = nn.GELU()# TRY Gelu
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


def calculate_nll(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[float], list[str]]:
    model.eval()

    sentence_perplexities, sentences = [], []
    with torch.no_grad():
        for context, target, batch_sentences in loader:
            context, target = context.to(device), target.to(device)
            # each of these are (batch_size, item)
            output = model(context)
            if output.size(0) != target.size(0):
                # Adjust output to match target size
                if output.size(0) < target.size(0):
                    target = target[: output.size(0)]
                else:
                    output = output[: target.size(0)]

            loss = torch.nn.NLLLoss(reduction="none")(output, target)
            num_sentences = len(loss)
            # append the perplexities one by one
            sentence_perplexities.extend(loss.cpu().numpy().tolist())
            sentences.extend(batch_sentences[:num_sentences])
    return sentence_perplexities, sentences

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple

def set_perplexity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    file_name: str,
) -> None:
    """
    Computes and saves the perplexity of sentences in the given DataLoader.

    Args:
    - model (torch.nn.Module): Trained model.
    - loader (DataLoader): DataLoader containing the dataset.
    - device (torch.device): CPU/GPU device.
    - file_name (str): Output file name to save perplexity scores.
    """
    nll_losses, sentences = calculate_nll(model, loader, device)
    assert len(nll_losses) == len(
        sentences
    ), "[set_perplexity] nll losses should be same length as sentences"

    sentence_perplexity = defaultdict(list)  
    
    for sentence, nll_loss in zip(sentences, nll_losses):
        sentence_perplexity[sentence].append(nll_loss)
        
    with open(file_name, "w") as f:
        for sentence in sentence_perplexity:
            perplexity = np.mean([np.exp(s) for s in sentence_perplexity[sentence]])
            f.write(f"{sentence}\t\t\t\t{(perplexity)}\n")

        average_perplexity = np.exp(sum(nll_losses) / len(nll_losses))
        f.write(f"\naverage perplexity: {average_perplexity}\n")
        print(f"Average perplexity of {file_name}: ", average_perplexity)


import torch
import torch.nn as nn
import torch.optim as optim

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train the model for epochs.

    Args:
    - model (torch.nn.Module): The FFNN language model.
    - train_loader (DataLoader): DataLoader for training dataset.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - criterion (nn.Module): Loss function (nn.NLLLoss).
    - device (torch.device): Device to train on (CPU/GPU).

    Returns:
    - float: Average training loss per sample.
    """
    num_items = len(train_loader.dataset)
    assert num_items > 0, "[train] Training data must be present"
    
    model.train()
    total_loss = 0

    for e in range(epoch):
        total_loss=0 
        for context, target,_ in train_loader:  # Expecting dataset to return (context, target, sentence)
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(context)

            # Ensure output and target sizes match
            if output.size(0) != target.size(0):
                if output.size(0) < target.size(0):
                    target = target[: output.size(0)]
                else:
                    output = output[: target.size(0)]

            # Compute loss
            loss = criterion(output, target)
            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {e} Loss: ",total_loss / num_items)

    return model

def run_experiment(corpus_path, n, model_type="FFNN"):
    # Tokenize, build vocab, etc.
    train_sentences, val_sentences, test_sentences, vocab = tokenize_corpus(corpus_path, n)

    # Create Datasets
    train_dataset = NGramDataset(train_sentences, n, vocab)
    val_dataset = NGramDataset(val_sentences, n, vocab)
    test_dataset = NGramDataset(test_sentences, n, vocab)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    # Initialize model
    vocab_size = len(vocab)
    embed_size = (300)
    hidden_size = (n*80)

    model = FFNNLanguageModel(vocab_size, embed_size, hidden_size, n,0.6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

     # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # Train the model
    print(f"\nTraining FFNN Model for {n}-grams on {corpus_path}...\n")
    epoch=5
    # if(n==5):
    #     epoch=4
    model = train(model, train_loader, optimizer, criterion, device,epoch)

    # Save the trained model
    model_save_path = f"trained_model_{n}_{os.path.basename(corpus_path)}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,  # Save the vocab so we can reuse it later
        'n': n
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Compute Sentence-Level Perplexity
    criterion = nn.CrossEntropyLoss(reduction='none')
    # Compute Perplexity for Train, Validation, and Test sets & Save to files
    train_file = f"{n}_{corpus_path.split('/')[-1]}_train_perplexity.txt"
    val_file = f"{n}_{corpus_path.split('/')[-1]}_val_perplexity.txt"
    test_file = f"{n}_{corpus_path.split('/')[-1]}_test_perplexity.txt"

    print("\nCalculating and saving Train Perplexity...")
    set_perplexity(model, train_loader, device, train_file)

    print("\nCalculating and saving Validation Perplexity...")
    set_perplexity(model, val_loader, device, val_file)

    print("\nCalculating and saving Test Perplexity...")
    set_perplexity(model, test_loader, device, test_file)

    print(f"Perplexity scores saved in: {train_file}, {val_file}, {test_file}")
    
    return model_save_path



if __name__ == "__main__":
    # Corpus Paths
    pride_corpus = "iNLP/Pride_and_Prejudice.txt"
    ulysses_corpus = "iNLP/Ulysses.txt"

    saved_models = []  # Store the saved model paths

    # Run for n=3 and n=5
    for n in [3,5]:
        saved_models.append(run_experiment(pride_corpus, n))
        saved_models.append(run_experiment(ulysses_corpus, n))

    print("\nSaved models:", saved_models)