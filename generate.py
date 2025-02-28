import torch
import torch.nn as nn
import sys
import os
import numpy as np
from FFNN import FFNNLanguageModel  
from RNN import RNNLanguageModel    
from LSTM import LSTMLanguageModel
# ---------------------------
# FFNN Generator Class
# ---------------------------

import torch
import torch.nn as nn
import numpy as np


# ---------------------------
# FFNN Generator Class
# ---------------------------
class FFNNGenerator:
    def __init__(self, model_path, device):
        """
        Initialize the FFNN Generator by loading the trained model.
        """
        checkpoint = torch.load(model_path, map_location=device)

        self.vocab = checkpoint['vocab']
        self.n = checkpoint['n']
        vocab_size = len(self.vocab)
        embed_size = 300
        hidden_size = (self.n*80)

        # Initialize the model
        self.model = FFNNLanguageModel(vocab_size, embed_size, hidden_size, self.n, dropout_rate=0.5)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict_next_words(self, input_text, k):
        """
        Predicts the top K most probable next words given an input text.
        """
        words = input_text.strip().split()

        if len(words) < self.n - 1:
            print(f"[ERROR] Input must have at least {self.n-1} words.")
            return []

        # Extract last (n-1) words for context
        context_words = words[-(self.n-1):]

        # Convert words to indices
        context_indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in context_words]
        context_tensor = torch.tensor(context_indices, dtype=torch.long, device=self.device).unsqueeze(0)

        # Get model predictions
        with torch.no_grad():
            output = self.model(context_tensor)  # Shape: (1, vocab_size)
            probabilities = torch.exp(output).squeeze(0).cpu().numpy()  # Convert log-probs to normal probs

        # Get the top K words
        top_k_indices = np.argsort(probabilities)[-k:][::-1]  # Sort in descending order
        index_to_word = {idx: word for word, idx in self.vocab.items()}
        top_k_words = [(index_to_word[idx], probabilities[idx]) for idx in top_k_indices]

        return top_k_words


# ---------------------------
# RNN Generator Class
# ---------------------------
class RNNGenerator:
    def __init__(self, model_path, device):
        """
        Initialize the RNN Generator by loading the trained model.
        """
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)

        self.vocab = checkpoint['vocab']
        vocab_size = len(self.vocab)
        embed_size = 300    # Ensure this matches the training config
        hidden_size = 512   # Ensure this matches the training config
        num_layers = 1      # As used during training
        dropout_rate = 0.3  # As used during training

        # Initialize the model
        self.model = RNNLanguageModel(vocab_size, embed_size, hidden_size, num_layers, dropout_rate)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict_next_words(self, input_sentence, k):
        """
        Given an input sentence, returns the top k predicted next words along with their probabilities.
        """
        words = input_sentence.strip().split()
        if not words:
            print("[ERROR] Input sentence is empty.")
            return []

        # Convert words to indices
        indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        input_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            log_probs, _ = self.model(input_tensor)

        # Determine sequence length
        seq_length = input_tensor.size(1)
        vocab_size = len(self.vocab)

        # Reshape log_probs
        log_probs = log_probs.view(1, seq_length, vocab_size)
        last_log_probs = log_probs[:, -1, :]
        probabilities = torch.exp(last_log_probs).squeeze(0)

        # Get the top k predictions
        topk = torch.topk(probabilities, k, largest=True, sorted=True)
        topk_indices = topk.indices.cpu().numpy()
        topk_probs = topk.values.cpu().numpy()

        # Create inverse mapping
        index_to_word = {idx: word for word, idx in self.vocab.items()}
        predictions = [(index_to_word.get(int(idx), "<UNK>"), float(prob)) for idx, prob in zip(topk_indices, topk_probs)]

        return predictions


# ---------------------------
# LSTM Generator Class
# ---------------------------
class LSTMGenerator:
    def __init__(self, model_path, device):
        """
        Initialize the LSTM Generator by loading the trained model.
        """
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)

        self.vocab = checkpoint['vocab']
        vocab_size = len(self.vocab)
        embed_size = 300   # Must match training configuration
        hidden_size = 256  # Must match training configuration
        num_layers = 1     # As used during training
        dropout_rate = 0.3 # As used during training

        self.model = LSTMLanguageModel(vocab_size, embed_size, hidden_size, num_layers, dropout_rate)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict_next_words(self, input_text, k):
        """
        Given an input sentence, predicts the top k most probable next words.
        """
        words = input_text.strip().split()
        if not words:
            print("[ERROR] Input sentence cannot be empty.")
            return []

        # Convert words to indices; use <UNK> if not found
        context_indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        context_tensor = torch.tensor(context_indices, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            log_probs, _ = self.model(context_tensor)

        batch_size = context_tensor.size(0)
        seq_len = context_tensor.size(1)
        vocab_size = len(self.vocab)

        # Reshape log_probs
        log_probs = log_probs.view(batch_size, seq_len, vocab_size)
        last_log_probs = log_probs[:, -1, :]
        probabilities = torch.exp(last_log_probs).squeeze(0)

        # Get top k indices sorted in descending order
        topk = torch.topk(probabilities, k, largest=True, sorted=True)
        topk_indices = topk.indices.cpu().numpy().flatten()
        topk_probs = topk.values.cpu().numpy().flatten()

        # Create an inverse mapping: index -> word
        index_to_word = {idx: word for word, idx in self.vocab.items()}
        predictions = [(index_to_word.get(int(idx), "<UNK>"), float(prob)) for idx, prob in zip(topk_indices, topk_probs)]
        
        return predictions



# ---------------------------
# Main Script
# ---------------------------

if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) != 4:
        print("Usage: python generator.py <lm_type> <corpus_path> <k>")
        sys.exit(1)

    lm_type = sys.argv[1]  # Language model type (-f, -r, -l)
    corpus_path = sys.argv[2]  # Corpus file path
    k = int(sys.argv[3])  # Number of top predictions

    if lm_type not in ["-f", "-r", "-l"]:
        print("[ERROR] Invalid language model type! Use -f for FFNN, -r for RNN, -l for LSTM.")
        sys.exit(1)
    lm = {}
    lm['-f'] = 'FFNN'    
    lm['-r'] = 'RNN' 
    lm['-l'] = 'LSTM'
    # Find the corresponding model file
    if lm_type == "-f": 
        n = int(input("Enter the ngram value: \n"))
        model_path = f"trained_model_{n}_{os.path.basename(corpus_path)}.pt"
    else:
        model_path = f"trained_model_{os.path.basename(corpus_path)}_{lm[lm_type]}.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} not found.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select the correct generator class
    if lm_type == "-f":
        generator = FFNNGenerator(model_path, device)
    elif lm_type == "-r":
        generator = RNNGenerator(model_path, device)
    elif lm_type == "-l":
        generator = LSTMGenerator(model_path, device)

    while True:
        input_text = input("\nEnter input sentence (or type 'exit' to quit): ").strip()
        if input_text.lower() == "exit":
            break

        top_k_words = generator.predict_next_words(input_text, k)

        top_k_predictions = sorted(top_k_words, key=lambda x: x[1], reverse=True)[:k]  
        if top_k_predictions:
            print("\nTop predictions:")
            for word, prob in top_k_predictions:
                print(f"  {word}: {prob:.4f}")
