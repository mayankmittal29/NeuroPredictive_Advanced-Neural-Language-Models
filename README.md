
# 🧠 NeuroPredictive: Advanced Neural Language Models

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)

> Powerful neural architectures for next word prediction using FFNN, RNN, and LSTM models

## 📚 Project Overview

NeuroPredictive is an implementation of state-of-the-art neural language models for next word prediction. This project explores the capabilities and performance differences between Feed-Forward Neural Networks (FFNN), vanilla Recurrent Neural Networks (RNN), and Long Short-Term Memory (LSTM) networks across different literary corpora.

### 🔍 Task Definition

Next Word Prediction (NWP) involves predicting the most probable word that follows a given sequence of words. Mathematically:

Given a sequence of words w₁, w₂, ..., wₙ₋₁, the goal is to determine the next word wₙ such that:
wₙ = arg max(w∈V) P(w|w₁, w₂, ..., wₙ₋₁)

where V is the vocabulary and P(w|w₁, w₂, ..., wₙ₋₁) is the conditional probability of w given the preceding words.

## 🚀 Models Implemented

### 1️⃣ Feed Forward Neural Network (FFNN)
- Fixed-size window of preceding words
- Implementation with n-grams of sizes 3 and 5
- Uses fully connected layers with activation functions
- Predicts based on top k probabilities

### 2️⃣ Vanilla Recurrent Neural Network (RNN)
- Maintains hidden state capturing sequential dependencies
- Iteratively processes input word embeddings
- Addresses limitations of fixed context windows
- Predicts next word at each step

### 3️⃣ Long Short-Term Memory Network (LSTM)
- Extends RNN with sophisticated memory cells
- Employs input, forget, and output gates
- Captures long-term dependencies in sequences
- Solves vanishing gradient problem

## 📊 Dataset

The models were trained and evaluated on two classic literary corpora:

- **Pride and Prejudice** by Jane Austen (124,970 words)
- **Ulysses** by James Joyce (268,117 words)

For testing, 1,000 sentences were randomly selected from each corpus and excluded from training.

## 🔧 Installation and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/NeuroPredictive.git
cd NeuroPredictive

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

Run the generator script with the following syntax:

```bash
python3 generator.py <lm_type> <corpus_path> <k>
```

### Parameters:
- `lm_type`: Type of language model
  - `-f`: Feed Forward Neural Network (FFNN)
  - `-r`: Recurrent Neural Network (RNN)
  - `-l`: Long Short-Term Memory (LSTM)
- `corpus_path`: Path to the dataset file
- `k`: Number of candidate next words to print

### Example:
```bash
python3 generator.py -l ./pride_and_prejudice.txt 3
```

This will prompt you to enter a sentence, and then output the top 3 most probable next words with their probability scores.

## 📈 Performance Analysis

### 📊 Model Rankings (Best to Worst)

1. LSTM Language Model
2. Vanilla RNN Language Model
3. FFNN Language Model (n=5)
4. FFNN Language Model (n=3)
5. N-gram models with Good-Turing Smoothing
6. N-gram models with Linear Interpolation
7. N-gram models with Laplace Smoothing

### 🔍 Key Findings

#### LSTM Performance
- **Superior performance** especially for longer sentences
- Effective management of long-term dependencies through gating mechanisms
- Stable gradient flow during backpropagation
- Implementation highlights:
  ```python
  self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
  self.dropout = nn.Dropout(dropout_rate)
  ```

#### RNN Performance
- Good sequential pattern capture with simple architecture
- Limited by vanishing gradients on longer sequences
- Implementation highlights:
  ```python
  self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
  ```

#### FFNN Performance
- Performance varies based on context size (n=3 vs n=5)
- Multiple linear layers with ReLU activation provide good learning capacity
- Limited by fixed context window
- Implementation highlights:
  ```python
  self.linear1 = nn.Linear(embedding_dim * context_size, hidden_dim)
  self.linear2 = nn.Linear(hidden_dim, hidden_dim)
  self.linear3 = nn.Linear(hidden_dim, vocab_size)
  ```

### 📏 N-gram Size Impact (FFNN)

#### N=3 Configuration
- ✅ Faster training due to smaller input dimension
- ✅ Better generalization on common short phrases
- ✅ Lower memory requirements
- ❌ Limited context capture
- ❌ Higher perplexity on complex sentences

#### N=5 Configuration
- ✅ Better capture of medium-range dependencies
- ✅ Improved prediction accuracy for longer phrases
- ✅ More contextual information for prediction
- ❌ Increased model complexity
- ❌ Higher computational requirements
- ❌ Risk of overfitting on smaller datasets

## 🌟 Performance on Longer Sentences

The LSTM model significantly outperforms others on longer sentences due to:

- **Memory Cell Advantage**: Maintains relevant information over longer sequences
- **Adaptive Context**: Dynamically adapts memory content based on sequence length
- **Gradient Stability**: Prevents vanishing/exploding gradient problems

Implementation example:
```python
hidden = (torch.zeros(1, inputs.size(0), self.hidden_dim),
          torch.zeros(1, inputs.size(0), self.hidden_dim))
output, hidden = self.lstm(embeds, hidden)
```

## 🧠 Conclusions

1. LSTM's sophisticated architecture provides the best overall performance, particularly for long sequences
2. Vanilla RNN offers good balance between performance and complexity but struggles with longer sequences
3. FFNN models show competitive performance for short sequences but are limited by fixed context window
4. Choice of n-gram size in FFNN models presents a clear trade-off between context capture and computational efficiency

For applications requiring robust language modeling, especially with varying sentence lengths, the LSTM model is the most suitable choice, while FFNN models might be preferred for applications with known, limited context requirements.

## 💾 Pretrained Models

Pretrained models are available at: [Model Download Link](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/mayank_mittal_students_iiit_ac_in/Ek00vuB-RTFBpEv0QI8ZN7EBTX1BZa5EAeEuoA-islsy9Q?e=uUl5mn)

## 👤 Author

Mayank Mittal (2022101094)
