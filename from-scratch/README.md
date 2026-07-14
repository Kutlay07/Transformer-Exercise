# Transformer From Scratch

A manual implementation of the original Transformer architecture introduced in:

**"Attention Is All You Need" (Vaswani et al., 2017)**

This project was built to understand the mathematical foundations of Transformer models by implementing the core operations manually using PyTorch tensor operations instead of high-level Transformer modules.

---

# Implementation Details

The model components were implemented manually, including:

## Attention Mechanism

- Query, Key, Value projections
- Scaled Dot-Product Attention
- Multi-Head Attention
- Causal Masking for decoder self-attention
- Encoder-Decoder Cross Attention

Attention calculations were performed using explicit matrix operations:

- Matrix multiplication
- Transpose operations
- Softmax normalization
- Masking with negative infinity

---

# Transformer Components

Implemented components:

- Encoder
- Decoder
- Multi-Head Attention
- Positional Encoding
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Output Projection Layer

---

# Training

The model was trained on a small custom text corpus for next-token prediction.

The training process includes:

- Sequence preparation
- Forward propagation
- Loss calculation
- Backpropagation
- Parameter optimization

---

# Technologies

- Python
- PyTorch Tensor Operations
- Deep Learning
- Transformer Architecture
- Natural Language Processing

---

# Purpose

The goal of this project was to understand how Transformer architectures work internally by manually implementing the mathematical operations behind attention mechanisms and decoder generation.

---

# Reference

Vaswani et al.

"Attention Is All You Need"

2017
