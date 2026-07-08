# Transformer from Scratch (Attention Is All You Need)

A complete educational implementation of the original Transformer architecture from the **Attention Is All You Need (2017)** paper using PyTorch.

The goal of this project is **not** to build a state-of-the-art translation model, but to understand every component of the Transformer by implementing it manually instead of relying on `nn.Transformer`.

---

## Project Goals

* Implement the original Encoder–Decoder Transformer architecture from scratch.
* Understand how self-attention and cross-attention work internally.
* Build the complete training pipeline manually.
* Learn masking, positional encoding, residual connections and normalization.
* Experiment with optimization techniques and debug real training issues.

---

## Features

* Word-level tokenizer
* Custom vocabulary generation
* Sinusoidal Positional Encoding
* Multi-Head Self Attention
* Cross Attention
* Encoder Stack
* Decoder Stack
* Feed Forward Networks
* Residual Connections
* Layer Normalization
* Dropout
* Teacher Forcing
* Causal Mask
* Padding Mask
* Custom Dataset & DataLoader
* Dynamic sequence padding
* Training loop from scratch
* Gradient clipping
* Checkpoint saving/loading
* Text generation after training

---

## Dataset

Training was performed on the WikiText-2 dataset.

After preprocessing:

* ~80,000 sentences
* Vocabulary size: ~33,000 words

---

## Training

Optimizer:

* AdamW

Loss:

* CrossEntropyLoss (PAD tokens ignored)

Experiments included:

* Xavier Initialization
* Noam Learning Rate Scheduler
* Label Smoothing

During experimentation it was observed that the default PyTorch initialization produced significantly better convergence than applying Xavier initialization to all model parameters. For this educational implementation, the default initialization was therefore kept.

The Noam scheduler was also implemented and tested. However, on this small-scale educational setup, a fixed learning rate produced more stable training, so the scheduler was removed from the final version.

Best training loss:

```text
1.26
```

---

## What I Learned

This project was mainly about understanding how every part of a Transformer works.

Some of the topics explored include:

* Multi-Head Attention
* Query / Key / Value projections
* Self Attention
* Cross Attention
* Positional Encoding
* Residual Connections
* Layer Normalization
* Masking
* Teacher Forcing
* Gradient Flow
* Weight Initialization
* Learning Rate Scheduling
* Checkpointing
* Training Debugging

One of the most valuable parts of the project was debugging why the model failed to learn. Several components—including initialization strategy, scheduler configuration and checkpoint handling—were investigated before reaching stable convergence.

---

## Limitations

This project intentionally uses a simple **word-level tokenizer**.

Modern large language models typically rely on subword tokenization methods such as:

* Byte Pair Encoding (BPE)
* SentencePiece
* WordPiece

Replacing the tokenizer with BPE is planned as the next step and is expected to improve generation quality considerably.

---

## Future Improvements

* Implement Byte Pair Encoding (BPE)
* Larger training datasets
* Beam Search decoding
* Better text generation strategies
* Mixed Precision Training
* Validation set & perplexity evaluation
* Inference script
* Better checkpoint management

---

## Purpose

This repository is intended as an educational implementation for learning Transformer architectures.

The objective is readability and understanding rather than maximum training performance.

---

## References

Ashish Vaswani et al.

**Attention Is All You Need**

https://arxiv.org/abs/1706.03762

PyTorch Documentation

https://pytorch.org/

Andrej Karpathy

https://github.com/karpathy
