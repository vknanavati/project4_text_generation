# Project 4 — Character-Level Language Model

## The Big Idea

We'll train a neural network to **predict the next character** in a sequence, one character at a time. Given the text `"To be or not to b"`, the model learns to predict `"e"`. Do that repeatedly and you get generated text.

This is how the earliest neural language models worked — and it's a beautiful way to understand the fundamentals before moving to word-level or token-level models (like GPT).

---

## The Learning Arc (Simple → Medium → Deep)

We'll build in three layers of complexity:

**Layer 1 — Simple (RNN)**
A basic Recurrent Neural Network. You'll understand what a "hidden state" is, how text gets fed character by character, and why sequence modeling is different from tabular or image data.

**Layer 2 — Medium (LSTM)**
Long Short-Term Memory — a smarter RNN that can remember things over longer distances. You'll learn about the "vanishing gradient problem" and why LSTMs were invented to fix it.

**Layer 3 — Deep (Transformer)**
A small version of the architecture behind GPT. You'll understand attention, positional encoding, and why Transformers took over the world. Same dataset, same task — but a fundamentally different (and more powerful) architecture.

All three models will be trained on the **same dataset**, so you can directly compare their outputs and performance.

---

## The Dataset

We'll use **Shakespeare's complete works** — a classic benchmark for character-level models, small enough to train fast, rich enough to produce interesting output.

---

## The 7-Script Structure

| Script | Purpose |
|---|---|
| `config.py` | Paths, hyperparameters, model settings |
| `dataset.py` | Load text, build character vocabulary, create sequences |
| `features.py` | Encode characters as integers, build PyTorch Dataset |
| `train.py` | Train whichever model is selected (RNN / LSTM / Transformer) |
| `evaluate.py` | Measure loss, perplexity, show sample generations |
| `predict.py` | Generate text from a prompt |
| `app.py` | Flask API — POST a prompt, get generated text back |

---

## What You'll Walk Away Knowing

- How sequence models process text character by character
- What hidden states, LSTMs, and attention actually *do* (not just that they exist)
- How to compare three architectures on the same task
- A new metric: **perplexity** (the standard way to evaluate language models)
- How to build a text generation API