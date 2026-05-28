# Project 4 Complete — Final Summary

## What we built
A character-level language model trained on the complete works of Shakespeare. Given any starting prompt, the model generates text one character at a time in the style of Shakespeare. Three completely different neural network architectures were implemented, trained, and compared side by side.

---

## The seven scripts

| Script | What it does |
|---|---|
| `config.py` | Central settings file — paths, hyperparameters, model dimensions |
| `dataset.py` | Downloads Shakespeare, loads text, builds character vocabulary, creates sliding window sequences |
| `features.py` | Encodes characters as integers, wraps dataset in a DataLoader for batched training |
| `train.py` | Defines all three model architectures and the training loop |
| `evaluate.py` | Measures loss and perplexity, generates sample text, plots model comparison |
| `predict.py` | Terminal interface for generating text with custom prompts and settings |
| `app.py` | Flask API with four endpoints — health check, model list, generate, and compare |

---

## The three architectures

**RNN** — the simplest architecture. Reads one character at a time and passes a hidden state forward. Struggles with long sequences due to the vanishing gradient problem. Final perplexity: 5.43.

**LSTM** — an improved RNN with two memory streams. The cell state carries long term memory forward via addition instead of multiplication, fixing the vanishing gradient problem. Best performer after 3 epochs. Final perplexity: 4.69.

**Transformer** — the most powerful architecture. Reads the entire sequence at once and uses self-attention to connect any two positions directly. Needs significantly more training than RNN or LSTM to reach its potential. Final perplexity: 5.41.

---

## Key metrics

| Model | Final Loss | Perplexity |
|---|---|---|
| Untrained | — | ~65 |
| RNN | 1.6926 | 5.43 |
| Transformer | 1.6885 | 5.41 |
| LSTM | 1.5446 | 4.69 |

All three models went from a perplexity of ~65 (completely random) down to ~5 — meaning the models went from choosing randomly among all 65 characters to confidently narrowing it down to about 5 likely candidates at every step.

---

## Key concepts learned

- Character-level language modeling — predicting one character at a time
- Hidden states — compressed memory that carries context forward through a sequence
- Embeddings — learned vector representations of characters
- Vanishing gradients — why error signals fade over long sequences in basic RNNs
- LSTM gates — forget, input, and output gates that control long term memory
- Self-attention — how Transformers connect any two positions directly
- Causal masking — preventing the model from cheating by looking at future characters
- Positional encoding — telling the Transformer where each character sits in the sequence
- Perplexity — the standard metric for evaluating language models
- Temperature sampling — controlling the creativity vs coherence of generated text
- Model caching — loading models once and reusing them across API requests

---

## The progression across all four projects

| Project | Problem type | Model | Key concept |
|---|---|---|---|
| Project 0 | Image classification | CNN | Convolutional filters, spatial features |
| Project 1 | Binary classification | Logistic Regression + Random Forest | Tabular data, feature engineering |
| Project 2 | Sentiment analysis | TF-IDF + DAN | NLP, word embeddings, text classification |
| Project 4 | Text generation | RNN + LSTM + Transformer | Sequence modeling, language models |