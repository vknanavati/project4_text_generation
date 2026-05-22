# Project 4 — Character-Level Text Generation

## Overview
A character-level language model trained on the complete works of Shakespeare. Given a starting prompt, the model generates text one character at a time in the style of Shakespeare. Three different neural network architectures are implemented and compared: RNN, LSTM, and Transformer.

---

## Results

| Model | Final Loss | Perplexity |
|---|---|---|
| LSTM | 1.5446 | 4.69 |
| Transformer | 1.6885 | 5.41 |
| RNN | 1.6926 | 5.43 |

The LSTM outperformed both the RNN and Transformer after 3 epochs of training. The Transformer is expected to outperform both given more training epochs.

### Sample Generated Text (LSTM)
To be or not to be entertain of commild
then were breep,
And there when the man.
DUKE VINCENTIO:
Now, took on; I set I scann bear and there it grief of the time beseech a bloody heacts of your crown,--
First Servan:
Some thine add a glard ere with me to him not sho follow the father to you one resing me brided of a dear it you did by his is any Caull weet father, the my lord in warles,
As I say, as all than the duke and for the crown beseech indop not shall be, my

---

## Dataset
- **Source:** Complete works of Shakespeare (via Andrej Karpathy's char-rnn repository)
- **Size:** 1,115,394 characters
- **Vocabulary:** 65 unique characters (letters, punctuation, spaces, newlines)

---

## Models

### RNN (Recurrent Neural Network)
Processes text one character at a time, maintaining a hidden state that carries memory forward. Struggles with long range dependencies due to the vanishing gradient problem.

### LSTM (Long Short-Term Memory)
An improved RNN with two memory streams — a hidden state (short term) and a cell state (long term). The cell state allows gradients to flow further back in time, fixing the vanishing gradient problem and enabling the model to capture longer range patterns.

### Transformer
Processes the entire sequence at once using self-attention to connect any two positions directly. A causal mask prevents the model from attending to future characters. Requires more training than RNN/LSTM to reach its full potential.

---

## Project Structure

| Script | Purpose |
|---|---|
| `config.py` | Paths, hyperparameters, model settings |
| `dataset.py` | Download Shakespeare, build character vocabulary |
| `features.py` | Encode characters, build PyTorch Dataset and DataLoader |
| `train.py` | Define all three architectures and training loop |
| `evaluate.py` | Measure loss and perplexity, generate sample text, plot comparison |
| `predict.py` | Generate text from terminal with custom prompts |
| `app.py` | Flask API for text generation |

---

## Setup

```bash
git clone https://github.com/vknanavati/project4_text_generation.git
cd project4_text_generation
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy flask requests matplotlib
```

---

## Training

Train each model separately:

```bash
python train.py rnn
python train.py lstm
python train.py transformer
```

---

## Generate Text

From the terminal:

```bash
# Single model
python predict.py --model lstm --prompt "To be or not to be"

# Compare all three models
python predict.py --compare --prompt "To be or not to be"

# Custom length and temperature
python predict.py --model transformer --prompt "Shall I compare" --temperature 0.5 --length 300
```

---

## API

Start the server:

```bash
python app.py
```

Endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check if API is running |
| GET | `/models` | See which models are trained |
| POST | `/generate` | Generate text from a prompt |
| POST | `/compare` | Compare all three models |

Example request:

```bash
curl -X POST http://127.0.0.1:5002/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "To be or not to be", "model_type": "lstm"}'
```

---

## Key Concepts Learned

- Character-level language modeling
- Recurrent Neural Networks and hidden states
- Vanishing gradient problem
- Long Short-Term Memory networks and cell states
- Transformer architecture and self-attention
- Causal masking
- Perplexity as a language model evaluation metric
- Temperature sampling for text generation
- Model caching in Flask APIs

---

## Baseline Comparison

| | Perplexity |
|---|---|
| Untrained model | ~65 (random across full vocabulary) |
| RNN (3 epochs) | 5.43 |
| Transformer (3 epochs) | 5.41 |
| LSTM (3 epochs) | 4.69 |