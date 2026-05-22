import os  # lets us interact with the file system and build file paths

# ── Paths ──────────────────────────────────────────────────────────────────────

# The root of the project — the folder this config lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where we'll store the raw Shakespeare text file
DATA_DIR = os.path.join(BASE_DIR, "data")

# Where trained model files will be saved
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Where evaluation plots and outputs will be saved
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# The actual Shakespeare text file
DATA_FILE = os.path.join(DATA_DIR, "shakespeare.txt")

# One saved model file per architecture
RNN_MODEL_PATH = os.path.join(MODELS_DIR, "rnn_model.pt")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")
TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, "transformer_model.pt")

# ── Sequence settings ──────────────────────────────────────────────────────────

# How many characters the model sees at once before making a prediction
SEQ_LENGTH = 50 # was 100
# ── Training settings ──────────────────────────────────────────────────────────

BATCH_SIZE = 256       # how many sequences to process at once.
NUM_EPOCHS = 3      # how many times to loop through the full dataset
LEARNING_RATE = 0.001  # how big a step the optimizer takes each update

# ── Model architecture settings ───────────────────────────────────────────────

EMBEDDING_DIM = 64     # size of the vector representing each character
HIDDEN_DIM = 64     # size of the hidden state in RNN/LSTM
NUM_LAYERS = 1         # how many stacked RNN/LSTM layers to use

# Transformer-specific settings
NUM_HEADS = 4          # number of attention heads in the Transformer
DROPOUT = 0.0          # fraction of neurons randomly turned off during training (prevents overfitting)

# ── Generation settings ────────────────────────────────────────────────────────

# How many characters to generate when producing sample output
GENERATE_LENGTH = 500

# Controls randomness of generation:
# lower = more predictable, higher = more creative/random
TEMPERATURE = 0.8

# ── Create directories if they don't exist yet ─────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)     # create data/ folder
os.makedirs(MODELS_DIR, exist_ok=True)   # create models/ folder
os.makedirs(OUTPUTS_DIR, exist_ok=True)  # create outputs/ folder