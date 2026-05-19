import torch                              # PyTorch core
import torch.nn as nn                    # neural network building blocks
import math                              # for positional encoding calculations
import config                            # our settings and hyperparameters
from features import prepare_data        # our data pipeline from features.py


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE 1 — RNN (Recurrent Neural Network)
# ══════════════════════════════════════════════════════════════════════════════

class CharRNN(nn.Module):
    """
    A character-level RNN. Processes text one character at a time,
    maintaining a hidden state that carries memory forward through the sequence.
    """

    def __init__(self, vocab_size):
        super(CharRNN, self).__init__()

        # Embedding layer: converts integer character indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)

        # RNN layer: processes the sequence and updates the hidden state at each step
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,   # size of each character's embedding
            hidden_size=config.HIDDEN_DIM,     # size of the hidden state
            num_layers=config.NUM_LAYERS,      # how many RNN layers stacked on top of each other
            batch_first=True,                  # input shape is (batch, sequence, features)
            dropout=config.DROPOUT             # randomly zero out neurons to prevent overfitting
        )

        # Fully connected output layer: maps hidden state to vocabulary probabilities
        self.fc = nn.Linear(config.HIDDEN_DIM, vocab_size)

    # Plain explanation: The constructor wires together three components —
    # an embedding layer that looks up character vectors, an RNN that processes
    # the sequence step by step, and a linear layer that converts the final
    # hidden state into a probability over every character in the vocabulary.
    #
    # Analogy: Think of it like a telephone game. Each player (RNN step) hears
    # the current character, updates their understanding, and whispers a summary
    # (hidden state) to the next player. The last player announces what they
    # think comes next.


    def forward(self, x, hidden=None):
        """
        Forward pass: takes a batch of sequences and returns predictions
        for the next character at every position.
        """

        # Step 1: Convert character indices to embedding vectors
        embedded = self.embedding(x)           # shape: (batch, seq_len, embedding_dim)

        # Step 2: Pass embeddings through the RNN
        output, hidden = self.rnn(embedded, hidden)  # output shape: (batch, seq_len, hidden_dim)

        # Step 3: Pass every output step through the linear layer
        logits = self.fc(output)               # shape: (batch, seq_len, vocab_size)

        return logits, hidden

    # Plain explanation: forward() is called every time we feed data through
    # the model. It embeds the input, runs it through the RNN to get a hidden
    # state at every position, then converts each hidden state into a prediction
    # over the vocabulary.
    #
    # Analogy: It's like reading a sentence out loud and pausing after each word
    # to guess what comes next. The RNN makes a guess at every single position,
    # not just at the end.


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE 2 — LSTM (Long Short-Term Memory)
# ══════════════════════════════════════════════════════════════════════════════

class CharLSTM(nn.Module):
    """
    A character-level LSTM. Similar to the RNN but with a more sophisticated
    memory mechanism that can remember things over much longer distances.
    """

    def __init__(self, vocab_size):
        super(CharLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)

        # LSTM layer: like RNN but maintains TWO states —
        # hidden state (short-term) and cell state (long-term)
        self.lstm = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT
        )

        self.fc = nn.Linear(config.HIDDEN_DIM, vocab_size)

    # Plain explanation: Almost identical to CharRNN, but uses nn.LSTM instead
    # of nn.RNN. The key difference is that LSTMs carry two pieces of state:
    # a hidden state (recent memory) and a cell state (long-term memory).
    #
    # Analogy: A basic RNN is like a person who can only remember the last few
    # things they heard. An LSTM is like a person with a notepad — they can
    # jot down important things to remember later, and erase things that are
    # no longer relevant.


    def forward(self, x, hidden=None):
        """
        Forward pass for the LSTM. Returns logits and the updated hidden state.
        Note: LSTM hidden state is a tuple of (hidden, cell) instead of just hidden.
        """

        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)  # hidden is now a tuple: (h, c)
        logits = self.fc(output)
        return logits, hidden

    # Plain explanation: Same flow as the RNN forward pass, but the hidden
    # state returned is a tuple of two tensors — (hidden_state, cell_state).
    # The cell state is the LSTM's long-term memory.
    #
    # Analogy: After reading each character, the LSTM updates both its
    # short-term impression (hidden state) and its long-term notes (cell state).
    # It's like a reader who highlights important passages AND keeps a summary
    # journal at the same time.


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE 3 — TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the embeddings so the Transformer knows
    the order of characters in the sequence.

    Transformers process all characters simultaneously (unlike RNNs which go
    step by step), so they need to be explicitly told the position of each one.
    """

    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, embedding_dim) filled with zeros
        pe = torch.zeros(max_len, embedding_dim)

        # Create a column vector of positions: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute the divisor term used in the sine/cosine formula
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            (-math.log(10000.0) / embedding_dim)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # Add a batch dimension and register as a buffer (not a learned parameter)
        pe = pe.unsqueeze(0)                           # shape: (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)

    # Plain explanation: Creates a fixed pattern of sine and cosine waves that
    # encodes the position of each character. Position 0 gets one pattern,
    # position 1 gets a slightly different pattern, and so on. These patterns
    # are added to the character embeddings so the model knows where each
    # character sits in the sequence.
    #
    # Analogy: Imagine numbering the seats in a theatre. The characters
    # (embeddings) are the people, and positional encoding is the seat number
    # printed on their ticket. Without it, everyone would just mill around
    # and the Transformer wouldn't know who sits where.


    def forward(self, x):
        """Adds positional encoding to the input embeddings."""
        # Add the positional encoding for the first seq_len positions
        x = x + self.pe[:, :x.size(1), :]
        return x

    # Plain explanation: Takes the character embeddings and adds the positional
    # pattern to each one. The result is an embedding that carries both
    # "what character am I" and "where am I in the sequence."
    #
    # Analogy: It's like stamping each letter in a word with its position —
    # the 'e' in position 3 gets a different stamp than the 'e' in position 7,
    # even though they're the same character.


class CharTransformer(nn.Module):
    """
    A character-level Transformer. Unlike RNNs, it processes the entire
    sequence at once using self-attention to decide which characters
    to pay attention to when predicting the next one.
    """

    def __init__(self, vocab_size):
        super(CharTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(config.EMBEDDING_DIM)

        # One Transformer encoder layer (attention + feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,      # size of embeddings
            nhead=config.NUM_HEADS,            # number of attention heads
            dim_feedforward=config.HIDDEN_DIM, # size of the feedforward layer
            dropout=config.DROPOUT,
            batch_first=True
        )

        # Stack multiple encoder layers on top of each other
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)

        self.fc = nn.Linear(config.EMBEDDING_DIM, vocab_size)

    # Plain explanation: Wires together an embedding layer, positional encoding,
    # a stack of Transformer encoder layers (each containing self-attention and
    # a feedforward network), and a final linear layer for predictions.
    #
    # Analogy: Think of each attention head as a different editor reading the
    # same manuscript simultaneously — one tracks grammar, one tracks character
    # names, one tracks tone. They all report back, and their findings are
    # combined into one rich understanding before making a prediction.


    def _make_causal_mask(self, seq_len, device):
        """
        Creates a causal mask so the model can only attend to past characters,
        not future ones. Without this, the model would cheat by looking ahead.
        """

        # Upper triangular matrix filled with -inf above the diagonal
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask

    # Plain explanation: Generates a triangular mask that blocks the model
    # from attending to future positions. At position 5, it can see positions
    # 0-5 but not 6, 7, 8... This is critical — without it the model would
    # see the answer while trying to predict it.
    #
    # Analogy: It's like taking an exam where you can review all previous
    # questions but the upcoming questions are covered with a sheet of paper.
    # You can only use what you've already seen.


    def forward(self, x, hidden=None):
        """
        Forward pass for the Transformer.
        hidden is accepted for API consistency with RNN/LSTM but not used.
        """

        seq_len = x.size(1)

        # Step 1: Embed characters
        embedded = self.embedding(x)                   # (batch, seq_len, embedding_dim)

        # Step 2: Add positional encoding
        embedded = self.pos_encoding(embedded)

        # Step 3: Build the causal mask
        mask = self._make_causal_mask(seq_len, x.device)

        # Step 4: Pass through Transformer layers
        output = self.transformer(embedded, mask=mask) # (batch, seq_len, embedding_dim)

        # Step 5: Project to vocabulary size
        logits = self.fc(output)                       # (batch, seq_len, vocab_size)

        return logits, None   # None because Transformers don't use a hidden state

    # Plain explanation: Embeds the input, adds positional info, applies the
    # causal mask so the model can't peek ahead, runs it through the Transformer
    # layers, then projects to vocabulary size. Returns None for hidden state
    # since Transformers don't carry one — they see the whole sequence at once.
    #
    # Analogy: Unlike an RNN which reads a book one word at a time and takes
    # notes as it goes, the Transformer reads the whole page at once and
    # highlights relationships between all the words simultaneously.


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

def get_model(model_type, vocab_size):
    """
    Returns the right model class based on the model_type string.
    Accepted values: 'rnn', 'lstm', 'transformer'
    """

    models = {
        'rnn':         CharRNN(vocab_size),
        'lstm':        CharLSTM(vocab_size),
        'transformer': CharTransformer(vocab_size)
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(models.keys())}")

    return models[model_type]

# Plain explanation: A simple factory function that takes a string like 'rnn'
# and returns the corresponding model object. Keeps train.py clean — instead
# of a long if/elif chain, one dictionary lookup does the job.
#
# Analogy: It's like a vending machine. You press a button ('rnn', 'lstm',
# 'transformer') and the right item comes out. You don't need to know how
# each item is made — just press the button.


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(model_type='lstm'):
    """
    Full training loop for the selected model architecture.
    Trains for NUM_EPOCHS epochs and saves the best model to disk.
    """

    print(f"\nTraining {model_type.upper()} model...")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataloader, char_to_idx, idx_to_char, vocab = prepare_data()
    vocab_size = len(vocab)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(model_type, vocab_size).to(device)  # move model to GPU if available

    # CrossEntropyLoss: measures how wrong the model's predictions are
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: adjusts model weights to reduce the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ── Model save path ───────────────────────────────────────────────────────
    save_paths = {
        'rnn':         config.RNN_MODEL_PATH,
        'lstm':        config.LSTM_MODEL_PATH,
        'transformer': config.TRANSFORMER_MODEL_PATH
    }
    save_path = save_paths[model_type]

    # ── Training ──────────────────────────────────────────────────────────────
    best_loss = float('inf')   # track the best loss seen so far

    for epoch in range(config.NUM_EPOCHS):
        model.train()          # set model to training mode
        total_loss = 0         # accumulate loss across all batches

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)    # move data to same device as model
            targets = targets.to(device)

            optimizer.zero_grad()         # clear gradients from previous step

            logits, _ = model(inputs)     # forward pass: get predictions

            # Reshape for loss calculation:
            # logits:  (batch, seq_len, vocab_size) → (batch * seq_len, vocab_size)
            # targets: (batch, seq_len)              → (batch * seq_len)
            loss = criterion(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )

            loss.backward()               # backpropagation: compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ↑ gradient clipping: prevents exploding gradients by capping their size

            optimizer.step()              # update model weights

            total_loss += loss.item()     # accumulate the loss for this batch

        # Average loss across all batches in this epoch
        avg_loss = total_loss / len(dataloader)

        # Perplexity: e^loss — the standard language model evaluation metric
        perplexity = math.exp(avg_loss)

        print(f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

        # Save the model if this is the best loss we've seen
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx':      char_to_idx,
                'idx_to_char':      idx_to_char,
                'vocab_size':       vocab_size,
                'model_type':       model_type
            }, save_path)
            print(f"  ✓ Saved best model to {save_path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Allow passing model type as a command line argument
    # e.g. python train.py lstm
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'lstm'
    train(model_type)