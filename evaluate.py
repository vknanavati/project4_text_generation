import torch                          # PyTorch core
import torch.nn as nn                 # neural network building blocks
import math                           # for perplexity calculation
import matplotlib.pyplot as plt       # for plotting
import os                             # for file path operations
import config                         # our settings and hyperparameters
from features import prepare_data, decode_output    # data pipeline and decoder
from train import get_model                         # model factory function


# ── Step 1: Load a saved model from disk ──────────────────────────────────────

def load_model(model_type):
    """
    Loads a trained model and its associated vocabulary from disk.
    Returns the model, char_to_idx, idx_to_char, and vocab_size.
    """

    # Map model type to the correct saved file path
    save_paths = {
        'rnn':         config.RNN_MODEL_PATH,
        'lstm':        config.LSTM_MODEL_PATH,
        'transformer': config.TRANSFORMER_MODEL_PATH
    }

    save_path = save_paths[model_type]

    # Load the saved checkpoint dictionary from disk
    checkpoint = torch.load(save_path, map_location='cpu')
    # ↑ map_location='cpu' ensures it loads correctly even if trained on GPU

    # Rebuild the model architecture
    vocab_size  = checkpoint['vocab_size']
    model       = get_model(model_type, vocab_size)

    # Load the learned weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode (disables dropout)
    model.eval()

    return model, checkpoint['char_to_idx'], checkpoint['idx_to_char'], vocab_size

# Plain explanation: Reads the saved .pt file from disk, rebuilds the model
# architecture, loads the learned weights into it, and returns everything
# needed to use the model.
#
# Analogy: It's like reopening a book you were reading. The architecture is
# the blank book — same structure, same number of pages. The weights are the
# words written on those pages. load_state_dict() is the act of copying all
# those words back in so you can continue where you left off.


# ── Step 2: Calculate loss and perplexity on the dataset ──────────────────────

def evaluate_model(model, dataloader, vocab_size):
    """
    Runs the model over the entire dataset without updating weights,
    and returns the average loss and perplexity.
    """

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss   = 0
    total_batches = 0

    # torch.no_grad() tells PyTorch not to track gradients — we're evaluating,
    # not training, so we don't need to compute or store them
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)

            # Reshape for loss calculation (same as in train.py)
            loss = criterion(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )

            total_loss    += loss.item()
            total_batches += 1

    avg_loss   = total_loss / total_batches
    perplexity = math.exp(avg_loss)   # perplexity = e^loss

    return avg_loss, perplexity

# Plain explanation: Feeds the entire dataset through the model, measures
# how wrong its predictions are at every step, and returns the average loss
# and perplexity. No weight updates happen here — we're purely measuring
# performance.
#
# Analogy: It's like giving a student a mock exam after they've finished
# studying. You're not teaching them anything new — you're just measuring
# how well they learned. torch.no_grad() is like telling the student
# "this doesn't count toward your grade, just show me what you know."


# ── Step 3: Generate sample text from a prompt ────────────────────────────────

def generate_text(model, prompt, char_to_idx, idx_to_char, length=None, temperature=None):
    """
    Generates text character by character starting from a prompt string.

    Temperature controls randomness:
    - Low  (e.g. 0.2): safe and repetitive
    - High (e.g. 1.5): creative but potentially chaotic
    - 0.8 (our default): a good balance
    """

    if length      is None: length      = config.GENERATE_LENGTH
    if temperature is None: temperature = config.TEMPERATURE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # Encode the prompt into a tensor of integers
    # Filter out any characters not in our vocabulary
    input_indices = [char_to_idx[ch] for ch in prompt if ch in char_to_idx]
    input_tensor  = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    # ↑ unsqueeze(0) adds the batch dimension: (seq_len,) → (1, seq_len)

    generated = list(input_indices)   # start with the encoded prompt
    hidden    = None                  # hidden state starts empty

    with torch.no_grad():
        for _ in range(length):

            # Forward pass through the model
            logits, hidden = model(input_tensor, hidden)

            # Take only the last position's logits (the most recent prediction)
            last_logits = logits[0, -1, :]          # shape: (vocab_size,)

            # Apply temperature scaling
            # Dividing by temperature sharpens or flattens the distribution
            scaled_logits = last_logits / temperature

            # Convert logits to probabilities using softmax
            probs = torch.softmax(scaled_logits, dim=0)

            # Sample from the probability distribution
            # (rather than always picking the highest probability character)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_idx)

            # Update input tensor to just the newly generated character
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(device)

            # For RNN/LSTM, carry the hidden state forward
            # For Transformer, hidden is None so this has no effect
            if isinstance(hidden, tuple):
                hidden = tuple(h.detach() for h in hidden)
            elif hidden is not None:
                hidden = hidden.detach()

    # Decode the full generated sequence back into a string
    return decode_output(generated, idx_to_char)

# Plain explanation: Starts with the encoded prompt, then repeatedly feeds
# the model one character at a time, samples the next character from the
# predicted probability distribution, appends it to the sequence, and
# repeats until the desired length is reached.
#
# Analogy: It's like autocomplete on your phone, but taken to an extreme.
# Each time you accept a suggestion, it becomes the new input for the next
# suggestion. Do that 500 times in a row and you have generated text.


# ── Step 4: Compare all three models ──────────────────────────────────────────

def compare_models(prompt="To be or not to be"):
    """
    Loads all three trained models, evaluates them, generates sample text
    from each, and prints a side-by-side comparison.
    """

    dataloader, _, _, _ = prepare_data()
    model_types = ['rnn', 'lstm', 'transformer']
    results     = {}

    for model_type in model_types:
        print(f"\nEvaluating {model_type.upper()}...")

        try:
            model, char_to_idx, idx_to_char, vocab_size = load_model(model_type)

            # Evaluate loss and perplexity
            avg_loss, perplexity = evaluate_model(model, dataloader, vocab_size)

            # Generate sample text
            generated = generate_text(model, prompt, char_to_idx, idx_to_char)

            results[model_type] = {
                'loss':       avg_loss,
                'perplexity': perplexity,
                'generated':  generated
            }

            print(f"  Loss:       {avg_loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Generated:  {generated[:200]}...")  # show first 200 chars

        except FileNotFoundError:
            print(f"  No saved model found for {model_type} — skipping.")

    return results


# ── Step 5: Plot perplexity comparison ────────────────────────────────────────

def plot_comparison(results):
    """
    Creates a bar chart comparing the perplexity of each trained model
    and saves it to the outputs folder.
    """

    model_types  = list(results.keys())
    perplexities = [results[m]['perplexity'] for m in model_types]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_types, perplexities, color=['#4C72B0', '#DD8452', '#55A868'])

    # Add the perplexity value on top of each bar
    for bar, perp in zip(bars, perplexities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{perp:.1f}',
            ha='center', va='bottom', fontsize=12
        )

    plt.title('Model Comparison — Perplexity (lower is better)')
    plt.xlabel('Model Architecture')
    plt.ylabel('Perplexity')
    plt.tight_layout()

    # Save to outputs folder
    save_path = os.path.join(config.OUTPUTS_DIR, 'model_comparison.png')
    plt.savefig(save_path)
    print(f"\nComparison plot saved to {save_path}")
    plt.close()

# Plain explanation: Takes the results dictionary from compare_models(),
# extracts the perplexity for each model, and draws a bar chart so you
# can visually compare how well each architecture performed.
#
# Analogy: It's like a sports scoreboard — instead of reading through
# a table of numbers, you get a visual snapshot of who won and by how much.


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = compare_models()

    if results:
        plot_comparison(results)