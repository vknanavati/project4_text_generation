import torch                                      # PyTorch — our ML framework
from torch.utils.data import DataLoader           # wraps our dataset into batches
import config                                     # our config file with paths and settings
from dataset import download_shakespeare, load_text, build_vocabulary, ShakespeareDataset
# ↑ importing all the functions and classes we wrote in dataset.py


# ── Step 1: Build the full feature pipeline ────────────────────────────────────

def prepare_data():
    """
    Runs the full data preparation pipeline:
    1. Downloads the Shakespeare text (if not already on disk)
    2. Loads the text into memory
    3. Builds the character vocabulary
    4. Creates the PyTorch Dataset
    5. Wraps it in a DataLoader for batched training

    Returns everything the training script will need.
    """

    # Step 1: Make sure the data file exists on disk
    download_shakespeare()

    # Step 2: Load the raw text as one big string
    text = load_text()

    # Step 3: Build the character ↔ integer lookup tables
    vocab, char_to_idx, idx_to_char = build_vocabulary(text)

    # Step 4: Create the PyTorch Dataset (sliding window sequences)
    dataset = ShakespeareDataset(text, char_to_idx)

    # Step 5: Wrap the dataset in a DataLoader
    # - batch_size: how many sequences to process at once
    # - shuffle: mix up the order so the model doesn't memorize sequence order
    # - drop_last: if the final batch is smaller than batch_size, discard it
    #   (keeps tensor sizes consistent during training)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    print(f"Dataset size:    {len(dataset):,} sequences")
    print(f"Batches per epoch: {len(dataloader):,}")
    print(f"Vocabulary size: {len(vocab)} characters")

    return dataloader, char_to_idx, idx_to_char, vocab

# Plain explanation: This function is the master coordinator — it calls all
# the individual steps from dataset.py in the right order and bundles
# everything up into one neat package that train.py can use directly.
#
# Analogy: Think of it like a mise en place in cooking — before you start
# actually cooking, you chop all the vegetables, measure all the ingredients,
# and lay everything out in order. prepare_data() does all that prep work
# so that train.py can just focus on the cooking.


# ── Step 2: Encode a string prompt for generation ─────────────────────────────

def encode_prompt(prompt, char_to_idx):
    """
    Converts a string prompt (e.g. "To be or not") into a PyTorch tensor
    of integers so it can be fed into the model during text generation.

    Any characters in the prompt that aren't in the vocabulary are skipped.
    """

    # Convert each character to its integer index, skipping unknown characters
    encoded = [char_to_idx[ch] for ch in prompt if ch in char_to_idx]

    # Shape: (1, sequence_length) — the 1 is the batch dimension
    # Models always expect a batch dimension even when processing one sequence
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    return tensor

# Plain explanation: Takes a text prompt typed by the user and converts it
# into the integer format the model understands. The unsqueeze(0) adds a
# batch dimension because PyTorch always expects input in batches — even
# if the batch is just one item.
#
# Analogy: It's like scanning a barcode at a store. The word "Apple" on the
# label means nothing to the register — it needs the numeric barcode.
# encode_prompt() is the barcode scanner that converts human-readable text
# into machine-readable numbers.


# ── Step 3: Decode model output back into text ────────────────────────────────

def decode_output(indices, idx_to_char):
    """
    Converts a list of integer indices back into a human-readable string.
    This is used after the model generates a sequence of predicted characters.
    """

    # Look up each index in idx_to_char and join the characters into a string
    return ''.join([idx_to_char[idx] for idx in indices])

# Plain explanation: Takes a list of integers that the model produced and
# converts them back into readable text using the idx_to_char lookup table.
# It's the reverse of encode_prompt.
#
# Analogy: It's like a decoder ring — you get a sequence of numbers from
# a secret message and the ring tells you which letter each number maps to.
# String them all together and you can read the message.