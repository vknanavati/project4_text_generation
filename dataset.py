import os                  # for checking if files exist on disk
import requests            # for downloading the Shakespeare text from the internet
import torch               # PyTorch — our ML framework
from torch.utils.data import Dataset  # base class we inherit from to build a custom dataset
import config              # our config file with paths and settings


# ── Step 1: Download the data ──────────────────────────────────────────────────

def download_shakespeare():
    """
    Downloads the complete works of Shakespeare as a plain text file
    and saves it to the path defined in config.py.
    If the file already exists, skips the download.
    """

    # The URL where the Shakespeare text lives (hosted by Andrej Karpathy)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # Only download if we don't already have the file
    if not os.path.exists(config.DATA_FILE):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)               # fetch the file from the internet
        with open(config.DATA_FILE, 'w') as f:     # open our local file for writing
            f.write(response.text)                 # write the downloaded text into it
        print(f"Saved to {config.DATA_FILE}")
    else:
        print("Shakespeare dataset already exists, skipping download.")

# Plain explanation: This function checks if the Shakespeare text file is already
# on disk. If not, it downloads it from the internet and saves it locally.
#
# Analogy: It's like checking your pantry before going to the grocery store.
# If you already have flour, you don't buy more — you only make the trip if
# you actually need something.


# ── Step 2: Load and inspect the text ─────────────────────────────────────────

def load_text():
    """
    Reads the Shakespeare text file from disk and returns it as a single string.
    """

    with open(config.DATA_FILE, 'r') as f:  # open the file in read mode
        text = f.read()                     # read the entire file into one big string

    print(f"Total characters in dataset: {len(text):,}")  # e.g. 1,115,394
    return text

# Plain explanation: Opens the saved text file and loads it entirely into memory
# as one long string. We also print how many characters are in it so we can
# get a sense of the dataset size.
#
# Analogy: It's like picking up a book and flipping to the last page to see
# how many pages it has before you start reading.


# ── Step 3: Build the vocabulary ──────────────────────────────────────────────

def build_vocabulary(text):
    """
    Finds every unique character in the text and creates two lookup tables:
    - char_to_idx: maps each character to a unique integer  ('a' → 0)
    - idx_to_char: maps each integer back to a character    (0 → 'a')
    """

    # sorted() gives us a consistent ordering every time we run the script
    vocab = sorted(set(text))   # set() finds all unique characters, sorted() orders them

    # Dictionary comprehension: loop over vocab and assign each character an index
    char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}

    # Reverse lookup: flip the dictionary so we can go from index back to character
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

    print(f"Vocabulary size: {len(vocab)} unique characters")
    return vocab, char_to_idx, idx_to_char

# Plain explanation: Scans the entire text, finds every unique character
# (letters, punctuation, spaces, newlines), and builds two dictionaries —
# one to convert characters to numbers, and one to convert numbers back
# to characters.
#
# Analogy: It's like building a codebook for a secret message. One side of
# the book says "A = 1, B = 2, C = 3..." and the other side is the reverse
# so you can decode messages too. The model uses the first side to read text
# and the second side to write it.


# ── Step 4: Build the PyTorch Dataset ─────────────────────────────────────────

class ShakespeareDataset(Dataset):
    """
    A custom PyTorch Dataset that slices the Shakespeare text into
    overlapping sequences of length SEQ_LENGTH.

    Each item in the dataset is a pair:
    - input:  a sequence of SEQ_LENGTH characters  e.g. "To be or not to b"
    - target: the same sequence shifted one step    e.g. "o be or not to be"

    The model learns: given this input sequence, predict the target sequence.
    """

    def __init__(self, text, char_to_idx):
        self.char_to_idx = char_to_idx   # store the lookup table
        self.seq_length = config.SEQ_LENGTH

        # Convert the entire text into a list of integers using char_to_idx
        self.encoded = [char_to_idx[ch] for ch in text]

    # Plain explanation: The constructor takes the raw text and the character
    # lookup table, then converts the entire text into a sequence of integers
    # and stores it. This encoded version is what the model will actually see.
    #
    # Analogy: It's like translating an entire novel into Morse code before
    # sending it — you do the translation once upfront, then work with the
    # coded version from that point on.


    def __len__(self):
        """Returns the total number of sequences we can extract from the text."""
        # We subtract SEQ_LENGTH because the last sequence needs a target too
        return len(self.encoded) - self.seq_length

    # Plain explanation: Tells PyTorch how many training examples exist in
    # this dataset. Each example is a window of SEQ_LENGTH characters sliding
    # through the text one step at a time.
    #
    # Analogy: Imagine sliding a 100-character window across the full text,
    # moving one character at a time. __len__ counts how many positions
    # that window can land on before it falls off the end.


    def __getitem__(self, idx):
        """
        Returns one training example: an input sequence and its target sequence.
        The target is the input shifted forward by one character.
        """

        # Slice out a chunk of SEQ_LENGTH characters starting at position idx
        input_seq = self.encoded[idx : idx + self.seq_length]

        # Target is the same window shifted one step to the right
        target_seq = self.encoded[idx + 1 : idx + self.seq_length + 1]

        # Convert to PyTorch tensors (the data format PyTorch works with)
        return torch.tensor(input_seq, dtype=torch.long), \
               torch.tensor(target_seq, dtype=torch.long)

    # Plain explanation: For any position idx in the text, this returns the
    # input sequence (characters at positions idx to idx+100) and the target
    # sequence (characters at positions idx+1 to idx+101). The model's job
    # is to predict the target from the input.
    #
    # Analogy: It's like a typing tutor that shows you a sentence with the
    # last character hidden, and asks you to guess it. Then it shifts the
    # window one character to the right and asks again. Every possible window
    # position is one training example.

    # ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    download_shakespeare()
    text = load_text()
    vocab, char_to_idx, idx_to_char = build_vocabulary(text)