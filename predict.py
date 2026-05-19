import torch                          # PyTorch core
import argparse                       # for parsing command line arguments
import config                         # our settings and hyperparameters
from evaluate import load_model, generate_text   # reuse what we already built


# ── Step 1: Load model and generate text from a prompt ────────────────────────

def predict(model_type, prompt, length=None, temperature=None):
    """
    Loads a trained model and generates text from a given prompt.

    Args:
        model_type:  'rnn', 'lstm', or 'transformer'
        prompt:      the starting text to generate from (e.g. "To be or not")
        length:      how many characters to generate (defaults to config value)
        temperature: randomness of generation (defaults to config value)
    """

    if length      is None: length      = config.GENERATE_LENGTH
    if temperature is None: temperature = config.TEMPERATURE

    print(f"\nModel:       {model_type.upper()}")
    print(f"Prompt:      {prompt}")
    print(f"Length:      {length} characters")
    print(f"Temperature: {temperature}")
    print("-" * 60)

    # Load the trained model and vocabulary from disk
    model, char_to_idx, idx_to_char, vocab_size = load_model(model_type)

    # Generate text using the model
    generated = generate_text(
        model,
        prompt,
        char_to_idx,
        idx_to_char,
        length=length,
        temperature=temperature
    )

    print(generated)
    print("-" * 60)

    return generated

# Plain explanation: The main prediction function. It loads the saved model,
# passes the prompt through generate_text() from evaluate.py, and prints
# the result. It's a clean wrapper that ties everything together.
#
# Analogy: It's like a vending machine interface. You press the buttons
# (model type, prompt, temperature) and the machine handles all the internal
# mechanics to give you your output. You don't need to know what's happening
# inside — you just get your snack.


# ── Step 2: Compare outputs across all three models ───────────────────────────

def compare_predictions(prompt, length=None, temperature=None):
    """
    Runs the same prompt through all three models and prints the outputs
    side by side so you can directly compare what each architecture generates.
    """

    if length      is None: length      = config.GENERATE_LENGTH
    if temperature is None: temperature = config.TEMPERATURE

    model_types = ['rnn', 'lstm', 'transformer']

    print(f"\n{'='*60}")
    print(f"Comparing all models on prompt: '{prompt}'")
    print(f"{'='*60}")

    for model_type in model_types:
        try:
            predict(model_type, prompt, length=length, temperature=temperature)
        except FileNotFoundError:
            print(f"\n{model_type.upper()}: No saved model found — skipping.")

# Plain explanation: Loops through all three model types and calls predict()
# on each one with the same prompt. This makes it easy to visually compare
# how the RNN, LSTM, and Transformer each continue the same starting text.
#
# Analogy: It's like giving the same writing prompt to three different authors
# and reading their responses side by side. Same starting sentence, three
# completely different continuations — each reflecting the author's style
# and capability.


# ── Step 3: Command line interface ────────────────────────────────────────────

if __name__ == '__main__':
    """
    Allows predict.py to be run directly from the terminal with arguments.

    Examples:
        python predict.py --model lstm --prompt "To be or not"
        python predict.py --model transformer --prompt "Shall I compare" --temperature 0.5
        python predict.py --compare --prompt "What is love"
    """

    # argparse lets us accept named arguments from the terminal
    parser = argparse.ArgumentParser(description='Generate text using a trained character-level model')

    # --model: which architecture to use (default: lstm)
    parser.add_argument('--model',       type=str,   default='lstm',
                        choices=['rnn', 'lstm', 'transformer'],
                        help='Model architecture to use')

    # --prompt: the starting text
    parser.add_argument('--prompt',      type=str,   default='To be or not to be',
                        help='Starting text for generation')

    # --length: how many characters to generate
    parser.add_argument('--length',      type=int,   default=config.GENERATE_LENGTH,
                        help='Number of characters to generate')

    # --temperature: controls randomness
    parser.add_argument('--temperature', type=float, default=config.TEMPERATURE,
                        help='Sampling temperature (lower=safer, higher=creative)')

    # --compare: if this flag is present, run all three models
    parser.add_argument('--compare',     action='store_true',
                        help='Compare output from all three models')

    args = parser.parse_args()

    if args.compare:
        compare_predictions(args.prompt, length=args.length, temperature=args.temperature)
    else:
        predict(args.model, args.prompt, length=args.length, temperature=args.temperature)

# Plain explanation: Sets up a command line interface so predict.py can be
# called directly from the terminal with named flags. argparse handles
# reading and validating those flags automatically.
#
# Analogy: It's like a drive-through menu. You pull up and say "I'll have
# the LSTM with a 'To be or not to be' prompt and low temperature please."
# argparse is the speaker box that takes your order and passes it to the
# kitchen.