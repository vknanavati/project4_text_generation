from flask import Flask, request, jsonify   # Flask web framework
import config                               # our settings and hyperparameters
from evaluate import load_model, generate_text  # reuse our generation logic


# ── Initialize the Flask app ──────────────────────────────────────────────────

app = Flask(__name__)

# ── Cache for loaded models ───────────────────────────────────────────────────

# Loading a model from disk is slow — we cache loaded models in this dictionary
# so each model is only loaded once, then reused for every subsequent request
loaded_models = {}

# Plain explanation: Instead of loading the model file from disk every time
# someone makes a request, we load it once and store it in memory. Every
# request after the first one just grabs it from this dictionary instantly.
#
# Analogy: It's like keeping a book on your desk instead of going to the
# library every time you need to reference it. The first trip to the library
# is slow — but after that it's right there whenever you need it.


# ── Helper: get or load a model ───────────────────────────────────────────────

def get_cached_model(model_type):
    """
    Returns a loaded model from the cache. If it hasn't been loaded yet,
    loads it from disk and stores it in the cache for future requests.
    """

    if model_type not in loaded_models:
        print(f"Loading {model_type.upper()} model into cache...")
        model, char_to_idx, idx_to_char, vocab_size = load_model(model_type)
        loaded_models[model_type] = {
            'model':       model,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'vocab_size':  vocab_size
        }
        print(f"{model_type.upper()} model loaded.")

    return loaded_models[model_type]

# Plain explanation: Checks if the requested model is already in the cache.
# If yes, returns it immediately. If no, loads it from disk, stores it in
# the cache, then returns it.
#
# Analogy: It's like a waiter checking if a dish is already prepared before
# going back to the kitchen. If it's ready, they bring it straight out.
# If not, they go to the kitchen, get it made, and bring it out — and now
# it's ready faster next time someone orders the same thing.


# ── Route 1: Health check ─────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """
    Simple health check endpoint. Returns a 200 OK if the server is running.
    Useful for confirming the API is live before sending real requests.
    """
    return jsonify({
        'status':  'ok',
        'message': 'Text generation API is running'
    })

# Plain explanation: A simple endpoint that just confirms the server is alive.
# No model loading, no generation — just a quick ping to check the API is up.
#
# Analogy: It's like knocking on someone's door to check if they're home
# before asking them a complicated question. If they answer, you know
# they're there and ready.


# ── Route 2: Generate text ────────────────────────────────────────────────────

@app.route('/generate', methods=['POST'])
def generate():
    """
    Main generation endpoint. Accepts a JSON body with:
    - prompt      (required): the starting text
    - model_type  (optional): 'rnn', 'lstm', or 'transformer' (default: 'lstm')
    - length      (optional): number of characters to generate
    - temperature (optional): sampling temperature

    Returns the generated text as JSON.

    Example request body:
    {
        "prompt": "To be or not to be",
        "model_type": "lstm",
        "length": 300,
        "temperature": 0.8
    }
    """

    # Parse the JSON body from the request
    data = request.get_json()

    # Validate that a prompt was provided
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Please provide a prompt in the request body'}), 400

    # Extract parameters from the request (use config defaults if not provided)
    prompt      = data['prompt']
    model_type  = data.get('model_type',  'lstm')
    length      = data.get('length',      config.GENERATE_LENGTH)
    temperature = data.get('temperature', config.TEMPERATURE)

    # Validate model type
    if model_type not in ['rnn', 'lstm', 'transformer']:
        return jsonify({'error': f"Invalid model_type '{model_type}'. Choose from: rnn, lstm, transformer"}), 400

    # Validate temperature
    if not (0.1 <= temperature <= 2.0):
        return jsonify({'error': 'Temperature must be between 0.1 and 2.0'}), 400

    # Validate length
    if not (1 <= length <= 2000):
        return jsonify({'error': 'Length must be between 1 and 2000'}), 400

    try:
        # Load (or retrieve from cache) the requested model
        cached = get_cached_model(model_type)

        # Generate text
        generated = generate_text(
            cached['model'],
            prompt,
            cached['char_to_idx'],
            cached['idx_to_char'],
            length=length,
            temperature=temperature
        )

        return jsonify({
            'prompt':      prompt,
            'model_type':  model_type,
            'length':      length,
            'temperature': temperature,
            'generated':   generated
        })

    except FileNotFoundError:
        return jsonify({'error': f"No trained model found for '{model_type}'. Please train it first."}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Plain explanation: The main endpoint. It reads the prompt and parameters
# from the request body, validates them, loads the right model, generates
# text, and returns everything as JSON.
#
# Analogy: It's like a custom printing press. You send in your starting
# sentence and your preferences (which model, how long, how creative),
# and the press generates the rest of the text and hands it back to you.


# ── Route 3: Compare all models ───────────────────────────────────────────────

@app.route('/compare', methods=['POST'])
def compare():
    """
    Runs the same prompt through all three models and returns their
    outputs side by side for comparison.

    Example request body:
    {
        "prompt": "To be or not to be",
        "length": 200,
        "temperature": 0.8
    }
    """

    data = request.get_json()

    if not data or 'prompt' not in data:
        return jsonify({'error': 'Please provide a prompt in the request body'}), 400

    prompt      = data['prompt']
    length      = data.get('length',      config.GENERATE_LENGTH)
    temperature = data.get('temperature', config.TEMPERATURE)

    results = {}

    for model_type in ['rnn', 'lstm', 'transformer']:
        try:
            cached = get_cached_model(model_type)
            generated = generate_text(
                cached['model'],
                prompt,
                cached['char_to_idx'],
                cached['idx_to_char'],
                length=length,
                temperature=temperature
            )
            results[model_type] = generated

        except FileNotFoundError:
            results[model_type] = f"No trained model found for '{model_type}'"

        except Exception as e:
            results[model_type] = f"Error: {str(e)}"

    return jsonify({
        'prompt':      prompt,
        'length':      length,
        'temperature': temperature,
        'results':     results
    })

# Plain explanation: Loops through all three model types, generates text
# from each using the same prompt, and bundles all three outputs into one
# JSON response. Useful for directly comparing what each architecture
# produces from the same starting point.
#
# Analogy: It's like sending the same writing prompt to three authors
# simultaneously and getting all three essays back in one envelope.


# ── Route 4: List available models ────────────────────────────────────────────

@app.route('/models', methods=['GET'])
def list_models():
    """
    Returns which models have been trained and are available for generation.
    Checks whether the saved .pt files exist on disk.
    """

    import os

    available = {}

    model_paths = {
        'rnn':         config.RNN_MODEL_PATH,
        'lstm':        config.LSTM_MODEL_PATH,
        'transformer': config.TRANSFORMER_MODEL_PATH
    }

    for model_type, path in model_paths.items():
        available[model_type] = os.path.exists(path)

    return jsonify({
        'available_models': available,
        'message': 'True means the model has been trained and is ready to use'
    })

# Plain explanation: Checks whether each model's saved .pt file exists on
# disk and reports back which ones are ready to use. Helpful for knowing
# which models you've trained before sending a generation request.
#
# Analogy: It's like checking your fridge before ordering takeout. You want
# to know what's already available before deciding what to ask for.


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Starting Text Generation API...")
    print("Available endpoints:")
    print("  GET  /health    — check if API is running")
    print("  GET  /models    — see which models are trained")
    print("  POST /generate  — generate text from a prompt")
    print("  POST /compare   — compare all three models on a prompt")
    app.run(debug=True, port=5002)