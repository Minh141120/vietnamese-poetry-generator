import os
import argparse
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import warnings

# Suppress TensorFlow and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
warnings.filterwarnings('ignore')  # Suppress warnings
tf.get_logger().setLevel(logging.ERROR)  # Suppress TensorFlow warnings

# Function to sample a word with temperature
def sample_with_temperature(preds, temperature=1.0):
    """Sample an index from a probability array using temperature."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(np.clip(preds, 1e-10, 1.0)) / temperature  # Avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate poetry

def generate_poem(model, seed_text, tokenizer, seq_length, length=100, temperature=0.7):
    """Generate poetry text following lục bát (6-8 syllable) structure."""
    generated_text = seed_text
    current_line = seed_text
    is_six_syllable_line = True  # Start with 6-syllable line after seed

    while len(generated_text.split()) < length:
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = token_list[-seq_length:]
        padded_sequence = pad_sequences(
            [token_list], maxlen=seq_length, padding='pre'
        )

        predictions = model.predict(padded_sequence, verbose=0)[0]
        next_index = sample_with_temperature(predictions, temperature)

        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                next_word = word
                break

        current_line_words = current_line.split()
        target_length = 6 if is_six_syllable_line else 8

        # Handle cases where not enough words are generated for a valid line
        if len(current_line_words) >= target_length:
            generated_text += "\n" + next_word  # Move to the next line
            current_line = next_word
            is_six_syllable_line = not is_six_syllable_line
        else:
            current_line += " " + next_word
            generated_text += " " + next_word

    return generated_text

# Main function
def main():
    parser = argparse.ArgumentParser(description='Generate poetry using the trained model')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--seq_length', type=int, default=50, help='Length of input sequences')
    parser.add_argument('--seed_text', required=True, help='Starting text for generation')
    parser.add_argument('--length', type=int, default=200, help='Length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')

    args = parser.parse_args()

    # Load model and tokenizer silently
    model = tf.keras.models.load_model(args.model_path, compile=False)
    tokenizer_path = os.path.join(os.path.dirname(args.model_path), 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Generate and print poem
    generated_poem = generate_poem(
        model=model,
        seed_text=args.seed_text,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        length=args.length,
        temperature=args.temperature
    )
    print(generated_poem)

if __name__ == "__main__":
    main()
