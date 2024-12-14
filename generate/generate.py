import argparse
import numpy as np
import tensorflow as tf
from preprocessing.data_loader import load_data, preprocess_text

def sample_with_temperature(preds, temperature=1.0):
    """Sample an index from a probability array using temperature."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_poem(model, seed_text, tokenizer, seq_length, length=100, temperature=0.7):
    """Generate poetry text."""
    generated_text = seed_text
    for _ in range(length):
        # Tokenize and pad the current text
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = token_list[-seq_length:]
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=seq_length, padding='pre'
        )
        
        # Predict next token
        predictions = model.predict(padded_sequence, verbose=0)[0]
        next_index = sample_with_temperature(predictions, temperature)
        
        # Convert token to word
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                next_word = word
                break
        
        # Add to generated text
        generated_text += " " + next_word
        
        # Add line break for poetry format
        if len(generated_text.split('\n')[-1].split()) >= 8:  # Approximate line length
            generated_text += "\n"
    
    return generated_text

def main(args):
    # Load data and tokenizer
    text = load_data(args.data_path)
    _, _, tokenizer, _ = preprocess_text(text, args.seq_length)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # Generate poem
    print(f"\nGenerating poem with seed text: '{args.seed_text}'")
    poem = generate_poem(
        model=model,
        seed_text=args.seed_text,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        length=args.length,
        temperature=args.temperature
    )
    
    print("\nGenerated Poem:")
    print("=" * 50)
    print(poem)
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate poetry using trained model')
    parser.add_argument('--data_path', type=str, default='data/truyen_kieu.txt',
                      help='Path to training data')
    parser.add_argument('--model_path', type=str, default='models/saved_models/final_model.h5',
                      help='Path to trained model')
    parser.add_argument('--seq_length', type=int, default=100,
                      help='Sequence length for generation')
    parser.add_argument('--seed_text', type=str, default='Trăm năm trong cõi người ta',
                      help='Starting text for generation')
    parser.add_argument('--length', type=int, default=100,
                      help='Length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature')
    
    args = parser.parse_args()
    main(args)