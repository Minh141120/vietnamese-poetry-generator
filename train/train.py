import argparse
import os
import tensorflow as tf
from preprocessing.data_loader import load_data, preprocess_text
from models.rnn_model import build_model

def train_model(args):
    """Train the poetry generation model."""
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    text = load_data(args.data_path)
    if not text or len(text) == 0:
        raise ValueError(f"No data found in {args.data_path}")
    
    X, y, tokenizer, max_len = preprocess_text(text, args.seq_length)
    print(f"Data loaded. Sample input: {X[:1]}, Target: {y[:1]}")
    
    # Build model
    print("Building model...")
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}, Sequence Length: {args.seq_length}")
    model = build_model(
        vocab_size=vocab_size,
        seq_length=args.seq_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units
    )
    model.summary()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            save_best_only=True,
            monitor='loss',
            mode='min',
            save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    print(f"Training model for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train poetry generation model')
    parser.add_argument('--data_path', type=str, default='data/truyen_kieu.txt',
                      help='Path to training data')
    parser.add_argument('--seq_length', type=int, default=100,
                      help='Length of input sequences')
    parser.add_argument('--embedding_dim', type=int, default=256,
                      help='Dimension of embedding layer')
    parser.add_argument('--lstm_units', type=int, default=256,
                      help='Number of LSTM units')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Training batch size')
    parser.add_argument('--model_path', type=str, default='models/saved_models/final_model.keras',
                      help='Path to save the trained model')

    args = parser.parse_args()
    # Enable GPU memory growth
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print("GPU configuration issue:", e)

    train_model(args)
