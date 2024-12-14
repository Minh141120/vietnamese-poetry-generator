import argparse
import os
import tensorflow as tf
from preprocessing.data_loader import load_data, preprocess_text
from models.rnn_model import build_model

def train_model(args):
    """Train the poetry generation model."""
    # Create directory for saved models if it doesn't exist
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    text = load_data(args.data_path)
    X, y, tokenizer, max_len = preprocess_text(text, args.seq_length)
    
    # Build model
    print("Building model...")
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(
        vocab_size=vocab_size,
        seq_length=args.seq_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units
    )
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/saved_models/model_{epoch:02d}.h5',
            save_best_only=True,
            monitor='loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/saved_models/final_model.h5')
    
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
    
    args = parser.parse_args()
    train_model(args)