import argparse
import os
import tensorflow as tf
from preprocessing.data_loader import load_data, preprocess_text
from models.rnn_model import build_model
from tensorflow.keras.utils import to_categorical

def train_model(args):
    """Train the poetry generation model."""
    # Create directory for saved models if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    text = load_data(args.data_path)
    X, y, tokenizer, max_len = preprocess_text(text, args.seq_length)
    
    # Convert target labels to one-hot encoded format
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)
    
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
            filepath=args.model_path,  # Use model_path provided by the user
            save_best_only=True,
            monitor='loss',
            mode='min',
            save_weights_only=False  # Ensure full model saving
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
    parser.add_argument('--model_path', type=str, default='models/saved_models/final_model.keras', help='Path to save the trained model')

    args = parser.parse_args()
    train_model(args)
