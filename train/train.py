import os
import argparse
import tensorflow as tf
from models.rnn_model import build_model
from preprocessing.data_loader import load_and_preprocess_data

def main():
    parser = argparse.ArgumentParser(description='Train the poetry generation model')
    parser.add_argument('--data_path', required=True, help='Path to the training data')
    parser.add_argument('--seq_length', type=int, default=50, help='Length of input sequences')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Dimension of embedding layer')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_path', required=True, help='Path to save the trained model')

    args = parser.parse_args()

    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Configure GPU memory growth
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print("GPU configuration issue:", e)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, tokenizer, max_len = load_and_preprocess_data(args.data_path, args.seq_length)

    # Build model
    print("Building model...")
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(
        vocab_size=vocab_size,
        seq_length=max_len,
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
            mode='min'
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

    # Save tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.model_path), 'tokenizer.pkl')
    import pickle
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    main()
