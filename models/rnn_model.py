import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import TopKCategoricalAccuracy

def build_model(vocab_size, seq_length, embedding_dim=256, lstm_units=256):
    """Build and compile the RNN model for poetry generation."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=seq_length-1),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(0.3),
        LSTM(lstm_units),
        Dropout(0.3),
        Dense(lstm_units,
              activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])

    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1_accuracy'),
        TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
        TopKCategoricalAccuracy(k=10, name='top_10_accuracy')
    ]

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=metrics
    )

    return model
