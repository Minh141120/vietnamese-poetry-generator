from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import regularizers

def build_model(vocab_size, seq_length, embedding_dim=100, lstm_units=150):
    """
    Build and compile the RNN model for poetry generation.
    
    Args:
        vocab_size (int): Size of vocabulary (total unique words)
        seq_length (int): Length of input sequences
        embedding_dim (int): Dimension of embedding layer, default 100
        lstm_units (int): Number of LSTM units in the layers, default 150
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        # Embedding layer - converts words to dense vectors
        Embedding(vocab_size, embedding_dim, input_length=seq_length-1),
        
        # First Bidirectional LSTM layer with configurable units
        # Bidirectional LSTMs improve context capture in both directions
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(0.2),  # Dropout to prevent overfitting
        
        # Second LSTM layer with fewer units for dimensionality reduction
        # Reducing the number of units helps in better feature selection
        LSTM(lstm_units // 2),
        
        # Intermediate dense layer with half vocabulary size
        # Activation function 'relu' adds non-linearity for complex mappings
        # L2 regularization helps prevent overfitting by penalizing large weights
        Dense(vocab_size // 2, 
              activation='relu',
              kernel_regularizer=regularizers.l2(0.01)),
        
        # Output layer with full vocabulary size
        # Softmax activation to generate probability distribution over vocabulary
        Dense(vocab_size, activation='softmax')
    ])
    
    # Compile model with categorical crossentropy
    # Adam optimizer adapts learning rates for efficient training
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


# Optional function to print model architecture
def print_model_summary(vocab_size, seq_length):
    """Print summary of model architecture."""
    model = build_model(vocab_size, seq_length)
    print(model.summary())
    return model