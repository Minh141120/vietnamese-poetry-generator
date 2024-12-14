from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import regularizers

def build_model(vocab_size, seq_length, embedding_dim=100):
    """
    Build and compile the RNN model for poetry generation.
    
    Args:
        vocab_size (int): Size of vocabulary (total unique words)
        seq_length (int): Length of input sequences
        embedding_dim (int): Dimension of embedding layer, default 100
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        # Embedding layer - converts words to dense vectors
        Embedding(vocab_size, embedding_dim, input_length=seq_length-1),
        
        # First Bidirectional LSTM layer with more units
        Bidirectional(LSTM(150, return_sequences=True)),
        Dropout(0.2),
        
        # Second LSTM layer with fewer units for dimensionality reduction
        LSTM(100),
        
        # Intermediate dense layer with half vocabulary size
        Dense(vocab_size//2, 
              activation='relu',
              kernel_regularizer=regularizers.l2(0.01)),
        
        # Output layer with full vocabulary size
        Dense(vocab_size, activation='softmax')
    ])
    
    # Compile model with categorical crossentropy
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