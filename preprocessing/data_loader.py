import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import re

def load_data(file_path):
    """Load text data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def clean_text(text):
    """Clean and normalize text data."""
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    return '\n'.join(lines)

def create_sequences(text, tokenizer, seq_length):
    """Create input sequences and corresponding targets."""
    input_sequences = []
    
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    # Pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    
    # Create predictors and label
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    
    # Convert labels to one-hot encoded form
    label = to_categorical(label, num_classes=len(tokenizer.word_index) + 1)
    
    return predictors, label, max_sequence_len

def preprocess_text(text, seq_length):
    """Main preprocessing function."""
    # Clean text
    cleaned_text = clean_text(text)
    
    # Create tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleaned_text.split('\n'))
    
    # Create sequences
    X, y, max_len = create_sequences(cleaned_text, tokenizer, seq_length)
    
    return X, y, tokenizer, max_len