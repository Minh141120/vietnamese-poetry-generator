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
    # Remove line numbers and dots
    text = re.sub(r'^\d+\.\.', '', text, flags=re.MULTILINE)
    
    # Split into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    return lines

def create_luc_bat_sequences(lines, seq_length=4):
    """Create sequences of lục bát format (groups of 4 lines)."""
    sequences = []
    
    # Process 4 lines at a time
    for i in range(0, len(lines)-3, 2):  # Step by 2 to keep pairs together
        four_lines = lines[i:i+4]
        if len(four_lines) == 4:  # Only use complete sets of 4 lines
            sequence = ' '.join(four_lines)
            sequences.append(sequence)
    
    return sequences

def create_training_sequences(sequences, tokenizer, seq_length):
    """Create training sequences from lục bát groups."""
    input_sequences = []
    
    for sequence in sequences:
        token_list = tokenizer.texts_to_sequences([sequence])[0]
        
        # Create subsequences while maintaining at least 4 lines context
        min_length = 20  # Approximate minimum length for 4 lines
        for i in range(min_length, len(token_list)):
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
    """Main preprocessing function for lục bát poetry."""
    # Clean text and get lines
    lines = clean_text(text)
    
    # Create lục bát sequences (groups of 4 lines)
    sequences = create_luc_bat_sequences(lines, seq_length)
    
    # Create tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    
    # Create training sequences
    X, y, max_len = create_training_sequences(sequences, tokenizer, seq_length)
    
    return X, y, tokenizer, max_len