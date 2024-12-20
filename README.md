# 🎨 Vietnamese Poetry Generator - Truyen Kieu RNN

A sophisticated RNN-based poetry generator trained on the classical Vietnamese epic poem "Truyện Kiều" by Nguyễn Du. This model uses deep learning to generate new poetry in the style of this masterpiece, achieving up to 88% accuracy with the larger model configuration.

## 📋 Table of Contents
- [Features](#-features)
- [Requirements](#-requirements)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Training Results](#-training-results)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features
- Bidirectional LSTM architecture with multiple layers
- Temperature-based text generation for creativity control
- Comprehensive preprocessing for Vietnamese text
- Support for both standard and large model configurations
- Automatic handling of lục bát (6-8 syllable) structure
- Checkpoint saving and early stopping
- Top-k accuracy metrics tracking

## 📦 Requirements
- Python 3.8+
- TensorFlow 2.4+
- NumPy 1.19+
- tqdm
- pickle (for tokenizer saving/loading)

## 🗂️ Project Structure
```
poetry_generator/
├── data/
│   └── truyen_kieu.txt          # Dataset
├── generate/
│   └── generate.py              # Generation script
├── models/
│   ├── saved_models/           # Trained models
│   └── rnn_model.py            # Model architecture
├── preprocessing/
│   ├── __init__.py
│   └── data_loader.py          # Data preprocessing
├── train/
│   └── train.py               # Training script
├── utils/
│   └── text_utils.py          # Utility functions
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Minh141120/vietnamese-poetry-generator.git
cd vietnamese-poetry-generator
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Standard Model Training (45% accuracy)
```bash
python train/train.py \
    --data_path data/truyen_kieu.txt \
    --seq_length 50 \
    --embedding_dim 256 \
    --lstm_units 256 \
    --epochs 50 \
    --batch_size 32 \
    --model_path models/saved_models/poetry_model.keras
```

### Large Model Training (88% accuracy)
```bash
python train/train.py \
    --data_path data/truyen_kieu.txt \
    --seq_length 100 \
    --embedding_dim 512 \
    --lstm_units 512 \
    --epochs 100 \
    --batch_size 64 \
    --model_path models/saved_models/poetry_model_large.keras
```

### Generating Poetry

Using standard model:
```bash
python generate/generate.py \
    --model_path models/saved_models/poetry_model.keras \
    --seq_length 50 \
    --seed_text "Trăm năm trong cõi người ta" \
    --length 200 \
    --temperature 0.7
```

Using large model:
```bash
python generate/generate.py \
    --model_path models/saved_models/poetry_model_large.keras \
    --seed_text "Trăm năm trong cõi người ta" \
    --length 200 \
    --seq_length 100 \
    --temperature 0.7
```

## 🏗️ Model Architecture

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding                   (None, seq_len, dim)      vocab_size * dim
bidirectional_lstm_1       (None, seq_len, dim*2)    4 * dim * dim
dropout_1                  (None, seq_len, dim*2)    0
bidirectional_lstm_2       (None, seq_len, dim*2)    4 * dim * dim
dropout_2                  (None, seq_len, dim*2)    0
lstm                       (None, dim)               4 * dim * dim
dropout_3                  (None, dim)               0
dense_1                    (None, dim)               dim * dim
dropout_4                  (None, dim)               0
dense_2                    (None, vocab_size)        dim * vocab_size
=================================================================
```

Where `dim` is either 256 (standard) or 512 (large model).

## 📊 Training Results

Standard Model (256 units):
- Sequence Length: 50
- Accuracy: ~45%
- Training Time: ~2 hours on GPU
- Model Size: ~100MB

Large Model (512 units):
- Sequence Length: 100
- Accuracy: ~88%
- Training Time: ~6 hours on GPU
- Model Size: ~400MB

## ⚠️ Troubleshooting

Common issues:
1. Memory errors: Reduce batch size or sequence length
2. Slow training: Enable GPU support or use smaller model
3. Poor generation quality: Try adjusting temperature (0.7 works well)
4. Tokenizer errors: Ensure proper UTF-8 encoding of input text
