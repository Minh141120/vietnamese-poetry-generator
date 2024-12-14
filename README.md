# 🎨 Vietnamese Poetry Generator - Truyen Kieu RNN

A sophisticated RNN-based poetry generator trained on the classical Vietnamese epic poem "Truyện Kiều" by Nguyễn Du. This model uses deep learning to generate new poetry in the style of this masterpiece.

## 📋 Table of Contents
- [Features](#-features)
- [Requirements](#-requirements)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Training Tips](#-training-tips)
- [Results & Examples](#-results--examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features
- Advanced Bidirectional LSTM architecture
- Temperature-based text generation for creativity control
- Comprehensive preprocessing for Vietnamese text
- Modular and maintainable codebase
- Configurable model parameters
- Checkpoint saving for best models

## 📦 Requirements
- Python 3.8+
- TensorFlow 2.4+
- NumPy 1.19+
- tqdm

## 🗂️ Project Structure
```
poetry_generator/
├── data/
│   └── truyen_kieu.txt          # Dataset
├── models/
│   ├── saved_models/           # Trained models
│   └── rnn_model.py            # Model architecture
├── preprocessing/
│   ├── __init__.py
│   └── data_loader.py          # Data preprocessing
├── train/
│   └── train.py               # Training script
├── generate/
│   └── generate.py            # Generation script
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
export PYTHONPATH=$(pwd)
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your dataset:
- Put `truyen_kieu.txt` in the `data/` directory
- Ensure the text is UTF-8 encoded

## 💻 Usage

### Training the Model

Basic training:
```bash
python train/train.py
```

Advanced training with custom parameters:
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

Available training parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| --data_path | data/truyen_kieu.txt | Path to dataset |
| --seq_length | 100 | Sequence length for training |
| --embedding_dim | 256 | Embedding dimension |
| --lstm_units | 256 | Number of LSTM units |
| --epochs | 50 | Number of training epochs |
| --batch_size | 64 | Batch size |
| --model_path | models/saved_models/poetry_model.h5 | Path to save model |

### Generating Poetry

Basic generation:
```bash
python generate/generate.py
```

Custom generation:
```bash
python generate/generate.py \
    --seed_text "Trăm năm trong cõi người ta" \
    --length 200 \
    --temperature 0.7
```

Generation parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| --seed_text | "Trăm năm trong cõi người ta" | Starting text |
| --length | 100 | Length of generated text |
| --temperature | 0.7 | Sampling creativity (0.1-1.0) |

## 🏗️ Model Architecture

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
embedding (Embedding)       (None, seq_len, 256)      vocab_size * 256
bidirectional_lstm         (None, seq_len, 512)      1,050,624
dropout (Dropout)          (None, seq_len, 512)      0
lstm (LSTM)                (None, 256)               787,456
dense_1 (Dense)            (None, vocab_size/2)      vocab_size * 128
dense_2 (Dense)            (None, vocab_size)        vocab_size * vocab_size/2
=================================================================
```

## 💡 Training Tips

1. **Temperature Tuning**:
   - Lower (0.3-0.7): More focused, consistent output
   - Higher (0.7-1.0): More creative, diverse output

2. **Sequence Length**:
   - Short (50-100): Better local coherence
   - Long (100-200): Better context preservation

3. **Batch Size**:
   - Smaller (32-64): Better generalization
   - Larger (128-256): Faster training

4. **Preventing Overfitting**:
   - Use dropout (0.2)
   - Enable early stopping
   - Monitor validation loss

## 📊 Results & Examples

Example generated poem:
```
Seed: "Trăm năm trong cõi người ta"
Output:
Trăm năm trong cõi người ta,
Chữ tình chữ hiếu thật là đắng cay.
Đêm thanh một giấc mộng say,
Hồn xiêu phách lạc như bay giữa trời...
```

⚠️ **Warning**:
- Training requires significant system memory (minimum 16GB RAM recommended)
- Large memory allocations may occur during training due to model size
- Without GPU acceleration, training will be slower
- Watch for memory warnings in the console during training
- If encountering memory issues, try reducing batch size

### Training the Model

Basic training:
```bash
python train/train.py
```

Advanced training with custom parameters:
```bash
python train/train.py \
    --data_path data/truyen_kieu.txt \
    --seq_length 100 \
    --embedding_dim 256 \
    --lstm_units 256 \
    --epochs 50 \
    --batch_size 64 \
    --model_path models/saved_models/poetry_model.keras
```
