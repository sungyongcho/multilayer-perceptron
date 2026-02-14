# multilayer-perceptron

> Neural network built from scratch — no framework, manual backpropagation.

## Overview

A fully connected multilayer perceptron implemented using only NumPy. The network is trained from scratch with manual forward pass, backpropagation, and gradient descent to classify breast tumors as malignant or benign on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

This project was built as part of the 42 school AI curriculum · Score: 125/100. No machine learning libraries (scikit-learn, PyTorch, TensorFlow) are used for the model itself.

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.x |
| Core | NumPy |
| Visualization | Matplotlib |
| Data | Pandas (data loading only) |

## Key Features

- Manual backpropagation with chain rule gradient computation
- Four gradient descent optimizers: SGD, Adam, Adagrad, RMSProp
- Numerically stable softmax with log-sum-exp trick and cross-entropy loss
- He weight initialization for ReLU-based hidden layers
- Mini-batch training with configurable batch size
- Early stopping with validation loss monitoring
- Model checkpoint persistence (save/load trained weights)
- Training and validation loss/accuracy curves

## Results

| Metric | Value |
|--------|-------|
| Dataset | WDBC (569 samples, 30 features) |
| Classification | Binary (Malignant / Benign) |
| Hidden Layers | Configurable (minimum 2) |
| Output | Softmax (2 classes) |
| Loss Function | Cross-entropy |

## Architecture

```
multilayer-perceptron/
├── multilayer_perceptron.py    # Main training script
├── predict.py                 # Prediction/evaluation script
├── network.py                 # Neural network class (forward, backward)
├── layer.py                   # Layer abstraction (weights, biases, gradients)
├── optimizers.py              # SGD, Adam, Adagrad, RMSProp implementations
├── activation.py              # ReLU, softmax, sigmoid activations
├── loss.py                    # Cross-entropy loss computation
├── utils.py                   # Data loading, preprocessing, metrics
├── data/
│   └── data.csv               # WDBC dataset
└── models/                    # Saved model checkpoints
```

## Getting Started

### Prerequisites

```bash
Python 3.x
NumPy
Matplotlib
Pandas
```

### Installation

```bash
git clone https://github.com/sungyongcho/multilayer-perceptron.git
cd multilayer-perceptron
pip install numpy matplotlib pandas
```

### Usage

```bash
# Train the network
python multilayer_perceptron.py data/data.csv

# Predict with a saved model
python predict.py data/data.csv
```

### Training

```bash
# Train with specific optimizer
python multilayer_perceptron.py data/data.csv --optimizer adam

# Available optimizers: sgd, adam, adagrad, rmsprop
```

## What This Demonstrates

- **ML Fundamentals**: Implemented backpropagation and gradient descent from first principles without any ML framework — demonstrating understanding of what happens under the hood of `model.fit()`.
- **Numerical Computing**: Built numerically stable softmax (log-sum-exp), cross-entropy loss, and He initialization using only NumPy.
- **Optimization Theory**: Implemented four gradient descent variants (SGD, Adam, Adagrad, RMSProp) with correct momentum and adaptive learning rate updates.

## License

This project was built as part of the 42 school curriculum.

---

*Part of [sungyongcho](https://github.com/sungyongcho)'s project portfolio.*
