# MNIST Handwritten Digit Classification

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset.

## Project Overview

The project demonstrates:
- Building a CNN architecture in PyTorch
- Training and evaluating the model on the MNIST dataset
- Visualizing results and model performance

## Contents

- `mnist_classification.ipynb`: Jupyter notebook containing the full implementation
- `mnist_cnn_model.pth`: Saved model weights
- `data/`: Directory containing the MNIST dataset (downloaded automatically)

## Requirements

- Python 3.6+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy

## Features

- CNN architecture with 3 convolutional blocks
- Batch normalization for improved training
- Dropout for regularization
- Support for GPU acceleration (CUDA and Apple Silicon MPS)

## Usage

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mnist_classification.git
   cd mnist_classification
   ```

2. Open and run the Jupyter notebook:
   ```
   jupyter notebook mnist_classification.ipynb
   ```

3. The notebook will:
   - Download the MNIST dataset
   - Build the CNN model
   - Train the model
   - Evaluate performance
   - Visualize results

## Model Architecture

The implemented CNN has:
- 3 convolutional layers with batch normalization
- Max pooling after each convolutional layer
- 2 fully connected layers
- Dropout for regularization
- Output layer with 10 classes (digits 0-9)

## Results

The model achieves approximately 99% accuracy on the MNIST test dataset.

## License

This project is open-source and available under the MIT License. 