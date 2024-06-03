# README

## Algorithm Purpose
To perform multi-label image classification on the Corel-5k dataset.

## Hyperparameters
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 50

## Background
Utilizes a Convolutional Neural Network (CNN) for classifying images into multiple labels.

## History
CNN-based multi-label image classification has evolved with advancements in deep learning. Initially, single-label classification dominated, but the need for models to handle multiple labels simultaneously led to the development of more sophisticated CNN architectures. Improvements in GPU technology and parallel processing have further enhanced the efficiency and effectiveness of these models.

## Variations
Compared linear and parallel processing methods for training the CNN model.

## Pseudo code
1. Load and preprocess data.
2. Define CNN architecture.
3. Compile model with binary crossentropy loss.
4. Train model using parallel and linear methods.
5. Evaluate and compare performance.

## benchmark results
Jaccard Score (Best Model): 0.823
Training Time:
Parallel: 1258.76 seconds
Linear: 1965.43 seconds


## Example code to import and use the module
```python
from model import train_model, evaluate_model
train_model()
evaluate_model()

