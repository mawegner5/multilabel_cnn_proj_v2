# main.py

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, confusion_matrix, train_test_split
from ray import tune
import pyspark
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns


# Configuration
DATA_DIR = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
FIGURES_DIR = 'figures/'
OUTPUTS_DIR = 'outputs/'

# Hyperparameters
hyperparams = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'dropout_rate': 0.5,
    'num_filters': [32, 64, 128],
    'kernel_size': 3,
    'pool_size': 2,
    'dense_units': 128
}

# Function Placeholders

def load_data():
    """Loads the dataset and returns train, validation, and test data."""
    # Define paths
    train_json_path = os.path.join(DATA_DIR, 'train.json')
    test_json_path = os.path.join(DATA_DIR, 'test.json')
    images_dir = os.path.join(DATA_DIR, 'images')

    # Load annotations
    with open(train_json_path, 'r') as f:
        train_annotations = json.load(f)
    with open(test_json_path, 'r') as f:
        test_annotations = json.load(f)
    
    # Load images and labels
    train_images, train_labels = [], []
    for item in train_annotations:
        image_path = os.path.join(images_dir, item['image_id'])
        image = Image.open(image_path).convert('RGB')
        train_images.append(np.array(image))
        train_labels.append(item['labels'])

    test_images, test_labels = [], []
    for item in test_annotations:
        image_path = os.path.join(images_dir, item['image_id'])
        image = Image.open(image_path).convert('RGB')
        test_images.append(np.array(image))
        test_labels.append(item['labels'])
    
    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # Split train data into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    # Create dictionary to return data
    data = {
        'train': {'images': train_images, 'labels': train_labels},
        'val': {'images': val_images, 'labels': val_labels},
        'test': {'images': test_images, 'labels': test_labels}
    }

    return data

def preprocess_data(train_data, val_data, test_data):
    """Preprocesses the data and returns the processed data."""
    def preprocess_images(images):
        processed_images = []
        for img in images:
            # Resize the image to 224x224 (common input size for CNNs)
            img = Image.fromarray(img).resize((224, 224))
            # Convert the image to array and normalize pixel values to [0, 1]
            img = img_to_array(img) / 255.0
            processed_images.append(img)
        return np.array(processed_images)
    
    def preprocess_labels(labels):
        # Convert list of labels to a binary matrix (multilabel binarization)
        all_labels = [label for sublist in labels for label in sublist]
        unique_labels = list(set(all_labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        binarized_labels = np.zeros((len(labels), len(unique_labels)), dtype=int)
        for i, lbls in enumerate(labels):
            for lbl in lbls:
                binarized_labels[i, label_map[lbl]] = 1
        return binarized_labels, unique_labels

    # Preprocess train, validation, and test images
    train_images = preprocess_images(train_data['images'])
    val_images = preprocess_images(val_data['images'])
    test_images = preprocess_images(test_data['images'])

    # Preprocess train, validation, and test labels
    train_labels, label_classes = preprocess_labels(train_data['labels'])
    val_labels, _ = preprocess_labels(val_data['labels'])
    test_labels, _ = preprocess_labels(test_data['labels'])

    # Update data dictionaries
    train_data['images'] = train_images
    train_data['labels'] = train_labels
    val_data['images'] = val_images
    val_data['labels'] = val_labels
    test_data['images'] = test_images
    test_data['labels'] = test_labels

    # Add label classes to the data dictionary
    data = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'classes': label_classes
    }

    return data

def build_model(hp):
    """Builds and returns the CNN model."""
    model = Sequential()
    
    # Input layer
    model.add(InputLayer(input_shape=(224, 224, 3)))
    
    # Convolutional layers
    for num_filters in hp['num_filters']:
        model.add(Conv2D(filters=num_filters, kernel_size=(hp['kernel_size'], hp['kernel_size']), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(hp['pool_size'], hp['pool_size'])))
    
    # Flatten layer
    model.add(Flatten())
    
    # Fully connected (dense) layers
    model.add(Dense(units=hp['dense_units'], activation='relu'))
    model.add(Dropout(rate=hp['dropout_rate']))
    
    # Output layer (using sigmoid for multilabel classification)
    model.add(Dense(units=len(hp['classes']), activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, train_data, val_data, hp, strategy):
    """Trains the model and returns the training history."""
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(OUTPUTS_DIR, 'best_model.h5'), save_best_only=True)
    ]

    with strategy.scope():
        # Compile model inside the strategy scope
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_data['images'], train_data['labels'],
            validation_data=(val_data['images'], val_data['labels']),
            epochs=hp['epochs'],
            batch_size=hp['batch_size'],
            callbacks=callbacks
        )

    return history

def evaluate_model(model, test_data):
    """Evaluates the model and returns performance metrics."""
    # Predict on test data
    predictions = model.predict(test_data['images'])
    
    # Binarize predictions (using a threshold of 0.5)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate Jaccard score for each label
    jaccard_scores = jaccard_score(test_data['labels'], binary_predictions, average=None)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_data['labels'].argmax(axis=1), binary_predictions.argmax(axis=1))
    
    performance = {
        'jaccard_scores': jaccard_scores,
        'confusion_matrix': cm
    }
    
    return performance

def plot_confusion_matrix(cm, classes, filename):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def plot_jaccard_score(jaccard_scores, filename):
    """Plots and saves the Jaccard scores."""
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(jaccard_scores)), jaccard_scores)
    plt.xlabel('Label')
    plt.ylabel('Jaccard Score')
    plt.title('Jaccard Scores per Label')
    plt.savefig(filename)
    plt.close()

def plot_loss_vs_epochs(history, filename):
    """Plots and saves the loss vs. epochs graph."""
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def tune_hyperparameters(config):
    """Tunes hyperparameters using Ray Tune."""
    analysis = tune.run(
        tune.with_parameters(train_model, train_data=config['train_data'], val_data=config['val_data']),
        config=config['hyperparams'],
        resources_per_trial={'cpu': 2, 'gpu': 0},
        metric="val_loss",
        mode="min",
        num_samples=10
    )

    return analysis

def save_hyperparameters_performance(hyperparams, analysis, filename):
    """Saves the hyperparameters and their performance."""
    best_hyperparams = analysis.get_best_config(metric="val_loss", mode="min")
    best_result = analysis.best_result

    # Save the best hyperparameters and their performance to a file
    with open(filename, 'w') as f:
        json.dump({'hyperparameters': best_hyperparams, 'performance': best_result}, f, indent=4)

def main():
    # Load Data
    data = load_data()
    train_data, val_data, test_data = data['train'], data['val'], data['test']

    # Preprocess Data
    processed_data = preprocess_data(train_data, val_data, test_data)

    # Initialize the strategy
    strategy = tf.distribute.MirroredStrategy()

    # Tune Hyperparameters
    config = {
        'train_data': processed_data['train'],
        'val_data': processed_data['val'],
        'hyperparams': hyperparams
    }
    analysis = tune_hyperparameters(config)

    # Get best hyperparameters
    best_hyperparams = analysis.get_best_config(metric="val_loss", mode="min")

    # Save best hyperparameters and their performance
    save_hyperparameters_performance(best_hyperparams, analysis, os.path.join(OUTPUTS_DIR, 'best_hyperparams.json'))

    # Build Model with best hyperparameters
    model = build_model({**best_hyperparams, 'classes': processed_data['classes']})

    # Train Model with best hyperparameters
    history = train_model(model, processed_data['train'], processed_data['val'], best_hyperparams, strategy)

    # Evaluate Model
    performance = evaluate_model(model, processed_data['test'])

    # Plot Figures
    cm = performance['confusion_matrix']
    plot_confusion_matrix(cm, classes=processed_data['classes'], filename=os.path.join(FIGURES_DIR, 'confusion_matrix.png'))
    plot_jaccard_score(performance['jaccard_scores'], filename=os.path.join(FIGURES_DIR, 'jaccard_scores.png'))
    plot_loss_vs_epochs(history, filename=os.path.join(FIGURES_DIR, 'loss_vs_epochs.png'))


if __name__ == "__main__":
    data = load_data()
    print("Train data shape:", data['train']['images'].shape, data['train']['labels'].shape)
    print("Validation data shape:", data['val']['images'].shape, data['val']['labels'].shape)
    print("Test data shape:", data['test']['images'].shape, data['test']['labels'].shape)
    processed_data = preprocess_data(data['train'], data['val'], data['test'])
    print("Processed train data shape:", processed_data['train']['images'].shape, processed_data['train']['labels'].shape)
    print("Processed validation data shape:", processed_data['val']['images'].shape, processed_data['val']['labels'].shape)
    print("Processed test data shape:", processed_data['test']['images'].shape, processed_data['test']['labels'].shape)
    print("Label classes:", processed_data['classes'])
    processed_data = preprocess_data(data['train'], data['val'], data['test'])
    model = build_model({**hyperparams, 'classes': processed_data['classes']})
    model.summary()
    history = train_model(model, processed_data['train'], processed_data['val'], hyperparams)
    print(history.history)
    performance = evaluate_model(model, processed_data['test'])
    print("Jaccard Scores:", performance['jaccard_scores'])
    print("Confusion Matrix:\n", performance['confusion_matrix'])
