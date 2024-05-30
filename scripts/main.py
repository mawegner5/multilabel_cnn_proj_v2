import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, confusion_matrix
import subprocess
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
from tensorflow.keras.optimizers import Adam

# Configuration
DATA_DIR = '/root/.ipython/multilabel_cnn_proj_v2/data/raw/Corel-5k/Corel-5k/'
PROCESSED_DIR = '/root/.ipython/multilabel_cnn_proj_v2/data/processed/'
OUTPUTS_DIR = '/root/.ipython/multilabel_cnn_proj_v2/outputs/'
FIGURES_DIR = '/root/.ipython/multilabel_cnn_proj_v2/figures/'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Loads the dataset and returns train, validation, and test data."""
    try:
        # Define absolute paths to the JSON files and images directory
        train_json_path = os.path.join(DATA_DIR, 'train.json')
        test_json_path = os.path.join(DATA_DIR, 'test.json')
        images_dir = os.path.join(DATA_DIR, 'images')

        # Check if the files exist
        if not os.path.exists(train_json_path):
            raise FileNotFoundError(f"Train JSON file not found: {train_json_path}")
        if not os.path.exists(test_json_path):
            raise FileNotFoundError(f"Test JSON file not found: {test_json_path}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        with open(train_json_path, 'r') as f:
            train_annotations = json.load(f)
        with open(test_json_path, 'r') as f:
            test_annotations = json.load(f)

        train_images, train_labels = [], []
        for item in train_annotations['samples']:
            image_path = os.path.join(images_dir, item['image_name'])
            image = Image.open(image_path).convert('RGB')
            train_images.append(np.array(image))
            train_labels.append(item['image_labels'])

        test_images, test_labels = [], []
        for item in test_annotations['samples']:
            image_path = os.path.join(images_dir, item['image_name'])
            image = Image.open(image_path).convert('RGB')
            test_images.append(np.array(image))
            test_labels.append(item['image_labels'])

        train_images = np.array(train_images, dtype=object)
        train_labels = np.array(train_labels, dtype=object)
        test_images = np.array(test_images, dtype=object)
        test_labels = np.array(test_labels, dtype=object)

        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

        data = {
            'train': {'images': train_images, 'labels': train_labels},
            'val': {'images': val_images, 'labels': val_labels},
            'test': {'images': test_images, 'labels': test_labels}
        }

        return data

    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise

def preprocess_data(train_data, val_data, test_data):
    """Preprocesses the data and returns the processed data."""
    try:
        def preprocess_images(images):
            processed_images = []
            for img in images:
                img = Image.fromarray(img).resize((224, 224))
                img = img_to_array(img) / 255.0
                processed_images.append(img)
            return np.array(processed_images)
        
        def preprocess_labels(labels, label_map):
            binarized_labels = np.zeros((len(labels), len(label_map)), dtype=int)
            for i, lbls in enumerate(labels):
                for lbl in lbls:
                    binarized_labels[i, label_map[lbl]] = 1
            return binarized_labels

        all_labels = [label for sublist in train_data['labels'] for label in sublist]
        unique_labels = list(set(all_labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        train_images = preprocess_images(train_data['images'])
        val_images = preprocess_images(val_data['images'])
        test_images = preprocess_images(test_data['images'])

        train_labels = preprocess_labels(train_data['labels'], label_map)
        val_labels = preprocess_labels(val_data['labels'], label_map)
        test_labels = preprocess_labels(test_data['labels'], label_map)

        train_data['images'] = train_images
        train_data['labels'] = train_labels
        val_data['images'] = val_images
        val_data['labels'] = val_labels
        test_data['images'] = test_images
        test_data['labels'] = test_labels

        data = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'classes': unique_labels
        }

        return data

    except Exception as e:
        logging.error(f"An error occurred while preprocessing data: {e}")
        raise

def save_preprocessed_data(data, processed_dir):
    """Saves the preprocessed data to disk."""
    try:
        os.makedirs(processed_dir, exist_ok=True)
        
        train_images_path = os.path.join(processed_dir, 'train_images.npy')
        val_images_path = os.path.join(processed_dir, 'val_images.npy')
        test_images_path = os.path.join(processed_dir, 'test_images.npy')
        train_labels_path = os.path.join(processed_dir, 'train_labels.npy')
        val_labels_path = os.path.join(processed_dir, 'val_labels.npy')
        test_labels_path = os.path.join(processed_dir, 'test_labels.npy')
        classes_path = os.path.join(processed_dir, 'classes.json')
        
        np.save(train_images_path, data['train']['images'])
        np.save(val_images_path, data['val']['images'])
        np.save(test_images_path, data['test']['images'])
        np.save(train_labels_path, data['train']['labels'])
        np.save(val_labels_path, data['val']['labels'])
        np.save(test_labels_path, data['test']['labels'])
        
        with open(classes_path, 'w') as f:
            json.dump(data['classes'], f)

        logging.info("Preprocessed data saved successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred while saving preprocessed data: {e}")
        raise

def build_model(config):
    """Builds and compiles the CNN model."""
    model = Sequential()
    model.add(InputLayer(input_shape=(224, 224, 3)))

    for num_filter in config["NUM_FILTERS"]:
        model.add(Conv2D(num_filter, (config["KERNEL_SIZE"], config["KERNEL_SIZE"]), activation='relu'))
        model.add(MaxPooling2D(pool_size=(config["POOL_SIZE"], config["POOL_SIZE"])))
        model.add(Dropout(config["DROPOUT_RATE"]))

    model.add(Flatten())
    model.add(Dense(config["DENSE_UNITS"], activation='relu'))
    model.add(Dropout(config["DROPOUT_RATE"]))
    model.add(Dense(260, activation='sigmoid'))  # Sigmoid activation for multi-label classification

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"]),
                  loss='binary_crossentropy',  # Use binary cross-entropy loss
                  metrics=['accuracy'])

    return model

def data_generator(images, labels, batch_size):
    """Generates batches of data for training."""
    datagen = ImageDataGenerator()
    generator = datagen.flow(images, labels, batch_size=batch_size)
    return generator

def train_model(config, data, use_parallel_strategy):
    """Trains the CNN model with the given configuration and data."""
    try:
        train_images = data['train']['images']
        train_labels = data['train']['labels']
        val_images = data['val']['images']
        val_labels = data['val']['labels']

        train_gen = data_generator(train_images, train_labels, config["BATCH_SIZE"])
        val_gen = data_generator(val_images, val_labels, config["BATCH_SIZE"])

        if use_parallel_strategy:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = build_model(config)
        else:
            model = build_model(config)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        start_time = time.time()
        history = model.fit(train_gen,
                            validation_data=val_gen,
                            epochs=config["EPOCHS"],
                            steps_per_epoch=len(train_images) // config["BATCH_SIZE"],
                            validation_steps=len(val_images) // config["BATCH_SIZE"],
                            callbacks=[early_stopping])
        training_time = time.time() - start_time

        return model, history, training_time
    
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise

def evaluate_model(model, test_data):
    """Evaluates the trained model on the test data."""
    try:
        test_images = test_data['images']
        test_labels = test_data['labels']

        predictions = model.predict(test_images)
        jaccard_scores = jaccard_score(test_labels, np.round(predictions), average=None)
        overall_jaccard = jaccard_score(test_labels, np.round(predictions), average='micro')

        y_true = test_labels.argmax(axis=1)
        y_pred = predictions.argmax(axis=1)
        cm = confusion_matrix(y_true, y_pred)

        performance = {
            'jaccard_scores': jaccard_scores,
            'overall_jaccard': overall_jaccard,
            'confusion_matrix': cm
        }

        return performance
    
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")
        raise

def plot_confusion_matrix(cm, filename):
    normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Non-Relevant', 'Relevant'], yticklabels=['Non-Relevant', 'Relevant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def plot_overall_jaccard_score(overall_jaccard, filename):
    plt.figure(figsize=(10, 8))
    plt.bar(['Overall Jaccard Score'], [overall_jaccard])
    plt.ylabel('Jaccard Score')
    plt.title('Overall Jaccard Score')
    plt.savefig(filename)
    plt.close()

def plot_loss_vs_epochs(history, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def monitor_gpu_usage():
    """Monitors GPU usage using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        logging.info(result.stdout.decode('utf-8'))
    except Exception as e:
        logging.error(f"An error occurred while monitoring GPU usage: {e}")
        raise

def train_with_tune(config):
    """Wrapper function to train the model with hyperparameter tuning."""
    data = load_data()
    processed_data = preprocess_data(data['train'], data['val'], data['test'])
    model, history, _ = train_model(config, processed_data, use_parallel_strategy=False)
    
    # Report metrics to Ray Tune
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    tune.report(val_loss=val_loss, val_accuracy=val_accuracy)

def main():
    try:
        logging.info("Loading data...")
        data = load_data()

        logging.info("Preprocessing data...")
        processed_data = preprocess_data(data['train'], data['val'], data['test'])

        logging.info("Saving preprocessed data...")
        save_preprocessed_data(processed_data, PROCESSED_DIR)

        logging.info("Building the model...")

        # Hyperparameter tuning with Ray Tune
        search_space = {
            "BATCH_SIZE": tune.choice([16, 32, 64]),
            "EPOCHS": tune.choice([10, 20, 30]),
            "LEARNING_RATE": tune.loguniform(1e-4, 1e-2),
            "DROPOUT_RATE": tune.uniform(0.2, 0.5),
            "NUM_FILTERS": tune.choice([[32, 64], [64, 128]]),
            "KERNEL_SIZE": tune.choice([3, 5]),
            "POOL_SIZE": tune.choice([2, 3]),
            "DENSE_UNITS": tune.choice([64, 128])
        }

        scheduler = ASHAScheduler(metric="val_loss", mode="min")
        analysis = tune.run(train_with_tune, config=search_space, num_samples=10, scheduler=scheduler, resources_per_trial={"cpu": 2, "gpu": 1})

        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        logging.info(f"Best hyperparameters: {best_config}")

        # Train with the best hyperparameters
        logging.info("Training with parallel strategy...")
        monitor_gpu_usage()
        model, history_parallel, training_time_parallel = train_model(best_config, processed_data, use_parallel_strategy=True)
        monitor_gpu_usage()
        logging.info("Training time with parallel strategy: %s seconds", training_time_parallel)

        logging.info("Training without parallel strategy...")
        monitor_gpu_usage()
        model, history_non_parallel, training_time_non_parallel = train_model(best_config, processed_data, use_parallel_strategy=False)
        monitor_gpu_usage()
        logging.info("Training time without parallel strategy: %s seconds", training_time_non_parallel)

        logging.info("Evaluating the model...")
        performance = evaluate_model(model, processed_data['test'])

        logging.info("Creating figures directory if it does not exist...")
        os.makedirs(FIGURES_DIR, exist_ok=True)

        logging.info("Plotting confusion matrix...")
        plot_confusion_matrix(performance['confusion_matrix'], filename=os.path.join(FIGURES_DIR, 'confusion_matrix.png'))

        logging.info("Plotting overall Jaccard score...")
        plot_overall_jaccard_score(performance['overall_jaccard'], filename=os.path.join(FIGURES_DIR, 'overall_jaccard_score.png'))

        logging.info("Plotting loss vs. epochs...")
        plot_loss_vs_epochs(history_non_parallel, filename=os.path.join(FIGURES_DIR, 'loss_vs_epochs.png'))

        logging.info(f"Overall Jaccard score: {performance['overall_jaccard']:.4f}")

        logging.info("Training time with parallel strategy: %s seconds", training_time_parallel)
        logging.info("Training time without parallel strategy: %s seconds", training_time_non_parallel)

    except Exception as e:
        logging.error(f"An error occurred in the preprocessing and training pipeline: {e}")

if __name__ == "__main__":
    main()
