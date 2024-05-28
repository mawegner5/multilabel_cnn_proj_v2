import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import zipfile

# Configuration
DATA_DIR = '/root/.ipython/multilabel_cnn_proj/data/raw/Corel-5k/Corel-5k/Corel-5k/'
PROCESSED_DIR = '/root/.ipython/multilabel_cnn_proj/data/processed/'
OUTPUTS_DIR = '/root/.ipython/multilabel_cnn_proj/outputs/'

# Hyperparameters
hyperparams = {
    'batch_size': 2,
    'epochs': 1,
    'learning_rate': 0.001,
    'dropout_rate': 0.5,
    'num_filters': [32],
    'kernel_size': 3,
    'pool_size': 2,
    'dense_units': 64
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_zip(file_path, extract_to):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f"Extracted {file_path} to {extract_to}.")
    except Exception as e:
        logging.error(f"An error occurred while extracting {file_path}: {e}")
        raise

def load_data():
    try:
        train_json_path = os.path.join(DATA_DIR, 'train.json')
        test_json_path = os.path.join(DATA_DIR, 'test.json')
        images_dir = os.path.join(DATA_DIR, 'images')

        logging.info(f"Train JSON path: {train_json_path}")
        logging.info(f"Test JSON path: {test_json_path}")
        logging.info(f"Images directory: {images_dir}")

        with open(train_json_path, 'r') as f:
            train_annotations = json.load(f)
        with open(test_json_path, 'r') as f:
            test_annotations = json.load(f)

        logging.info("Loaded annotations successfully.")
        logging.info(f"Train JSON keys: {train_annotations.keys()}")
        logging.info(f"Test JSON keys: {test_annotations.keys()}")
        logging.info(f"First train sample: {train_annotations['samples'][0]}")
        logging.info(f"First test sample: {test_annotations['samples'][0]}")

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

        logging.info("Loaded images and labels successfully.")

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

        logging.info(f"Number of classes: {len(unique_labels)}")
        logging.info(f"Shape of train labels: {train_labels.shape}")
        logging.info(f"Shape of val labels: {val_labels.shape}")
        logging.info(f"Shape of test labels: {test_labels.shape}")

        logging.info("Data preprocessed successfully.")
        return data

    except Exception as e:
        logging.error(f"An error occurred while preprocessing data: {e}")
        raise

def save_preprocessed_data(data, processed_dir):
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

def build_model(input_shape, num_classes, hyperparams):
    try:
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))

        for num_filter in hyperparams['num_filters']:
            model.add(Conv2D(num_filter, (hyperparams['kernel_size'], hyperparams['kernel_size']), activation='relu'))
            model.add(MaxPooling2D(pool_size=(hyperparams['pool_size'], hyperparams['pool_size'])))
            model.add(Dropout(hyperparams['dropout_rate']))
        
        model.add(Flatten())
        model.add(Dense(hyperparams['dense_units'], activation='relu'))
        model.add(Dropout(hyperparams['dropout_rate']))
        model.add(Dense(num_classes, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        logging.info("Model built and compiled successfully.")
        return model

    except Exception as e:
        logging.error(f"An error occurred while building the model: {e}")
        raise

def train_model(model, train_data, val_data, hyperparams):
    try:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join(OUTPUTS_DIR, 'best_model.h5'), save_best_only=True)
        ]

        history = model.fit(
            train_data['images'], train_data['labels'],
            validation_data=(val_data['images'], val_data['labels']),
            epochs=hyperparams['epochs'],
            batch_size=hyperparams['batch_size'],
            callbacks=callbacks
        )

        logging.info("Model trained successfully.")
        return history

    except Exception as e:
        logging.error(f"An error occurred while training the model: {e}")
        raise

if __name__ == "__main__":
    try:
        extract_zip('/root/.ipython/multilabel_cnn_proj/data/Corel-5k.zip', '/root/.ipython/multilabel_cnn_proj/data/raw/Corel-5k/')
        
        logging.info("Loading data...")
        data = load_data()

        logging.info("Preprocessing data...")
        processed_data = preprocess_data(data['train'], data['val'], data['test'])

        logging.info("Saving preprocessed data...")
        save_preprocessed_data(processed_data, PROCESSED_DIR)

        logging.info("Building the model...")
        input_shape = processed_data['train']['images'][0].shape
        num_classes = len(processed_data['classes'])
        model = build_model(input_shape, num_classes, hyperparams)

        logging.info("Training the model...")
        history = train_model(model, processed_data['train'], processed_data['val'], hyperparams)

        logging.info("Data preprocessing and model training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred in the preprocessing and training pipeline: {e}")
