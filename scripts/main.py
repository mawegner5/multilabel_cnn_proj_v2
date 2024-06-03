import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, Callback
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, confusion_matrix, f1_score
import subprocess
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
from tensorflow.keras.optimizers import Adam
from ray.air import session

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
            'classes': unique_labels  # Ensure 'classes' key is included
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

        # Define callbacks here
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

        start_time = time.time()
        history = model.fit(train_gen,
                            validation_data=val_gen,
                            epochs=config["EPOCHS"],
                            steps_per_epoch=len(train_images) // config["BATCH_SIZE"],
                            validation_steps=len(val_images) // config["BATCH_SIZE"],
                            callbacks=callbacks)
        training_time = time.time() - start_time

        return model, history, training_time

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise

def evaluate_model(model, test_data):
    """Evaluates the trained model on the test data and saves predicted labels."""
    try:
        test_images = test_data['images']
        test_labels = test_data['labels']

        predictions = model.predict(test_images)
        jaccard_scores = jaccard_score(test_labels, np.round(predictions), average=None)
        overall_jaccard = jaccard_score(test_labels, np.round(predictions), average='micro')
        
        # Calculate F1-score
        f1_scores = f1_score(test_labels, np.round(predictions), average=None)
        overall_f1 = f1_score(test_labels, np.round(predictions), average='micro')

        y_true = test_labels.argmax(axis=1)
        y_pred = predictions.argmax(axis=1)
        cm = confusion_matrix(y_true, y_pred)

        performance = {
            'jaccard_scores': jaccard_scores,
            'overall_jaccard': overall_jaccard,
            'f1_scores': f1_scores,
            'overall_f1': overall_f1,
            'confusion_matrix': cm
        }

        # Save image labels to JSON
        save_image_labels(test_data, predictions, os.path.join(OUTPUTS_DIR, 'image_labels.json'))

        return performance
    
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")
        raise

def save_image_labels(test_data, predictions, output_file_path):
    """Saves the image IDs, predicted labels, actual labels, and label IDs to a JSON file."""
    try:
        image_labels = []
        for idx, prediction in enumerate(predictions):
            image_id = idx  # Using the index as the image ID
            predicted_labels = [label for label, pred in zip(test_data['classes'], prediction) if pred > 0.5]
            actual_labels = [label for label, true in zip(test_data['classes'], test_data['labels'][idx]) if true == 1]
            label_ids = [test_data['classes'].index(label) for label in actual_labels]
            image_labels.append({
                'image_id': image_id,
                'predicted_labels': predicted_labels,
                'actual_labels': actual_labels,
                'label_ids': label_ids
            })

        with open(output_file_path, 'w') as f:
            json.dump(image_labels, f, indent=4)
        logging.info(f"Image labels saved to {output_file_path}")

    except Exception as e:
        logging.error(f"An error occurred while saving image labels: {e}")
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

def plot_loss_vs_epochs(parallel_history, linear_history, filename):
    plt.figure(figsize=(10, 8))
    plt.plot(parallel_history.history['loss'], label='Parallel Processing Loss')
    plt.plot(linear_history.history['loss'], label='Linear Processing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs for Parallel and Linear Processing')
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

class SaveModelInfoCallback(tune.Callback):
    def __init__(self):
        self.model_info_list = []

    def on_trial_complete(self, iteration, trials, trial, **info):
        model_name = trial.trial_id
        hyperparameters = trial.config
        performance_metrics = {
            'val_loss': trial.last_result['val_loss'],
            'val_accuracy': trial.last_result['val_accuracy'],
            'val_f1': trial.last_result['val_f1']
        }
        model_info = {
            'model_name': model_name,
            'hyperparameters': hyperparameters,
            'performance_metrics': performance_metrics
        }
        self.model_info_list.append(model_info)

    def save_to_file(self, output_file_path):
        with open(output_file_path, 'w') as f:
            json.dump(self.model_info_list, f, indent=4)
        logging.info(f"Model information saved to {output_file_path}")

# Custom Keras callback to report metrics through Ray Train
class RayTuneReportCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        session.report({"loss": logs["val_loss"], "accuracy": logs["val_accuracy"]})

def train_with_tune(config, checkpoint_dir=None):
    # Data loading and preprocessing
    data = load_data()  # Corrected to the right function name
    processed_data = preprocess_data(data['train'], data['val'], data['test'])

    # Build model
    model = build_model(config)

    # Define data generators
    train_gen = data_generator(processed_data['train']['images'], processed_data['train']['labels'], config["BATCH_SIZE"])
    val_gen = data_generator(processed_data['val']['images'], processed_data['val']['labels'], config["BATCH_SIZE"])

    # Set callbacks
    callbacks = [RayTuneReportCallback(), EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    # Train the model
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=config["EPOCHS"],
              steps_per_epoch=len(processed_data['train']['images']) // config["BATCH_SIZE"],
              validation_steps=len(processed_data['val']['images']) // config["BATCH_SIZE"],
              callbacks=callbacks,
              verbose=1)

def main():
    try:
        logging.info("Loading data...")
        data = load_data()

        logging.info("Preprocessing data...")
        processed_data = preprocess_data(data['train'], data['val'], data['test'])

        logging.info("Saving preprocessed data...")
        save_preprocessed_data(processed_data, PROCESSED_DIR)

        logging.info("Building and training the model with Ray Tune...")

        # Define the search space for hyperparameters
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

        # Configure the scheduler and execution
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=30,
            grace_period=5,
            reduction_factor=2
        )

        # Initialize Ray
        if not ray.is_initialized():
            ray.init()

        # Start Ray Tune run with the updated training function
        result = tune.run(
            train_with_tune,
            name="Multi-Label-Classification",
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=search_space,
            num_samples=10,
            scheduler=scheduler,
            progress_reporter=tune.CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
        )

        # Fetch the best trial
        best_trial = result.get_best_trial("accuracy", "max", "last")
        logging.info(f"Best trial config: {best_trial.config}")
        logging.info(f"Best trial final validation loss: {best_trial.last_result['loss']}")
        logging.info(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

        # Rebuild and retrain the model using the best trial parameters
        best_trained_model = build_model(best_trial.config)
        best_train_gen = data_generator(processed_data['train']['images'], processed_data['train']['labels'], best_trial.config["BATCH_SIZE"])
        best_val_gen = data_generator(processed_data['val']['images'], processed_data['val']['labels'], best_trial.config["BATCH_SIZE"])

        # Retrain the best model
        best_trained_model.fit(
            best_train_gen,
            validation_data=best_val_gen,
            epochs=best_trial.config["EPOCHS"],
            steps_per_epoch=len(processed_data['train']['images']) // best_trial.config["BATCH_SIZE"],
            validation_steps=len(processed_data['val']['images']) // best_trial.config["BATCH_SIZE"]
        )

        # Evaluate the best model
        logging.info("Evaluating the best model...")
        test_performance = evaluate_model(best_trained_model, processed_data['test'])
        logging.info(f"Test Jaccard Score: {test_performance['overall_jaccard']:.4f}")
        logging.info(f"Test F1 Score: {test_performance['overall_f1']:.4f}")

        # Saving the performance metrics and model
        performance_file_path = os.path.join(OUTPUTS_DIR, "best_model_performance.json")
        with open(performance_file_path, 'w') as f:
            json.dump(test_performance, f, indent=4)
        logging.info("Model performance metrics saved.")

        # Plotting results
        plot_confusion_matrix(test_performance['confusion_matrix'], os.path.join(FIGURES_DIR, 'confusion_matrix.png'))
        plot_overall_jaccard_score(test_performance['overall_jaccard'], os.path.join(FIGURES_DIR, 'overall_jaccard_score.png'))
        plot_loss_vs_epochs(best_trained_model.history, os.path.join(FIGURES_DIR, 'loss_vs_epochs.png'))

        # Save the hyperparameters of the best model
        best_model_config_path = os.path.join(OUTPUTS_DIR, "best_model_config.json")
        with open(best_model_config_path, 'w') as f:
            json.dump(best_trial.config, f, indent=4)
        logging.info("Best model hyperparameters saved.")

        # Train parallel and linear models
        parallel_model, parallel_history, parallel_time = train_model(best_trial.config, processed_data, use_parallel_strategy=True)
        linear_model, linear_history, linear_time = train_model(best_trial.config, processed_data, use_parallel_strategy=False)

        # Plot loss vs. epochs for both models
        plot_loss_vs_epochs(parallel_history, linear_history, os.path.join(FIGURES_DIR, 'loss_vs_epochs_parallel_vs_linear.png'))

        # Shutdown Ray
        ray.shutdown()

    except Exception as e:
        logging.error(f"An error occurred in the preprocessing and training pipeline: {e}")
        ray.shutdown()  # Ensure Ray shuts down regardless of the error

if __name__ == "__main__":
    main()
