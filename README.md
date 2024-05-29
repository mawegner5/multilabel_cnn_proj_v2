# multilabel_cnn_proj
# Parallel CNN Project

## Project Overview
The fundamental objective of this project is to create a Convolutional Neural Network (CNN) for multi-label image classification. Given an image containing multiple objects, the model correctly labels the image with the objects in it. The project involves training on roughly 3000 images, validating on another 1000 images, and testing on approximately 1000 images to evaluate the final Jaccard score.

## Initial Setup and Planning

### Select Topic Area
This project is directly related to the thesis on multi-label image classification using CNNs.

### Define Inputs and Outputs
- **Inputs**: Images with multiple objects, JSON files with associated labels.
- **Outputs**: Predicted labels for each image.

### Data Acquisition
The dataset is existing and contains labeled images necessary for training, validation, and testing.

## Week 3 Deliverables

### Meeting with Instructor
- **Discussion Topics**:
  - Data acquisition methods.
  - ANN architecture feasibility.
  - Baseline metrics and comparison strategies.
  - Hardware setup readiness.

### Data Preparation
- **Datasets**: Ensured the datasets are collected and ready for use.
- **Data Processing**: Implemented strategies for processing and splitting data into training, validation, and testing sets.

## Development

### ANN Architecture
- **Architecture Type**: Convolutional Neural Network (CNN).
- **Scalability**: Designed to scale with the quantity of data.

### Parallel Processing and Code Management
- **Parallel Processing**: Implemented using TensorFlow's MirroredStrategy.
- **Training Comparison**: Compared training speed and computational requirements with and without parallel processing.
- **Git Repository**: Code is stored on GitHub, configured to run unit tests on code changes. Dependencies specified in `requirements.txt`.

## Testing and Hyperparameter Tuning

### Unit Tests
- Created unit tests for the `load_data` and `preprocess_data` functions, ensuring they accept and output the correct data.

### Hyperparameter Specification
- Users can specify hyper-parameters for the module.

### Distributed Processing
- Ensured the algorithm works in a distributed manner for enhanced performance.

## Implementation and Flexibility

### Implementation with Multiple Datasets
- Demonstrated implementation using the Corel-5k dataset. Future work includes testing with additional datasets.

### Code Flexibility
- Designed to work with other datasets having different numbers or types of variables.

## Methodology

### Repeatable Architecture
- Ensured the architecture can be replicated.

### Regularization and Hyperparameters
- **Regularization Techniques**: Dropout.
- **Learning Rate**: 0.001.
- **Optimizer**: Adam.
- **Epochs**: 20.

### Model Fitting
- Developed a model that fits the data shapes.

### Scaling Up
- Scaled the model to learn the training data effectively.

### Generalization
- Regularized and tweaked hyperparameters to generalize well to validation data.

## Documentation and Presentation

### README.md Sections
- **Algorithm Purpose**: Explained above.
- **Hyperparameters**: Detailed in the "Hyperparameter Specification" section.
- **Background**: CNNs for multi-label classification.
- **History**: Use of CNNs in image classification.
- **Variations**: Different architectures tested.
- **Pseudo code**: Provided below.
- **Example Code**: Usage example provided.
- **Visualization**: Included in the results section.
- **Benchmark Results**: Compared parallel and non-parallel processing times and accuracy.
- **Lessons Learned**: Documented below.
- **Unit-testing Strategy**: Detailed in the "Unit Tests" section.
- **Checkpoint Technique**: Model checkpoints included.

### Pseudo Code
```python
# Pseudo code for the main steps
load_data()
preprocess_data()
save_preprocessed_data()
build_model()
train_model()
evaluate_model()
plot_results()
