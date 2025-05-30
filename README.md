# Image Object Recognition using CIFAR-10

## Project Overview

This project focuses on image object recognition using the CIFAR-10 dataset. The primary goal is to build, train, and evaluate machine learning models, specifically a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), to classify images from the CIFAR-10 dataset effectively. The project explores various aspects of machine learning, including data preprocessing, feature engineering, model selection, hyperparameter tuning, and multi-metric evaluation.

## Dataset

The project utilizes the CIFAR-10 dataset, a widely used benchmark for image classification tasks.

* **Total Images:** 60,000 (32x32 RGB color images)
* **Number of Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
* **Train/Test Split:** 50,000 training images and 10,000 test images. Each class has 6,000 images in total.
* **Image Format:** 32x32x3 (RGB)

## Models

Two primary models were developed and evaluated:

### 1. Multi-Layer Perceptron (MLP)

* **Description:** A feedforward neural network used as a baseline classification model. Images are flattened into 1D vectors (3072 features) before being fed into the network.
* **Architecture:**
    * Input Layer: Flattens 32x32x3 images.
    * Hidden Layer 1: Dense layer with 1024 neurons, Batch Normalization, ReLU activation, Dropout (0.3).
    * Hidden Layer 2: Dense layer with 512 neurons, Batch Normalization, ReLU activation, Dropout (0.3).
    * Hidden Layer 3: Dense layer with 256 neurons, Batch Normalization, ReLU activation.
    * Output Layer: Dense layer with 10 neurons and Softmax activation.
* **Data Preprocessing:**
    * Pixel normalization (0-1).
    * One-hot encoding for labels.
    * Data augmentation: random rotation, width/height shift, horizontal flip.
* **Performance:** Achieved a test accuracy of approximately 58.37%.

### 2. Convolutional Neural Network (CNN)

* **Description:** A specialized neural network architecture designed for image data, capable of learning spatial hierarchies of features.
* **Architecture:**
    * Images are kept in their original 32x32x3 format.
    * **Block 1:** Conv2D (32 filters, 3x3), Batch Normalization, ReLU, Conv2D (32 filters, 3x3), Batch Normalization, ReLU, MaxPooling2D (2x2), Dropout (0.2).
    * **Block 2:** Conv2D (64 filters, 3x3), Batch Normalization, ReLU, Conv2D (64 filters, 3x3), Batch Normalization, ReLU, MaxPooling2D (2x2), Dropout (0.3).
    * **Block 3:** Conv2D (128 filters, 3x3), Batch Normalization, ReLU, Conv2D (128 filters, 3x3), Batch Normalization, ReLU, MaxPooling2D (2x2), Dropout (0.4).
    * **Classification Head:** GlobalAveragePooling2D, Dense (256 neurons), Batch Normalization, ReLU, Dropout (0.5), Dense (10 neurons, Softmax activation).
* **Data Preprocessing:**
    * Pixel normalization (0-1).
    * One-hot encoding for labels.
    * Dataset split: 80% training, 20% validation.
    * Data augmentation: random rotation (15 degrees), width/height shift (0.1), horizontal flip.
* **Performance:** Achieved a test accuracy of approximately 85-86%.

## Tools and Libraries

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle
* OS
* Google Colab
* Google Docs
* Google Slides

## Setup and Usage

1.  **Clone the repository.**
2.  **Ensure Python and the necessary libraries (listed above) are installed.**
    ```bash
    pip install numpy tensorflow scikit-learn matplotlib seaborn
    ```
3.  **Dataset:** The scripts expect the CIFAR-10 dataset to be available.
    * The `cnn.py` script specifies `data_dir = '/kaggle/input/cifar1'`.
    * The `mlp.py` script specifies `data_dir = r"C:\Users\yagmu\Downloads\cifar-10-batches-py"`.
    You will need to modify these paths to point to your local CIFAR-10 dataset directory, or ensure the data is in the specified Kaggle path if running in that environment. The dataset is typically downloaded as Python pickle files (`data_batch_1` to `data_batch_5`, `test_batch`).
4.  **Run the scripts:**
    ```bash
    python cnn.py
    python mlp.py
    ```
    The scripts will load the data, preprocess it, build the respective models, train them, evaluate their performance, and display visualizations like confusion matrices and accuracy/loss curves. The trained CNN model will be saved as `cifar10_cnn_model.h5`.

## Evaluation Summary

The CNN model significantly outperformed the MLP model across all metrics, demonstrating the effectiveness of convolutional layers in capturing spatial features for image classification.

| Model | Accuracy      | Precision (avg) | Recall (avg) | F1-Score (avg) |
| :---- | :------------ | :-------------- | :----------- | :------------- |
| MLP   | 0.5837 (58.4%) | 0.5871          | 0.5837       | 0.5771         | 
| CNN   | 0.850 (85.0%)  | 0.8542          | 0.8498       | 0.8476         | 
*(Note: Presentation mentions 86% accuracy for CNN and slightly different precision/recall/F1 for CNN)*

## Challenges

* MLP models are not inherently suited for capturing spatial relationships in images, making it difficult to distinguish between visually similar classes.
* Training deep models, especially CNNs, can be computationally expensive and time-consuming.
* The CIFAR-10 dataset's low-resolution images, background clutter, and variations in object orientation pose challenges for consistent feature learning.

## Team Contributions

* **Kerim Ercan:** Led the creation, coding, training, and evaluation of the CNN model, including visualization outputs and accuracy analysis.
* **Berat Kaya:** Worked on implementing data preprocessing steps, training MLP models, and tuning hyperparameters. Contributed to the visual layout and technical content of the presentation.
* **Fatma Zehra Bayır:** Collaborated on the extended MLP model, implementing dropout and regularization methods. Took an active role in writing the project report, content monitoring, final control, and team coordination for the presentation.
* **Yağmur Parmaksız:** Took an active role in creating the basic MLP model, monitoring its training, and evaluating results. Contributed significantly to writing and editing the project report and the content layout of the presentation.

---

This README provides a good starting point. You can further customize it as needed!
