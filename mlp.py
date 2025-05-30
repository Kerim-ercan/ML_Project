# Import necessary libraries
import pickle  # For loading the CIFAR-10 dataset from pickle files
import numpy as np  # For numerical operations
import os  # For file path operations
import tensorflow as tf  # Deep learning framework
from tensorflow.keras import layers, models  # For defining neural network layers and models
from tensorflow.keras.layers import BatchNormalization  # Batch normalization layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.callbacks import ReduceLROnPlateau  # Learning rate scheduler callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics


# 1. Load dataset

# Function to load a single data batch from CIFAR-10
def load_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')  # Load with Latin-1 encoding to avoid byte issues
    data = batch['data']  # Raw image data
    labels = batch['labels']  # Corresponding labels
    # Reshape data to (num_samples, 32, 32, 3) from flat vectors
    data = data.reshape((len(data), 3, 32, 32)).transpose(0, 2, 3, 1)
    return data, labels

# Function to load the entire CIFAR-10 training and test dataset
def load_cifar10_data(data_dir):
    x_train = []
    y_train = []
    # Load all 5 training batches
    for i in range(1, 6):
        file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_batch(file)
        x_train.append(data)
        y_train.extend(labels)

    # Concatenate the batches into a single training dataset
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)

    # Load test batch
    x_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

# 2. Load data

data_dir = r"C:\Users\yagmu\Downloads\cifar-10-batches-py"  # Directory containing CIFAR-10 data
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)  # Load data

# Print dataset shapes for confirmation
print("Training data:", x_train.shape)
print("Test data:", x_test.shape)

# Normalize image pixel values to [0, 1] for better training stability
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Data augmentation

# Create an image data generator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=15,  # Random rotation up to 15 degrees
    width_shift_range=0.1,  # Horizontal shift
    height_shift_range=0.1,  # Vertical shift
    horizontal_flip=True  # Random horizontal flipping
)
datagen.fit(x_train)  # Fit the generator to training data


# 4. Model architecture

model = tf.keras.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),  # Flatten image to vector
    layers.Dense(1024),  # First dense layer with 1024 neurons
    BatchNormalization(),  # Normalize inputs for stability
    layers.Activation('relu'),  # Activation function
    layers.Dropout(0.3),  # Dropout for regularization

    layers.Dense(512),  # Second dense layer
    BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(256),  # Third dense layer
    BatchNormalization(),
    layers.Activation('relu'),

    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])


# 5. Compile model and set learning rate

# Compile model with Adam optimizer and cross-entropy loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])  # Track accuracy during training

# Learning rate scheduler: reduce LR if validation loss doesn't improve for 3 epochs
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)


# 6. Training

# Train the model using augmented data
history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                    steps_per_epoch=len(x_train) // 128,
                    epochs=30,
                    validation_data=(x_test, y_test),
                    callbacks=[lr_scheduler])  # Use learning rate scheduler during training


# 7. Evaluation

# Evaluate model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nMLP Model Test Accuracy: {test_acc:.4f}")

# Predict class probabilities for test set
y_pred = model.predict(x_test)
# Convert probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

# Print performance metrics
print(f"\nModel Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# 8. Visualization

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for MLP Model on CIFAR-10')
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, digits=4))

# Accuracy and Loss Curves
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Visualize Some Misclassified Images

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

misclassified_idxs = np.where(y_pred_classes != y_test)[0]
plt.figure(figsize=(12, 12))
for i, idx in enumerate(misclassified_idxs[:16]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_test[idx])
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.suptitle('Examples of Misclassified Images by MLP Model', fontsize=16)
plt.tight_layout()
plt.show()
