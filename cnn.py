# Import required libraries
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns

# Configure TensorFlow to dynamically allocate GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU found")
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("No GPU found, using CPU")


# Load a single CIFAR-10 batch file
def load_batch(filepath):
    """Load a single CIFAR-10 batch file"""
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    # Reshape and transpose the image data to (num_samples, 32, 32, 3)
    data = data.reshape((len(data), 3, 32, 32)).transpose(0, 2, 3, 1)
    return data, labels


# Load all CIFAR-10 training and test data from disk
def load_cifar10_data(data_dir):
    """Load CIFAR-10 data from batch files according to professor's description"""
    print("Loading CIFAR-10 data from batches...")

    x_train = []
    y_train = []
    for i in range(1, 6):
        filepath = os.path.join(data_dir, f'data_batch_{i}')
        print(f"Loading training batch {i} of 5: {filepath}")
        data, labels = load_batch(filepath)
        x_train.append(data)
        y_train.extend(labels)

    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)

    # Load test batch
    test_filepath = os.path.join(data_dir, 'test_batch')
    print(f"Loading test batch: {test_filepath}")
    x_test, y_test = load_batch(test_filepath)

    return (x_train, y_train), (x_test, y_test)


# Create CNN model architecture
def create_model(input_shape, num_classes, learning_rate=0.001):
    """Create a simpler CNN model"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Fully connected classification head
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])

    # Compile the model with optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Evaluate model and print performance metrics
def evaluate_model_metrics(model, x_test, y_test, class_names):
    """Evaluates the model's performance with various metrics and confusion matrix"""
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print("\n===== Classification Report =====")
    print(report)

    # Print average metrics
    precision_avg = precision_score(y_true, y_pred_classes, average='macro')
    recall_avg = recall_score(y_true, y_pred_classes, average='macro')
    f1_avg = f1_score(y_true, y_pred_classes, average='macro')

    print("\n===== Average Metrics =====")
    print(f"Precision (avg): {precision_avg:.4f}")
    print(f"Recall (avg): {recall_avg:.4f}")
    print(f"F1-Score (avg): {f1_avg:.4f}")

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return precision_avg, recall_avg, f1_avg, conf_matrix


# Visualize the class distribution in the training and test sets
def analyze_class_distribution(y_train, y_test, class_names):
    """Analyze and visualize class distribution in both datasets"""
    print("\n===== Class Distribution Analysis =====")

    train_counts = np.bincount(y_train.flatten())
    test_counts = np.bincount(y_test.flatten())

    print("Training set class distribution:")
    for i, (name, count) in enumerate(zip(class_names, train_counts)):
        print(f"  {name}: {count} images")

    print("\nTest set class distribution:")
    for i, (name, count) in enumerate(zip(class_names, test_counts)):
        print(f"  {name}: {count} images")

    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(class_names, train_counts)
    plt.title('Training Set Class Distribution')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(class_names, test_counts)
    plt.title('Test Set Class Distribution')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# Main function to load data, train and evaluate the model
def main():
    tf.keras.backend.clear_session()  # Reset TensorFlow state

    try:
        data_dir = '/kaggle/input/cifar1'  # Dataset path

        # Load CIFAR-10 data
        (x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

        # Class labels
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        analyze_class_distribution(y_train, y_test, class_names)

        # Normalize images to range [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # One-hot encode the labels
        num_classes = 10
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

        # Split into training and validation sets
        X_train, X_valid, y_train_cat, y_valid_cat = train_test_split(
            x_train, y_train_cat, test_size=0.2, random_state=42
        )

        input_shape = X_train.shape[1:]

        # Set training parameters
        batch_size = 64
        epochs = 100
        learning_rate = 0.001

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]

        # Create CNN model
        model = create_model(input_shape, num_classes, learning_rate)
        model.summary()

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(X_train)

        # Train the model
        train_generator = datagen.flow(X_train, y_train_cat, batch_size=batch_size)
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_valid, y_valid_cat),
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )

        # Evaluate model on test data
        test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=1)
        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")

        # Detailed evaluation
        evaluate_model_metrics(model, x_test, y_test_cat, class_names)

        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Visualize predictions
        indices = np.random.choice(range(len(x_test)), size=25, replace=False)
        images = x_test[indices]
        true_labels = y_test[indices].flatten()
        predictions = model.predict(images)
        pred_labels = np.argmax(predictions, axis=1)

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i])
            color = "green" if pred_labels[i] == true_labels[i] else "red"
            plt.title(f"{class_names[pred_labels[i]]}", color=color)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Save the model to file
        model.save('cifar10_cnn_model.h5')
        print("Model saved as 'cifar10_cnn_model.h5'")

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()


# Run the main function
if __name__ == "__main__":
    main()
