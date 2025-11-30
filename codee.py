"""
mnist_cnn.py
End-to-end MNIST handwritten digit recognition using TensorFlow/Keras.

Usage:
    python mnist_cnn.py
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
import itertools

# ------------------------
# Reproducibility settings
# ------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------
# Load MNIST dataset
# ------------------------
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print shapes
print("Raw data shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# ------------------------
# Preprocessing
# ------------------------
# MNIST images are 28x28 greyscale; expand dims to (28,28,1)
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

x_train = np.expand_dims(x_train, -1)  # shape -> (num, 28, 28, 1)
x_test  = np.expand_dims(x_test, -1)

# One-hot encode targets
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)

print("Processed shapes:", x_train.shape, y_train_cat.shape, x_test.shape, y_test_cat.shape)

# ------------------------
# Simple data augmentation (optional)
# ------------------------
datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.08
)
datagen.fit(x_train)

# ------------------------
# Build CNN model
# ------------------------
def build_model(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model()
model.summary()

# ------------------------
# Compile
# ------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------
# Callbacks
# ------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("mnist_cnn_best.h5", save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]

# ------------------------
# Train
# ------------------------
batch_size = 128
epochs = 20

history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=batch_size, seed=SEED),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test_cat),
    callbacks=callbacks,
    verbose=2
)

# ------------------------
# Save final model
# ------------------------
model.save("mnist_cnn_final.h5")
print("Saved model to mnist_cnn_final.h5")

# ------------------------
# Plot training curves
# ------------------------
def plot_history(history):
    plt.figure(figsize=(12,4))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    # loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history)

# ------------------------
# Evaluate on test set
# ------------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# ------------------------
# Predictions and analysis
# ------------------------
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix shape:", cm.shape)

# Pretty print classification report
report = classification_report(y_test, y_pred, digits=4)
print("Classification report:\n", report)

# ------------------------
# Utility: plot confusion matrix
# ------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm_sum
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=False)
plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=True)

# ------------------------
# Show some sample predictions
# ------------------------
def show_sample_predictions(x, y_true, y_pred, n=10):
    idxs = np.random.choice(range(len(x)), n, replace=False)
    plt.figure(figsize=(12,4))
    for i, idx in enumerate(idxs):
        plt.subplot(1, n, i+1)
        plt.imshow(x[idx].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
    plt.show()

show_sample_predictions(x_test, y_test, y_pred, n=10)

# ------------------------
# Example single image inference
# ------------------------
def predict_single_image(model, image_28x28):
    # image_28x28 should be shape (28,28) or (28,28,1), pixel values in [0,1]
    img = image_28x28.astype('float32') / 1.0
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)  # batch dimension
    probs = model.predict(img)
    pred = np.argmax(probs, axis=1)[0]
    return pred, probs[0]

# Example using first test image:
pred, prob_vec = predict_single_image(model, x_test[0])
print("First test image: true label =", y_test[0], "predicted =", pred, "probabilities =", np.round(prob_vec,3))