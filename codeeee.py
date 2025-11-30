#!/usr/bin/env python3
"""
mnist_cnn_with_gui.py
End-to-end MNIST CNN + Tkinter drawing panel.

Usage:
    # Train if no saved model exists (default)
    python mnist_cnn_with_gui.py

    # Force training (retrain & overwrite mnist_cnn_final.h5)
    python mnist_cnn_with_gui.py --train

    # Skip evaluation (train/load model then open GUI directly)
    python mnist_cnn_with_gui.py --no-eval
"""

import os
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Metrics
from sklearn.metrics import confusion_matrix, classification_report

# GUI
import tkinter as tk
from PIL import Image, ImageDraw

# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MODEL_PATH = "mnist_cnn_final.h5"

# ------------------------
# Build model function
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

# ------------------------
# Training routine
# ------------------------
def train_and_save_model(x_train, y_train_cat, x_test, y_test_cat, epochs=20, batch_size=128):
    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.08
    )
    datagen.fit(x_train)

    model = build_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=batch_size, seed=SEED),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_cat),
        callbacks=callbacks,
        verbose=2
    )

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model, history

# ------------------------
# Evaluation + plotting
# ------------------------
def evaluate_and_plot(model, x_test, y_test, y_test_cat, history=None):
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    if history is not None:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch'); plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epoch'); plt.legend()
        plt.tight_layout()
        plt.show()

    # Predictions
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=False)
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=True)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm.astype('float') / cm_sum
        print("Normalized confusion matrix")
    else:
        cm_to_plot = cm
        print("Confusion matrix, without normalization")

    plt.figure(figsize=(6,5))
    plt.imshow(cm_to_plot, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_to_plot.max() / 2.
    for i, j in itertools.product(range(cm_to_plot.shape[0]), range(cm_to_plot.shape[1])):
        plt.text(j, i, format(cm_to_plot[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_to_plot[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# ------------------------
# Tkinter drawing GUI
# ------------------------
def start_drawing_gui(model):
    # NOTE: This opens a native window. Run script from terminal/IDE (not inside Jupyter).
    canvas_width = 300
    canvas_height = 300
    white = (255, 255, 255)

    window = tk.Tk()
    window.title("Digit Recognition Panel")
    window.resizable(0, 0)

    # Create a PIL image we can draw on, and a draw object
    image = Image.new("L", (canvas_width, canvas_height), color=1)
    draw = ImageDraw.Draw(image)

    def paint(event):
        # Draw thick black circles where the mouse moves (simulates brush)
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
        draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas():
        canvas.delete("all")
        draw.rectangle([0, 0, canvas_width, canvas_height], fill=white)
        result_label.config(text="")

    def predict_digit():
        # Resize to 28x28 and invert/scale appropriately
        img = image.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        # If background is black for some environments, invert: (optional)
        # img_array = 1.0 - img_array
        img_array = img_array.reshape(1, 28, 28, 1)
        probs = model.predict(img_array)
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        result_label.config(text=f"Predicted: {pred}  (conf: {conf:.2f})", font=("Arial", 16))

    # Widgets
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
    canvas.grid(row=0, column=0, columnspan=3, pady=5, padx=5)
    canvas.bind("<B1-Motion>", paint)

    predict_btn = tk.Button(window, text="Predict", width=12, command=predict_digit)
    predict_btn.grid(row=1, column=0, pady=5)

    clear_btn = tk.Button(window, text="Clear", width=12, command=clear_canvas)
    clear_btn.grid(row=1, column=1, pady=5)

    quit_btn = tk.Button(window, text="Quit", width=12, command=window.destroy)
    quit_btn.grid(row=1, column=2, pady=5)

    result_label = tk.Label(window, text="", font=("Arial", 14))
    result_label.grid(row=2, column=0, columnspan=3, pady=8)

    window.mainloop()


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Force retrain the model")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation & plotting")
    args = parser.parse_args()

    # ------------------------
    # Load MNIST
    # ------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)  # (n,28,28,1)
    x_test  = np.expand_dims(x_test, -1)

    num_classes = 10
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    model = None
    history = None

    # Decide whether to train or load
    if args.train or not os.path.exists(MODEL_PATH):
        print("Training model (this may take a while)...")
        model, history = train_and_save_model(x_train, y_train_cat, x_test, y_test_cat, epochs=20, batch_size=128)
    else:
        print(f"Loading model from {MODEL_PATH} ...")
        model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate and plot (unless user disabled it)
    if not args.no_eval:
        evaluate_and_plot(model, x_test, y_test, y_test_cat, history=history)

    # Start GUI drawing panel
    print("Opening drawing panel... (use mouse to draw, then click Predict)")
    start_drawing_gui(model)
#!/usr/bin/env python3
"""
mnist_gui_preprocess.py
Improved MNIST GUI with robust preprocessing:
- auto invert detection
- bounding-box crop + aspect-ratio resize to 20x20
- centered into 28x28 (MNIST-style)
- shows the 28x28 image the model sees
"""

import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf

MODEL_PATH = "mnist_cnn_final.h5"

# Load model (ensure you have trained and saved mnist_cnn_final.h5)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train and save model first.")
model = tf.keras.models.load_model(MODEL_PATH)

# Canvas size and brush
CANVAS_SIZE = 300
BRUSH_RADIUS = 12  # thicker brush for clarity

# --------- Preprocessing utilities (MNIST-style) ----------
def preprocess_pil_image(pil_img):
    """
    Convert PIL grayscale image (L) -> normalized 28x28 numpy array matching MNIST preprocessing:
    - Auto-detect and invert if needed (so digit strokes are dark, background white)
    - Crop to bounding box of ink
    - Resize to fit inside 20x20 box while preserving aspect ratio
    - Center the 20x20 inside 28x28 with equal padding
    Returns array shape (28,28) with values in [0,1].
    """
    # ensure L mode
    img = pil_img.convert('L')
    arr = np.array(img).astype('float32') / 255.0  # background ~1, strokes ~0 (if normal)
    # Auto-invert: if mean intensity is low (dark image), invert to make background ~1
    if arr.mean() < 0.5:
        arr = 1.0 - arr

    # Identify foreground (ink) where intensity < 0.95 (tolerant)
    mask = arr < 0.95
    if not mask.any():
        # empty -> return blank 28x28 (all ones)
        return np.ones((28, 28), dtype='float32')

    ys, xs = np.where(mask)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()

    # crop with small padding
    pad = 8
    minx = max(minx - pad, 0)
    miny = max(miny - pad, 0)
    maxx = min(maxx + pad, arr.shape[1] - 1)
    maxy = min(maxy + pad, arr.shape[0] - 1)

    cropped = arr[miny:maxy+1, minx:maxx+1]

    # Resize cropped to fit in 20x20 box
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    cropped_img = Image.fromarray((cropped * 255).astype('uint8')).resize((new_w, new_h), Image.BILINEAR)
    resized = np.array(cropped_img).astype('float32') / 255.0

    # Place resized into 28x28 center
    final = np.ones((28, 28), dtype='float32')
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Final ensure digits are dark: (already arranged so strokes ~0)
    return final

# --------- Tkinter GUI ----------
class DigitApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Recognizer (Improved Preprocessing)")
        master.resizable(0, 0)

        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=4, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # PIL image to draw on (grayscale L)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        self.predict_btn = tk.Button(master, text="Predict", command=self.predict, width=18)
        self.predict_btn.grid(row=0, column=1, padx=10, pady=6)

        self.clear_btn = tk.Button(master, text="Clear", command=self.clear, width=18)
        self.clear_btn.grid(row=1, column=1, padx=10, pady=6)

        self.quit_btn = tk.Button(master, text="Quit", command=master.destroy, width=18)
        self.quit_btn.grid(row=2, column=1, padx=10, pady=6)

        # Prediction label
        self.result_label = tk.Label(master, text="", font=("Helvetica", 16))
        self.result_label.grid(row=3, column=1, padx=10, pady=6)

        # Panel to show the 28x28 image the model sees (upscaled)
        self.seen_img_label = tk.Label(master)
        self.seen_img_label.grid(row=4, column=0, columnspan=2, pady=(0,10))

    def paint(self, event):
        x, y = event.x, event.y
        # Draw a filled oval on both PIL image and Tk canvas
        x1, y1 = x - BRUSH_RADIUS, y - BRUSH_RADIUS
        x2, y2 = x + BRUSH_RADIUS, y + BRUSH_RADIUS
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.result_label.config(text="")
        self.seen_img_label.config(image='')

    def predict(self):
        # Preprocess the PIL image into 28x28 normalized array
        final28 = preprocess_pil_image(self.image)  # shape (28,28) values in [0,1], digit dark (low values)
        # Show what the model sees (upscaled for user)
        display_img = Image.fromarray((final28 * 255).astype('uint8')).resize((140, 140), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(display_img)
        self.seen_img_label.config(image=tk_img)
        # Keep a reference to avoid garbage collection
        self.seen_img_label.image = tk_img

        # Prepare batch and predict
        x = final28.reshape(1, 28, 28, 1).astype('float32')
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        self.result_label.config(text=f"Predicted: {pred}  (confidence: {conf:.2f})")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    print("Open the drawing panel. Draw a digit, then click Predict.")
    root.mainloop()

if __name__ == "__main__":
    main()