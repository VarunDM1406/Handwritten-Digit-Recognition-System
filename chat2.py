# ---------------------------------------------------------
# FINAL MNIST DIGIT RECOGNIZER
# HIGH ACCURACY + FIXED PREPROCESSING
# ---------------------------------------------------------

import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import os

# ---------------------------------------------------------
# TRAIN MODEL IF NOT AVAILABLE
# ---------------------------------------------------------

def train_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining model...")

    model.fit(x_train, y_train, epochs=6, batch_size=64,
              validation_data=(x_test, y_test))

    model.save("mnist_cnn_model.h5")
    print("\nModel saved as mnist_cnn_model.h5\n")

    return model


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

if os.path.exists("mnist_cnn_model.h5"):
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
else:
    model = train_model()


# ---------------------------------------------------------
# FIXED MNIST PREPROCESSING (THIS FIXES 3 → 8 PROBLEM)
# ---------------------------------------------------------

def preprocess(img_pil):
    # Convert to grayscale
    img = img_pil.convert("L")
    img = np.array(img)

    # Invert (MNIST is white digit on black)
    img = 255 - img

    # Remove weak noise
    img[img < 40] = 0

    # Find digit bounding box
    coords = cv2.findNonZero(img)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    # Crop
    img = img[y:y+h, x:x+w]

    # Resize proportionally to max 20px
    max_dim = max(w, h)
    scale = 20 / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Build 28×28 canvas
    canvas = np.zeros((28, 28), dtype="uint8")

    # Center the digit
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img

    # Normalize
    canvas = canvas.astype("float32") / 255.0
    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas


# ---------------------------------------------------------
# TKINTER DRAWING INTERFACE
# ---------------------------------------------------------

class DrawApp:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("MNIST Digit Recognizer")
        self.win.resizable(False, False)

        self.size = 280
        self.canvas = tk.Canvas(self.win, bg="black",
                                width=self.size, height=self.size)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.img = Image.new("RGB", (self.size, self.size), "black")
        self.draw = ImageDraw.Draw(self.img)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        tk.Button(self.win, text="Predict",
                  command=self.predict_digit, width=20).grid(row=1, column=0, pady=5)

        tk.Button(self.win, text="Clear",
                  command=self.clear_canvas, width=20).grid(row=2, column=0, pady=5)

        self.result = tk.Label(self.win, text="Draw a digit", font=("Arial", 18))
        self.result.grid(row=3, column=0, pady=10)

        self.win.mainloop()

    def paint(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, self.size, self.size), fill="black")
        self.result.config(text="Draw a digit")

    def predict_digit(self):
        img = self.img.copy()
        img = img.convert("L")

        processed = preprocess(img)
        if processed is None:
            self.result.config(text="No digit detected")
            return

        pred = model.predict(processed)[0]
        digit = np.argmax(pred)
        confidence = pred[digit] * 100

        self.result.config(text=f"Prediction: {digit} ({confidence:.2f}%)")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    DrawApp()