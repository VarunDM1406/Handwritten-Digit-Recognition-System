import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

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

    print("\nTraining model… This may take a few minutes…\n")

    model.fit(x_train, y_train, epochs=6, batch_size=64, validation_data=(x_test, y_test))

    model.save("mnist_cnn_model.h5")
    print("\nModel saved as mnist_cnn_model.h5\n")

    return model

import os
if os.path.exists("mnist_cnn_model.h5"):
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
else:
    model = train_model()

def preprocess(img_pil):
    img = img_pil.resize((28,28))
    img = img.convert("L")
    img = np.array(img)

    img = 255 - img

    img = img / 255.0

    return img.reshape(1,28,28,1)

class DrawApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Digit Recognizer - MNIST")
        self.window.resizable(False, False)

        self.canvas_size = 280  # 10x bigger for drawing
        self.canvas = tk.Canvas(self.window, bg="black",
                                width=self.canvas_size,
                                height=self.canvas_size)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Pillow image to store drawing
        self.img = Image.new("RGB", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.img)

        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<Button-1>", self.draw_stroke)

        # Buttons
        tk.Button(self.window, text="Predict",
                  command=self.predict_digit, width=15).grid(row=1, column=0, pady=5)

        tk.Button(self.window, text="Clear",
                  command=self.clear_canvas, width=15).grid(row=2, column=0, pady=5)

        # Prediction Label
        self.result_label = tk.Label(self.window, text="Draw a digit (0–9)", font=("Arial", 18))
        self.result_label.grid(row=3, column=0, pady=10)

        self.window.mainloop()

    def draw_stroke(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle((0,0,self.canvas_size,self.canvas_size), fill="black")
        self.result_label.config(text="Draw a digit (0–9)")

    def predict_digit(self):
        img = self.img.copy()
        img = img.convert("L")

        # crop to bounding box
        arr = np.array(img)
        coords = cv2.findNonZero(arr)
        if coords is None:
            self.result_label.config(text="(No digit drawn)")
            return

        x, y, w, h = cv2.boundingRect(coords)
        cropped = img.crop((x, y, x+w, y+h))

        processed = preprocess(cropped)
        pred = model.predict(processed)[0]
        digit = np.argmax(pred)
        confidence = pred[digit] * 100

        self.result_label.config(
            text=f"Prediction: {digit}  ({confidence:.2f}%)"
        )
if __name__ == "__main__":
    DrawApp()