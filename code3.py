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
