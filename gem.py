import numpy as np
import tkinter as tk
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk

def preprocess(pil_img):
    arr = np.array(pil_img.convert('L')).astype('float32') / 255.0
    if arr.mean() < 0.5: arr = 1.0 - arr
    mask = arr < 0.95
    if not mask.any(): return np.ones((28, 28), dtype='float32')
    
    y, x = np.where(mask)
    h, w = arr.shape
    y1, y2 = max(y.min()-8, 0), min(y.max()+8, h)
    x1, x2 = max(x.min()-8, 0), min(x.max()+8, w)
    
    crop = Image.fromarray((arr[y1:y2, x1:x2] * 255).astype('uint8'))
    scale = 20.0 / max(crop.size)
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.BILINEAR)
    
    final = np.ones((28, 28), dtype='float32')
    py, px = (28 - crop.height) // 2, (28 - crop.width) // 2
    final[py:py+crop.height, px:px+crop.width] = np.array(crop) / 255.0
    return final

class DigitApp:
    def __init__(self, root):
        root.title("MNIST Recognizer")
        self.cv = tk.Canvas(root, width=300, height=300, bg='white')
        self.cv.grid(row=0, column=0, rowspan=4, padx=10, pady=10)
        self.cv.bind("<B1-Motion>", self.draw_digit)
        
        self.img = Image.new("L", (300, 300), 255)
        self.draw = ImageDraw.Draw(self.img)
        
        tk.Button(root, text="Predict", command=self.predict, width=15).grid(row=0, column=1)
        tk.Button(root, text="Clear", command=self.clear, width=15).grid(row=1, column=1)
        tk.Button(root, text="Quit", command=root.destroy, width=15).grid(row=2, column=1)
        
        self.lbl = tk.Label(root, font=("Helvetica", 16))
        self.lbl.grid(row=3, column=1)
        self.view = tk.Label(root)
        self.view.grid(row=4, column=0, columnspan=2)

    def draw_digit(self, e):
        r = 12
        self.cv.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, fill='black')
        self.draw.ellipse([e.x-r, e.y-r, e.x+r, e.y+r], fill=0)

    def clear(self):
        self.cv.delete("all")
        self.draw.rectangle([0, 0, 300, 300], fill=255)
        self.lbl.config(text="")
        self.view.config(image='')

    def predict(self):
        processed = preprocess(self.img)
        show = ImageTk.PhotoImage(Image.fromarray((processed*255).astype('uint8')).resize((140, 140), 0))
        self.view.config(image=show)
        self.view.image = show
        
        probs = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
        self.lbl.config(text=f"Pred: {np.argmax(probs)} ({np.max(probs):.2f})")

if __name__ == "__main__":
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()