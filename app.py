"""Desktop demo: open a traffic-sign image and see what the neural network recognises.

    python app.py
"""
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from common import get_device, load_models
from predict import classify

THRESHOLD = 0.7


class App:
    def __init__(self, root):
        self.root = root
        self.dev = get_device()
        self.models = load_models(self.dev)
        root.title("Traffic Sign Recognition")
        root.geometry("560x560")
        root.configure(bg="#f4f5f7")

        ttk.Label(root, text="Traffic Sign Recognition", font=("Segoe UI", 18, "bold")).pack(pady=(18, 2))
        ttk.Label(root, text=f"Ensemble of {len(self.models)} CNN + Spatial Transformer models "
                             f"({self.dev.type.upper()})", font=("Segoe UI", 9)).pack()
        ttk.Button(root, text="Open image...", command=self.open_image).pack(pady=14)

        self.canvas = tk.Label(root, bg="#f4f5f7")
        self.canvas.pack()
        self.result = tk.Label(root, text="", font=("Segoe UI", 15, "bold"), bg="#f4f5f7")
        self.result.pack(pady=(14, 4))
        self.details = tk.Label(root, text="", font=("Segoe UI", 10), bg="#f4f5f7", justify="left")
        self.details.pack()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.ppm *.bmp *.webp")])
        if not path:
            return
        img = Image.open(path).convert("RGB")
        preview = img.copy()
        preview.thumbnail((240, 240))
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas.configure(image=self.photo)

        top = classify(self.models, img, self.dev)
        name, conf = top[0]
        if conf >= THRESHOLD:
            self.result.configure(text=f"{name}  ({conf * 100:.1f}%)", fg="#1a7f37")
        else:
            self.result.configure(text=f"UNCERTAIN - best guess: {name} ({conf * 100:.1f}%)", fg="#c4511a")
        self.details.configure(text="\n".join(f"{i + 2}. {n}  ({c * 100:.2f}%)" for i, (n, c) in enumerate(top[1:])))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
