from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tkinter import *
from tkinter import ttk, filedialog

import sys
import os

#File path
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
    )
    if file_path:
        path_var.set(file_path)

model = load_model('MNIST_keras_CNN.h5',compile=False)  

#Prediction
def img_to_num(d):
    try:
        digit = cv2.imread(d)
        if digit is None:
            raise ValueError("Failed to load image. Check the file path or file type.")
        digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
        digit = cv2.resize(digit, (28, 28))
        digit = 255 - digit
        digit = digit / 255.0
        digit = digit.reshape(1, 28, 28, 1)

        predicted = model.predict(digit)
        val = np.argmax(predicted)
        ans_var.set(f"Predicted Value: {val}")
    except Exception as e:
        ans_var.set(f"Error: {e}")

#GUI
root = Tk()
root.title("Digit Recognition")
root.geometry("400x300")

frm = ttk.Frame(root,padding=(20,40))
frm.grid(sticky="nsew")
root.columnconfigure(0, weight=1)
frm.columnconfigure(0, weight=1)

path_var = StringVar()
ans_var = StringVar()
ttk.Button(frm, text="Choose File", command=browse_file).grid(column=0, row=2, pady=10)
ttk.Label(frm, textvariable=path_var, anchor="center").grid(column=0, row=3, pady=10, sticky="ew")
ttk.Button(frm, text="Predict", command=lambda: img_to_num(path_var.get())).grid(column=0, row=4, pady=10)
ttk.Label(frm, textvariable=ans_var, anchor="center", font=("Arial", 12)).grid(column=0, row=5, pady=10, sticky="ew")

root.mainloop()
