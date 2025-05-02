import tkinter as tk
from tkinter import filedialog, Label
import cv2
from PIL import Image, ImageTk
import numpy as np
import joblib

model = joblib.load('emotion_model.pkl')

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48)).flatten().reshape(1, -1)
    return model.predict(img)[0]

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        emotion = predict_image(file_path)
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        result_label.config(text=f"Predicted Emotion: {emotion}")

root = tk.Tk()
root.title("Emotion Detector")

panel = Label(root)
panel.pack()

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack()

result_label = Label(root, text="Predicted Emotion: None", font=("Arial", 14))
result_label.pack()

root.mainloop()
