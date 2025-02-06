import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

# Load Facial Expression Model
def FacialExpressionModel(json_file, weights_file):
    try:
        with open(json_file, 'r') as file:
            loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize Tkinter Window
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detection')
top.configure(background='#CDCDCD')

# UI Components
label1 = tk.Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = tk.Label(top)

# Load Haarcascade Classifier
haar_path = r'C:/Users/dinesh mishra/OneDrive/Desktop/Emotion-detector/Real-Time-Emotion-detector/haarcascade_frontalface_default.xml'

if not os.path.exists(haar_path):
    print("Error: Haarcascade file not found!")
    label1.configure(text="Error: Haarcascade file missing")
    
facec = cv2.CascadeClassifier(haar_path)

# Load Model
model_path_json = r'C:/Users/dinesh mishra/OneDrive/Desktop/Emotion-detector/Real-Time-Emotion-detector/Model_file.json'
model_path_weights = r'C:/Users/dinesh mishra/OneDrive/Desktop/Emotion-detector/Real-Time-Emotion-detector/model_weights.weights.h5'

if not os.path.exists(model_path_json) or not os.path.exists(model_path_weights):
    print("Error: Model files not found!")
    label1.configure(text="Error: Model files missing")

model = FacialExpressionModel(model_path_json, model_path_weights)

# Emotion Labels
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Emotion Detection Function
def Detect(file_path):
    if model is None:
        label1.configure(foreground="#011638", text="Error: Model not loaded")
        return

    image = cv2.imread(file_path)
    
    if image is None:
        label1.configure(foreground="#011638", text="Error: Image not loaded")
        return

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces with optimized parameters
    faces = facec.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    if len(faces) == 0:
        label1.configure(foreground="#011638", text="No face detected")
        return

    for (x, y, w, h) in faces:
        fc = gray_img[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))  # Resize to match model input

        # Expand dimensions for model prediction
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension

        # Predict Emotion
        pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
        print(f"Predicted Emotion: {pred}")

        label1.configure(foreground="#011638", text=pred)

# Show Detect Button
def show_Detect_button(file_path):
    detect_b = tk.Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

# Upload Image Function
def upload_image():
    file_path = filedialog.askopenfilename()
    
    if not file_path:
        return

    try:
        uploaded = Image.open(file_path)
        uploaded.thumbnail((350, 350))  # Resize for UI
        im = ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')

        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")

# Upload Button
upload = tk.Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

# Pack Components
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)

# Heading
heading = tk.Label(top, text="Emotion Detector", pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

# Run App
top.mainloop()

