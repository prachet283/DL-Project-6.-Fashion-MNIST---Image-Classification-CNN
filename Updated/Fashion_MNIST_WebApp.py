import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# Class labels for Fashion MNIST
class_names = [
    'T-shirt/top 👕', 'Trouser 👖', 'Pullover 🧥', 'Dress 👗', 'Coat 🧥',
    'Sandal 🩴', 'Shirt 👔', 'Sneaker 👟', 'Bag 👜', 'Ankle boot 👢'
]

# Load the model
model = load_model("fashion_mnist_model.h5")

# Title
st.title("👗 Fashion MNIST Classification Web App")

# Upload file
uploaded_file = st.file_uploader("Upload a fashion image (28x28 grayscale preferred)", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_image(uploaded_image):
    image_pil = uploaded_image.convert('L')  # convert to grayscale
    img_resized = image_pil.resize((28, 28))  # resize to 28x28
    img_array = img_to_array(img_resized) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index]) * 100

    return class_names[class_index], round(confidence, 2), np.array(image_pil)

# Display result
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    label, confidence, display_image = predict_image(image)

    st.image(display_image, caption=f"Prediction: {label} ({confidence}%)", use_column_width=True)
    st.success(f"✅ Prediction: {label} with {confidence}% confidence")
