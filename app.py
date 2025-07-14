import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# Set Streamlit page settings
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.title("🐶🐱 Cat vs Dog Classifier")
st.markdown("Snap. Upload. I’ll tell you: cat or dog? 🐾")

# Load model only once using caching
@st.cache_resource
def load_model():
    model_path = "model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("❌ Model file not found! Please make sure 'model.h5' is in the app folder.")
        return None

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((150, 150))  # Match training shape
    img = np.array(img)

    # If image has alpha channel (RGBA), convert to RGB
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = img / 255.0  # Normalize
    img = np.reshape(img, (1, 150, 150, 3))  # Reshape for prediction

    # Prediction
    if model:
        prediction = model.predict(img)[0][0]
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")
    else:
        st.warning("Model not loaded properly.")
        
        