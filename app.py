import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import gdown

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.title("🐶🐱 Cat vs Dog Classifier")
st.markdown("Upload an image of a **cat or dog**, and the model will predict what it is.")

@st.cache_resource
def load_model():
    model_path = "model.h5"
    try:
        if not os.path.exists(model_path):
            url = "https://drive.google.com/uc?id=1ZszVy2n7MFrkhbuA_xfB1bCdKNXOtTI8"
            gdown.download(url, model_path, quiet=False)
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}. Please ensure the model is accessible.")
        return None

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((64, 64))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if model:
        prediction = model.predict(img)[0][0]
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        st.success(f"Prediction: **{label}** (Confidence: {prediction:.2f})")
    else:
        st.warning("Model not loaded properly. Please try again later.")