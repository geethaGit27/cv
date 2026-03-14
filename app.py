import streamlit as st
from keras.models import load_model
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load model
MODEL_PATH = 'mnist_model.h5'
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    st.error("Model file not found! Train it first.")

st.title("MNIST Digit Recognition")

# Upload an image
uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img_resized = img.resize((28,28))
    st.image(img_resized, caption="Resized Image", use_column_width=False)

    img_array = np.array(img_resized)/255.0
    img_array = img_array.reshape(1,28,28)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")
