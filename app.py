# app.py
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import numpy as np
import io

st.title("MNIST Digit Recognition (No TensorFlow)")

# Step 1: Train a tiny model (RandomForest)
@st.cache_data
def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

model = train_model()

# Step 2: Upload handwritten digit
uploaded_file = st.file_uploader("Upload a handwritten digit image (28x28 grayscale)", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("L")
    img_resized = img.resize((8,8))  # match sklearn digits size
    st.image(img_resized, caption="Resized to 8x8", use_column_width=False)

    # Convert image to sklearn format
    img_array = np.array(img_resized)
    img_scaled = (16 - (img_array / 255.0 * 16)).flatten().reshape(1,-1)

    # Predict
    pred = model.predict(img_scaled)
    st.success(f"Predicted Digit: {pred[0]}")
