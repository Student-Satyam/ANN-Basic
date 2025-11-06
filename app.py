# app_draw.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load trained ANN model
model = load_model("digit_ann_model.h5")

st.title("✍️ Draw a Digit - ANN Recognition (By Satyam)")

# Info message for Project Version
st.info(
    "📌 This is **Version 1** of my Handwritten Digit Recognition Project.\n\n"
    "This version uses **ANN (Artificial Neural Network)** which may give incorrect predictions sometimes.\n\n"
    "✅ In Future: I will upgrade to **CNN (Convolutional Neural Network)** for higher accuracy and better real handwritten digit recognition. 🚀"
)

st.write("Draw a digit (0–9) in the box below and click **Predict** 👇")

# Create a drawing canvas
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas image to PIL
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

        # Invert (white → black) for MNIST format
        img = ImageOps.invert(img)

        # Resize to 28x28
        img = img.resize((28, 28))

        # Normalize and reshape
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 28, 28)

        # Predict digit
        prediction = model.predict(img_arr)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"✅ Predicted Digit: **{digit}**")
        st.write(f"🎯 Confidence: **{confidence:.2f}%**")
    else:
        st.warning("⚠️ Please draw a digit first!")
