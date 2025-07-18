import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mnist_cnn.h5")

st.title("🖌️ Handwritten Digit Recognition")
st.markdown("Draw a digit (0–9) below:")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = Image.fromarray((img[:, :, 0]).astype(np.uint8))  # Grayscale
    img = img.resize((28, 28)).convert('L')
    img_arr = np.array(img).reshape(1, 28, 28, 1) / 255.0

    if st.button("Predict"):
        pred = model.predict(img_arr)
        st.success(f"Predicted Digit: {np.argmax(pred)}")
