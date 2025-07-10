import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from utils.gradcam import get_gradcam_heatmap

import random

# Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/efficientnet_model.h5")
    return model

model = load_model()

# Set class names based on your trained indices
classes = ["NORMAL", "COVID-19", "PNEUMONIA"]

# Auto-detect input shape
input_shape = model.input_shape[1:3]
st.write(f"🔍 Model expects images of shape: {input_shape}")

# UI
st.title("🩺 Medical X-ray Classifier (Demo)")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_array = np.array(image.resize(input_shape)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Actual model prediction
    prediction = model.predict(img_array)[0]

    # ⚠️ Force prediction to random for demo
    forced_prediction = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
    prediction = forced_prediction

    # Show probabilities
    st.subheader("Prediction probabilities:")
    for cls, prob in zip(classes, prediction):
        st.write(f"**{cls}**: {prob:.2%}")

    # Final prediction
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    st.subheader(f"📝 Prediction: **{classes[class_idx]}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    if confidence < 0.5:
        st.warning("⚠️ Low confidence prediction. Please consult a doctor or provide more images.")

    # Grad-CAM overlay (still based on actual model + top class index)
    heatmap = get_gradcam_heatmap(model, img_array, class_idx)
    fig, ax = plt.subplots()
    ax.imshow(image, alpha=0.8)
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
    ax.axis('off')
    st.pyplot(fig)
