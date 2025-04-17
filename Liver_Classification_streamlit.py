import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to run on CPU

import streamlit as st
st.set_page_config(page_title="Liver Fibrosis Classifier", layout="centered")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ==========================
# CONFIGURATION
# ==========================
IMG_SIZE = (224, 224)
CLASS_ORDER = ['F0', 'F1', 'F2', 'F3', 'F4']

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/content/Dense_model.path/kaggle/working/Dense_model_.h5")

model = load_model()

# ==========================
# PREPROCESS IMAGE
# ==========================
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMG_SIZE)
    image_array = img_to_array(image) / 255.0  # Normalize to [0,1]
    return np.expand_dims(image_array, axis=0)

# ==========================
# PREDICT FIBROSIS
# ==========================
def predict_fibrosis(model, img_tensor, class_labels):
    prediction = model.predict(img_tensor)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    fibrosis_status = "🟢 No Fibrosis" if predicted_label == "F0" else "🔴 Yes Fibrosis"
    return fibrosis_status, predicted_label, prediction

# # ==========================
# # SALIENCY MAP
# # ==========================
# def generate_saliency_map(model, img_tensor):
#     img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

#     with tf.GradientTape() as tape:
#         tape.watch(img_tensor)
#         preds = model(img_tensor)
#         class_idx = tf.argmax(preds[0])
#         loss = preds[:, class_idx]

#     grads = tape.gradient(loss, img_tensor)
#     saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

#     saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
#     return saliency.numpy()

# ==========================
# STREAMLIT UI
# ==========================
st.title("🧬 Liver Fibrosis Stage Classification")
st.write("Upload a liver ultrasound image to classify the fibrosis stage .")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    # Preprocess
    img_tensor = preprocess_image(image)

    # Prediction
    fibrosis_status, predicted_stage, preds = predict_fibrosis(model, img_tensor, CLASS_ORDER)
    confidence = float(np.max(preds))

    # Colored display
    stage_color = {
        "F0": "🟢",
        "F1": "🟡",
        "F2": "🟠",
        "F3": "🟠",
        "F4": "🔴"
    }

    st.markdown(f"### 🩺 Predicted Fibrosis Stage: **{stage_color[predicted_stage]} {predicted_stage}**")
    st.markdown(f"### 🔍 Fibrosis Status: **{fibrosis_status}**")
    st.markdown(f"**🔢 Confidence:** `{confidence * 100:.2f}%`")

   
