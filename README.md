## 🧠 Liver Fibrosis

Liver fibrosis is a medical condition characterized by the excessive accumulation of extracellular matrix proteins, including collagen, that occurs in most types of chronic liver diseases. It results from sustained wound-healing responses to chronic liver injury caused by factors such as:

![F0](https://github.com/user-attachments/assets/f3419c5c-f206-4d10-bca4-c2f8fd1b710e)

- Viral hepatitis (Hepatitis B, C)
- Alcohol abuse
- Non-alcoholic fatty liver disease (NAFLD)
- Autoimmune liver diseases

If left undetected or untreated, liver fibrosis can progress to cirrhosis, liver failure, and even liver cancer.

---

## ❗ The Challenge

Traditional diagnosis of liver fibrosis involves invasive procedures such as **liver biopsies**, which are:

- Painful  
- Costly  
- Risky due to potential complications  
- Subject to sampling errors  

There is a growing demand for **non-invasive, accurate, and efficient** techniques to detect and stage liver fibrosis. Medical imaging, particularly **CT scans** and **ultrasound**, has become an essential tool in this domain. However, interpretation of medical images requires expertise and time.

---

## 💡 Our Solution

In this project, we propose a **deep learning-based system** to **automatically classify liver fibrosis stages** from medical imaging data. Using convolutional neural networks (CNNs), specifically a **MobileNet-based architecture**, we aim to:

- Analyze liver CT scan images  
- Detect features associated with fibrosis  
- Classify the condition into stages (e.g., normal, mild, moderate, severe)

---
We now extend the system to support several additional powerful models for better performance:

- **ResNet-50**: A residual network for improved learning on deeper networks.
- **VGG-19**: A classic model known for its simplicity and high performance.
- **NasNetMobile**: A mobile-friendly model for resource-efficient prediction.
- **DenseNet-121**: Uses dense connections between layers to enhance feature reuse.
- **InceptionResNetV2**: Combines the strengths of Inception and ResNet architectures.
- **Unet++**: A more complex U-Net for image segmentation tasks, especially in medical imaging.
---
## 🖼️ Application Interface

Our Streamlit application provides a **user-friendly interface** to classify liver fibrosis stages based on ultrasound images.

### 🔍 How It Works

1. The user uploads a **liver ultrasound image** (JPG, JPEG, or PNG).
2. The application displays the image for confirmation.
3. A **trained deep learning model** processes the image.
4. The model outputs the **predicted fibrosis stage** (e.g., F0, F1, F2, ..., F4).

### ✅ Example

Below is an example of the interface after uploading an image:

![Streamlit App Interface](path/to/screenshot.png)

- **Uploaded File:** `F0.jpg`
- **Predicted Fibrosis Stage:** 🟢 `F0` (No fibrosis)

This enables doctors and researchers to quickly and efficiently determine the fibrosis stage without requiring invasive procedures.

---

## 🧠 Deployment Streamlit Code - Liver Fibrosis Classifier

The backend of the liver fibrosis classification application is powered by TensorFlow, Streamlit, and a pre-trained deep learning model. Here is an overview of how the backend works:

### Key Components:

1. **CUDA Configuration**:
    - TensorFlow is configured to run on the CPU by setting the `CUDA_VISIBLE_DEVICES` environment variable to `-1`, ensuring compatibility in environments without GPU support.
    ```python
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to run on CPU
    ```

2. **Streamlit Setup**:
    - Streamlit is used to create an interactive web interface for the user, centered around the liver fibrosis classification task.
    ```python
    import streamlit as st
    st.set_page_config(page_title="Liver Fibrosis Classifier", layout="centered")
    ```

3. **Model Loading**:
    - The `load_model()` function is used to load the pre-trained Keras model that classifies liver fibrosis stages. The model is cached for performance optimization.
    ```python
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("/content/Dense_model.path/kaggle/working/Dense_model_.h5")
    model = load_model()
    ```

4. **Image Preprocessing**:
    - The `preprocess_image()` function resizes the uploaded image to 224x224 pixels, normalizing the pixel values to [0, 1] before passing it to the model for prediction.
    ```python
    def preprocess_image(image: Image.Image) -> np.ndarray:
        image = image.resize(IMG_SIZE)
        image_array = img_to_array(image) / 255.0  # Normalize to [0,1]
        return np.expand_dims(image_array, axis=0)
    ```

5. **Prediction Function**:
    - The `predict_fibrosis()` function predicts the fibrosis stage by using the model's output and provides a corresponding fibrosis status (whether fibrosis is present) based on the prediction.
    ```python
    def predict_fibrosis(model, img_tensor, class_labels):
        prediction = model.predict(img_tensor)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]
        fibrosis_status = "🟢 No Fibrosis" if predicted_label == "F0" else "🔴 Yes Fibrosis"
        return fibrosis_status, predicted_label, prediction
    ```

6. **Streamlit UI Integration**:
    - The Streamlit app interface allows users to upload an image, displays the image, and shows the model's prediction along with the confidence level.
    ```python
    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

        # Preprocess and Predict
        img_tensor = preprocess_image(image)
        fibrosis_status, predicted_stage, preds = predict_fibrosis(model, img_tensor, CLASS_ORDER)
        confidence = float(np.max(preds))
        st.markdown(f"### 🩺 Predicted Fibrosis Stage: **{predicted_stage}**")
        st.markdown(f"### 🔍 Fibrosis Status: **{fibrosis_status}**")
        st.markdown(f"**🔢 Confidence:** `{confidence * 100:.2f}%`")
    ```

---

## 🧪 Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV / PIL  
- Streamlit (for interactive web-based deployment)  
- scikit-learn (for preprocessing and evaluation)
