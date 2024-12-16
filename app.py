import streamlit as st
import os
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from matplotlib import pyplot as plt
import tempfile
import gdown


# Streamlit App Interface
st.title("Historical Image Restoration & Monument Prediction")

# Paths for Historical Monument Model and Class Indices (these are stored in the same directory as app.py)
MONUMENT_MODEL_PATH = 'indian_monuments_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'

# Load the historical monument classification model
monument_model = load_model(MONUMENT_MODEL_PATH)
monument_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
monument_model.save('indian_monuments_model.h5')

# Paths for Colorization Model (located in the /colorization_models directory)
MODEL_DIR = 'colorization_models'
PROTOTXT = os.path.join(MODEL_DIR, 'colorization_deploy_v2.prototxt')
POINTS = os.path.join(MODEL_DIR, 'pts_in_hull.npy')
COLORIZATION_MODEL = os.path.join(MODEL_DIR, 'colorization_release_v2.caffemodel')

# Function to load the colorization model
def load_colorization_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, COLORIZATION_MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

# Function to colorize the image
def colorize_image(image):
    net = load_colorization_model()
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")

# Function to reduce noise in an image
def reduce_noise(image, method="gaussian"):
    if method == "gaussian":
        return cv2.GaussianBlur(image, ( 5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError("Method not recognized. Choose 'gaussian' or 'median'.")

# Function to generate a mask based on thresholding
def generate_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return mask

# Function to perform inpainting using the generated mask
def inpaint_image(image, mask):
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

# Function to preprocess the image for model input
def preprocess_image(image, model, target_size=None):
    # If target size is None, fetch model's input shape
    if target_size is None:
        target_size = model.input_shape[1:3]  # (height, width) from the model input shape
    # Resize the image to match the model's expected input size
    image_resized = cv2.resize(image, target_size)
    img_array = img_to_array(image_resized) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the monument in the image
def predict_image(image, model, class_indices):
    # Preprocess the image with dynamic resizing based on model's input shape
    img_array = preprocess_image(image, model)  # Dynamic target size fetching

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Map index to class label
    if predicted_class_index in class_indices.values():
        predicted_class_label = [k for k, v in class_indices.items() if v == predicted_class_index][0]
        confidence = predictions[0][predicted_class_index]
    else:
        predicted_class_label = "Unknown"
        confidence = 0.0

    # Display the result
    st.image(image, caption=f"Predicted: {predicted_class_label} ({confidence * 100:.2f}%)")
    return predicted_class_label, confidence


# Streamlit File Upload and Operations
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Error loading image. Please upload a valid image.")
    else:
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", channels="BGR")

        # Operation selection
        operation = st.radio("Choose an operation:", ["Colorize Image", "Reduce Noise", "Inpaint Image", "Predict Monument"])

        if operation == "Colorize Image":
            colorized_image = colorize_image(image)
            st.image(colorized_image, caption="Colorized Image", channels="BGR")
            if st.button("Predict Monument in Colorized Image"):
                with open(CLASS_INDICES_PATH, 'r') as f:
                    class_indices = json.load(f)
                label, confidence = predict_image(colorized_image, monument_model, class_indices)
                st.write(f"Predicted Monument: {label}, Confidence: {confidence * 100:.2f}%")

        elif operation == "Reduce Noise":
            method = st.selectbox("Choose noise reduction method:", ["Gaussian", "Median"])
            denoised_image = reduce_noise(image, method.lower())
            st.image(denoised_image, caption="Denoised Image", channels="BGR")

        elif operation == "Inpaint Image":
            mask = generate_mask(image)
            inpainted_image = inpaint_image(image, mask)
            st.image(inpainted_image, caption="Inpainted Image", channels="BGR")

        elif operation == "Predict Monument":
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
            label, confidence = predict_image(image, monument_model, class_indices)
            st.write(f"Predicted Monument: {label}, Confidence: {confidence * 100:.2f}%")
