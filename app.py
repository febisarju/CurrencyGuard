import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# Load the trained model
model = load_model('currency_detection_model.h5')

# Define class indices
class_indices = {0: 'Fake', 1: 'Real'}

# Function to predict currency authenticity
def predict_currency(img):
    img = cv2.resize(img, (255, 255))  # Resize to match model input
    img = np.expand_dims(img, axis=0)  # Expand dimensions
    result = model.predict(img)  # Predict
    prediction = class_indices[int(result[0][0])]
    return prediction

# Streamlit UI
st.title("CurrencyGuard: Fake Currency Detection")
st.subheader("Using Convolutional Neural Networks")

# Upload image
uploaded_file = st.file_uploader("Upload an image of the currency", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict authenticity
    prediction = predict_currency(image)

    # Display prediction
    st.success(f"**Prediction: {prediction}**")

# 📸 Webcam Capture Feature
st.markdown("### Capture Image from Webcam")

# Button to Start Webcam
if st.button("Open Webcam"):
    cap = cv2.VideoCapture(0)  # Open webcam

    # Create a temporary file to store the image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam!")
            break

        # Show the live frame in Streamlit
        st.image(frame, channels="BGR", caption="Live Webcam Feed")

        # Capture Image on Button Click
        if st.button("Capture Image"):
            cv2.imwrite(temp_file.name, frame)
            cap.release()
            cv2.destroyAllWindows()
            st.success("Image Captured Successfully!")
            break

    # Load the captured image
    image = cv2.imread(temp_file.name)

    # Display captured image
    st.image(image, caption="Captured Image", use_column_width=True)

    # Predict authenticity
    prediction = predict_currency(image)
    st.success(f"**Prediction: {prediction}**")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by **Febi**")
