import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gdown
import os
import requests

# Page config
st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="wide")

# Load model (cached)
@st.cache_resource
def load_model():
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model (first time only)... Please wait."):
            try:
                file_id = "1li65r7GXuNQ1fNl9GvtB20FYhyVzwBDt"
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.stop()
    
    return YOLO(model_path)

# Main app
def main():
    st.title("🤟 Indian Sign Language Recognition")

    # Load model
    model = load_model()

    # Tabs for different modes
    tab1, tab2 = st.tabs(["📷 Live Camera", "📤 Upload Image"])

    # Live Camera Tab
    with tab1:
        st.subheader("Capture Image from Webcam")
        frame = st.camera_input("Take a picture")

        if frame is not None:
            image = Image.open(frame)
            st.image(image, caption="Captured Image", use_column_width=True)

            # Run prediction
            results = model(image)
            if results and len(results) > 0:
                names = results[0].names
                probs = results[0].probs

                if probs is not None:
                    top_class = probs.top1
                    confidence = probs.top1conf.item() * 100
                    predicted_letter = names[top_class]

                    st.markdown(f"""
                    ### Prediction Result
                    **Letter: {predicted_letter}**  
                    **Confidence: {confidence:.1f}%**
                    """)
                    st.progress(confidence / 100)

    # Upload Image Tab
    with tab2:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            col1, col2 = st.columns(2)

            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with col2:
                # Predict
                results = model(image)
                if results and len(results) > 0:
                    names = results[0].names
                    probs = results[0].probs

                    if probs is not None:
                        top_class = probs.top1
                        confidence = probs.top1conf.item() * 100
                        predicted_letter = names[top_class]

                        st.markdown(f"""
                        ### Prediction Result
                        **Letter: {predicted_letter}**  
                        **Confidence: {confidence:.1f}%**
                        """)
                        st.progress(confidence / 100)

if __name__ == "__main__":
    main()
