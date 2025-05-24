import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page config
st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="wide")

# Load model (cached)
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.info("Loading model (first time)...")
        file_id = "1li65r7GXuNQ1fNl9GvtB20FYhyVzwBDt"  # Replace with your ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return YOLO(model_path)

# Main app
def main():
    st.title("🤟 Indian Sign Language Recognition")
    
    # Load model
    model = load_model()
    
    # Tabs for different modes
    tab1, tab2 = st.tabs(["📹 Live Camera", "📤 Upload Image"])
    
    # Live Camera Tab
    with tab1:
        st.subheader("Live Camera Feed")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            start_btn = st.button("Start Camera", type="primary")
            stop_btn = st.button("Stop Camera")
            camera_placeholder = st.empty()
        
        with col2:
            prediction_placeholder = st.empty()
        
        # Initialize session state
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
        
        if start_btn:
            st.session_state.camera_on = True
        if stop_btn:
            st.session_state.camera_on = False
        
        # Camera logic
        if st.session_state.camera_on:
            run_camera(camera_placeholder, prediction_placeholder, model)
    
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
                    # Get prediction
                    names = results[0].names
                    probs = results[0].probs
                    
                    if probs is not None:
                        top_class = probs.top1
                        confidence = probs.top1conf.item() * 100
                        predicted_letter = names[top_class]
                        
                        # Display result
                        st.markdown(f"""
                        ### Prediction Result
                        **Letter: {predicted_letter}**  
                        **Confidence: {confidence:.1f}%**
                        """)
                        
                        # Progress bar for confidence
                        st.progress(confidence/100)

def run_camera(camera_placeholder, prediction_placeholder, model):
    """Handle camera feed"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access camera")
        return
    
    try:
        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert for prediction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict
            results = model(rgb_frame)
            prediction_text = "No detection"
            confidence = 0
            
            if results and len(results) > 0:
                names = results[0].names
                probs = results[0].probs
                
                if probs is not None:
                    top_class = probs.top1
                    confidence = probs.top1conf.item() * 100
                    predicted_letter = names[top_class]
                    prediction_text = f"{predicted_letter} ({confidence:.1f}%)"
            
            # Add text to frame
            cv2.putText(frame, prediction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB")
            
            # Update prediction display
            with prediction_placeholder.container():
                if confidence > 0:
                    st.markdown(f"""
                    ### Current Prediction
                    **{predicted_letter}**  
                    Confidence: {confidence:.1f}%
                    """)
                    st.progress(confidence/100)
                else:
                    st.markdown("### No detection")
                    
    finally:
        cap.release()

if __name__ == "__main__":
    main()
