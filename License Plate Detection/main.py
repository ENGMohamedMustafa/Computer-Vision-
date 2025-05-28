# main.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from utils import detect_license_plates

# Set page config
st.set_page_config(
    page_title="License Plate Detection & Recognition",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 License Plate Detection and Recognition")
st.markdown("Upload an image to detect and recognize license plates using YOLO and EasyOCR.")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.info("This app uses YOLOv8 for vehicle detection and EasyOCR for text recognition.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image containing vehicles with license plates"
)

if uploaded_file is not None:
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Read the image file
    file_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Display original image
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, channels="BGR", caption="Original Image")
    
    # Process the image
    with st.spinner("🔍 Processing image... This may take a few moments."):
        # Save image temporarily since your utils function expects a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            temp_path = tmp_file.name
        
        try:
            # Call your detection function
            detected_plates, processed_image = detect_license_plates(temp_path)
            
            # Ensure we have a valid processed image
            if processed_image is None:
                processed_image = image.copy()
                st.warning("⚠️ Could not process the image properly. Showing original image.")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            detected_plates = []
            processed_image = image.copy()
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Display processed image
    with col2:
        st.subheader("🎯 Processed Image")
        if processed_image is not None:
            st.image(processed_image, channels="BGR", caption="Detected License Plates")
        else:
            st.error("❌ Could not process the image")
    
    # Display results
    st.subheader("📋 Detection Results")
    
    if detected_plates:
        st.success(f"✅ Found {len(detected_plates)} license plate(s)!")
        
        # Create columns for displaying plates
        if len(detected_plates) <= 3:
            cols = st.columns(len(detected_plates))
        else:
            cols = st.columns(3)
        
        for i, plate_text in enumerate(detected_plates):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.metric(
                    label=f"License Plate {i+1}",
                    value=plate_text,
                    help=f"Detected license plate text: {plate_text}"
                )
        
        # Display as a list as well
        st.markdown("### 📝 Detected License Plates:")
        for i, plate_text in enumerate(detected_plates, 1):
            st.write(f"**Plate {i}:** `{plate_text}`")




