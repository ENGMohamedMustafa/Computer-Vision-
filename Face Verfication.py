import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile

st.title("Face Verification App")

img1_file = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png", "webp"])
img2_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png", "webp"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="First Image", use_column_width=True)
    with col2:
        st.image(img2, caption="Second Image", use_column_width=True)

    if st.button("Verify"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img1_temp:
            img1.save(img1_temp.name)
            img1_path = img1_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img2_temp:
            img2.save(img2_temp.name)
            img2_path = img2_temp.name

        result = DeepFace.verify(img1_path, img2_path)
        st.write("Verification Result:", result)
