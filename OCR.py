import streamlit as st
import cv2 
import pytesseract 
import numpy as np 
from PIL import Image 

pytesseract.pytesseract.tesseract_cmd =rr"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("Document Scanner Application")
upload = st.file_uploader("Please Upload you Image ", type =['jpg', 'png','jpeg', 'webp'])
def Extract_text(img):
  text =pytesseract.image_to_string(img)
  return text

if upload is not None:
  img = Image.open(upload)
  image_array = np.array(img)
  st.image(image_array, caption='Uploaded Image.....', use_column_width= True)
  with st.spinner("Extracting Text from ur Image..."):
    text = Extract_text(image_array)
    st.subheader("Text Scanned")
    text_list = text.splitlines()
    st.write(text_list)