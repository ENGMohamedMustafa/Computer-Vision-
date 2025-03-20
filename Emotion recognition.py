from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import streamlit as st


st.title("Human Emotion Reco")


def analysis(img):
    results = DeepFace.analyze(img, actions=["emotion"])
    return results[0]["emotion"]


upload = st.file_uploader("Choose file", type=["png", "jpg", "jpeg", "webp"])

if upload is not None:
    img = Image.open(upload)
    img_np = np.array(img)

    st.image(img_np, channels="RGB")

    results = analysis(img_np)
    emotion = max(results, key=results.get)

    st.write("Detected Emotion", emotion)
