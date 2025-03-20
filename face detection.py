import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


# using opencv face detection
def detect_opencv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img)
    # some points 4pt rectangle on face
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return img


def detect_mediapipe(img):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_detection.FaceDetection() as face_detection:
        results = face_detection.process(img)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)
    return img


st.title("face detection app")
upload = st.file_uploader("Please upload an image", type=["png", "jpg", "jpeg"])

if upload is not None:
    img = Image.open(upload)
    img_array = np.array(img)
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    options = st.selectbox("Choose between methods", ("None", "opencv", "mediapipe"))

    if options == "None":
        detection = img
    elif options == "opencv":
        detection = detect_opencv(img)
    elif options == "mediapipe":
        detection = detect_mediapipe(img)

    st.image(detection, channels="BGR")
