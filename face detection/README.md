# Face Detection App

A user-friendly web application built with **Streamlit** that enables face detection in images using either **OpenCV's Haar Cascades** or **Google MediaPipe**.

---

## 🎯 Features

- Upload an image in `.png`, `.jpg`, or `.jpeg` format.
- Choose between two face detection methods:
  - **OpenCV** (Haar Cascade)
  - **MediaPipe** (Google's high-accuracy face detection)
- Display the image with detected faces highlighted.

---

## 📸 Demo

1. Upload an image with visible human faces.
2. Select your desired detection method:
   - `None`: Show original image.
   - `opencv`: Detects faces using Haar Cascade.
   - `mediapipe`: Detects faces using Google's MediaPipe Face Detection.
3. Output is rendered in real-time with bounding boxes over faces.

---

## 🧪 How It Works

### OpenCV Detection
- Converts the image to grayscale.
- Uses Haar Cascade classifier to detect face coordinates.
- Draws blue rectangles around faces.

### MediaPipe Detection
- Uses the `mediapipe.solutions.face_detection` pipeline.
- Detects and annotates faces with confidence score and key facial landmarks.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
