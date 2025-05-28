# 🚗 License Plate Detection & Recognition

An advanced AI-powered web application for detecting and recognizing vehicle license plates using YOLOv8 and EasyOCR, all wrapped in a user-friendly Streamlit interface.

---

## 🔍 Overview

This project performs automatic detection of vehicles and their license plates from uploaded images. It uses:
## 🎯 Features

- ✅ Real-time detection of vehicles in images using YOLOv8
- 🔍 Localization of license plate regions within detected vehicles
- 🧠 OCR-based recognition using EasyOCR
- 💻 Interactive web interface built with Streamlit
- 🛠️ Fully modular and readable code with extensibility in mind
- 📷 Supports JPG, PNG, and WebP image formats
## 🖥️ Demo

### Original Image:
![Original Car](https://github.com/ENGMohamedMustafa/Computer-Vision-/blob/main/License%20Plate%20Detection/why-are-number-plates-yellow-and-white.jpg)
### Detected License Plate:
![Detect](https://github.com/ENGMohamedMustafa/Computer-Vision-/blob/main/License%20Plate%20Detection/Screenshot%202025-05-29%20000307.png)

![Detect](https://github.com/ENGMohamedMustafa/Computer-Vision-/blob/main/License%20Plate%20Detection/annotated_license_plate.jpg)

## 🧠 How It Works

1. **YOLOv8** detects all vehicles in the image (cars, trucks, buses, motorcycles).
2. For each detected vehicle:
   - The system looks for candidate license plate regions based on contours and aspect ratios.
   - Each region is preprocessed and passed through **EasyOCR** to extract the text.
   - Only valid license plate patterns are accepted.
3. The results are displayed visually using bounding boxes and labeled on the web app.

## 📁 Project Structure

License-Plate-Detection/
│
├── main.py # Streamlit app
├── utils.py # Core logic: detection, OCR, preprocessing
├── yolov8x.pt # YOLOv8 model weights (external)
├── requirements.txt # Dependencies
├── README.md # Documentation
├── example_images/ # Input/output samples

