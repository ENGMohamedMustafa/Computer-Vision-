# 🎯 Video Object Detection with YOLOv9 and Streamlit

This project is a Streamlit-based web app that performs object detection and tracking on uploaded videos using YOLOv9. It provides a smooth UI, detection/tracking visualization, and advanced analytics about detected objects over time.

---

## 🚀 Features

- 📦 Supports **YOLOv9** for object detection
- 📹 Accepts user-uploaded videos
- 🎛️ Sidebar controls:
  - Confidence Threshold
  - Detection Mode: `Detection` or `Tracking`
- 📊 Displays analytics:
  - Object count over time
  - Confidence distribution
  - Class distribution
- 📥 Processed video download
- ⌛ Progress indicator and feedback

---

## 📦 Installation

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install YOLOv9 if not available
pip install git+https://github.com/ultralytics/ultralytics
