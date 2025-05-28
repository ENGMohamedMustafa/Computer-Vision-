# Student Focus Monitoring System


A real-time computer vision application that monitors student attention and focus during lectures or study sessions using YOLOv9 object detection and behavioral analysis.

## Features

- Real-time student focus detection with multiple state classification:
  - Focused (looking at material/screen)
  - Phone usage detection
  - Talking detection
  - Sleeping/drowsy detection
  - Looking away detection
  - Writing (note-taking) detection
- Gaze direction estimation
- Comprehensive analytics dashboard:
  - Real-time focus statistics
  - State distribution visualization
  - Focus timeline tracking
  - Individual student metrics
- Support for both live camera feeds and recorded videos
- Configurable detection thresholds and monitoring options

## Technologies Used

- Python 3.x
- Streamlit (Web UI)
- OpenCV (Computer Vision)
- PyTorch (Deep Learning)
- YOLOv9 (Object Detection)
- Plotly (Data Visualization)
- Haar Cascades (Face/Eye Detection)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/EngMohamedMustafa/Computer-Vision-
/Student Focus Monitoring System Real_time
   cd student-focus-monitor
