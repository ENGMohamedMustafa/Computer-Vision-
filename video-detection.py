import streamlit as st
import cv2
import torch
import numpy as np
import time
import tempfile
import os
from ultralytics import YOLO
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="YOLOv9-l Video Detection",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("🎥 Video Object Detection with YOLOv9-l")
st.markdown("""
Upload a video file to detect and analyze objects using YOLOv9-l, 
a state-of-the-art object detection model.
""")

# Sidebar configuration
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence score for detection"
)

detection_mode = st.sidebar.radio(
    "Detection Mode",
    ["Standard Detection", "Object Tracking"],
    help="Standard detection processes each frame independently. Object tracking follows objects across frames."
)

display_mode = st.sidebar.radio(
    "Display Mode", 
    ["Show Video", "Show Stats Only"],
    help="Choose to display the processed video or just the statistics"
)

show_frame_rate = st.sidebar.checkbox("Show Processing Stats", value=True)


@st.cache_resource
def load_model():
    """Load YOLOv9-l model and cache it"""
    # Display a spinner during model loading
    with st.spinner("Loading YOLOv9-l model... This might take a moment."):
        model = YOLO('yolov9l.pt')  # Will download if not available
        return model


def process_video(video_path, conf_threshold, tracking=False):
    """Process a video with YOLOv9-l for object detection and analysis"""
    # Get the model
    model = load_model()
    
    # Create a temporary file for the output
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output_file.name
    temp_output_file.close()
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        return None, None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics for analysis
    all_detections = []
    frame_times = []
    frame_count = 0
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # If tracking mode is selected
    if tracking:
        cap.release()
        
        # Use tempfile for tracking output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_track_file:
            track_output = temp_track_file.name
        
        # Run tracking
        with st.spinner("Processing video with object tracking..."):
            results = model.track(
                source=video_path,
                conf=conf_threshold,
                save=True,
                project=os.path.dirname(track_output),
                name=os.path.basename(track_output).split('.')[0],
                tracker="bytetrack.yaml",
                verbose=False
            )
        
        # Get actual output file (ultralytics saves to a runs directory)
        runs_dir = os.path.join(os.path.dirname(track_output), "runs")
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                if file.endswith(".mp4"):
                    track_output = os.path.join(root, file)
                    break
        
        # Collect statistics from tracking results
        stats = {
            'total_frames': len(results),
            'tracking_enabled': True
        }
        
        # Copy the tracking output to our standard output path
        os.system(f"cp {track_output} {output_path}")
        
        return output_path, stats
    
    # Standard detection processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
        
        # Measure detection time
        start_time = time.time()
        
        # Run YOLOv9 detection on the frame
        results = model(frame, verbose=False)[0]
        
        # Record processing time
        end_time = time.time()
        frame_times.append(end_time - start_time)
        
        # Process the results
        frame_detections = []
        annotated_frame = frame.copy()
        
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            
            if confidence < conf_threshold:
                continue
                
            class_id = int(class_id)
            class_name = results.names[class_id]
            
            frame_detections.append({
                'class': class_name,
                'confidence': confidence,
                'box': [x1, y1, x2, y2],
                'frame': frame_count
            })
            
            # Draw bounding box and label on the frame
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        all_detections.extend(frame_detections)
        
        # Add frame number and processing time if enabled
        if show_frame_rate:
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Time: {(end_time - start_time)*1000:.1f}ms", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if len(frame_times) > 0:
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                cv2.putText(annotated_frame, f"Avg FPS: {avg_fps:.1f}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Write the frame to output video
        out.write(annotated_frame)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Release resources
    cap.release()
    out.release()
    
    # Compute statistics for analysis
    stats = {
        'total_frames': frame_count,
        'avg_processing_time': np.mean(frame_times) if frame_times else 0,
        'fps': 1.0 / np.mean(frame_times) if frame_times and np.mean(frame_times) > 0 else 0,
        'total_detections': len(all_detections),
        'unique_classes': Counter([d['class'] for d in all_detections]),
        'detections_per_frame': Counter([d['frame'] for d in all_detections]),
        'avg_detections_per_frame': len(all_detections) / frame_count if frame_count > 0 else 0,
        'tracking_enabled': False,
        'all_detections': all_detections
    }
    
    return output_path, stats


def display_stats(stats):
    """Display detection statistics with charts"""
    if not stats:
        return
    
    st.header("Detection Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Frames Processed", stats['total_frames'])
        st.metric("Total Detections", stats['total_detections'])
        
        if not stats.get('tracking_enabled', False):
            st.metric("Average Processing Time", f"{stats['avg_processing_time']*1000:.1f} ms")
            st.metric("Average FPS", f"{stats['fps']:.1f}")
    
    with col2:
        if not stats.get('tracking_enabled', False):
            st.metric("Average Detections per Frame", f"{stats['avg_detections_per_frame']:.2f}")
    
    # Object class distribution chart
    if 'unique_classes' in stats and stats['unique_classes']:
        st.subheader("Object Class Distribution")
        class_data = pd.DataFrame({
            'Class': list(stats['unique_classes'].keys()),
            'Count': list(stats['unique_classes'].values())
        }).sort_values('Count', ascending=False)
        
        fig = px.bar(class_data, x='Class', y='Count', color='Count',
                    color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detections over time chart (if not tracking)
    if 'detections_per_frame' in stats and stats['detections_per_frame'] and not stats.get('tracking_enabled', False):
        st.subheader("Detections Over Time")
        
        # Create data for the chart
        frames = sorted(stats['detections_per_frame'].keys())
        counts = [stats['detections_per_frame'][frame] for frame in frames]
        
        # Create a more efficient representation by sampling if too many frames
        if len(frames) > 100:
            sample_rate = len(frames) // 100
            frames = frames[::sample_rate]
            counts = counts[::sample_rate]
        
        time_data = pd.DataFrame({
            'Frame': frames,
            'Detections': counts
        })
        
        fig = px.line(time_data, x='Frame', y='Detections')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Object confidence distribution
    if 'all_detections' in stats and stats['all_detections'] and not stats.get('tracking_enabled', False):
        confidences = [d['confidence'] for d in stats['all_detections']]
        
        st.subheader("Confidence Score Distribution")
        fig = px.histogram(confidences, nbins=20, 
                          labels={'value': 'Confidence Score', 'count': 'Number of Detections'},
                          range_x=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


# Main file uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()
    
    # Process the video
    st.header("Processing Video")
    tracking_mode = detection_mode == "Object Tracking"
    
    with st.spinner("Processing video..."):
        output_path, stats = process_video(
            video_path=video_path,
            conf_threshold=confidence_threshold,
            tracking=tracking_mode
        )
    
    if output_path and os.path.exists(output_path):
        # Display the processed video
        if display_mode == "Show Video":
            st.header("Processed Video")
            st.video(output_path)
        
        # Display detection statistics
        display_stats(stats)
        
        # Download button for the processed video
        with open(output_path, 'rb') as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
        
        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(output_path)
    else:
        st.error("Video processing failed. Please try a different video or adjust the settings.")
else:
    # Display sample
    st.info("Upload a video to begin detection. YOLOv9-l can detect a wide range of objects including people, vehicles, animals, and everyday items.")
    
    # Show a placeholder image or example
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov9/yolov9_architecture.png", 
             caption="YOLOv9 Architecture Overview", use_column_width=True)

# Instructions and about section
st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a video file (.mp4, .avi, .mov, or .mkv)
2. Adjust detection settings in the sidebar
3. Wait for processing to complete
4. View results and download processed video
""")

st.sidebar.header("About")
st.sidebar.markdown("""
This app uses YOLOv9-l for real-time object detection. 
YOLOv9 is a state-of-the-art detection model capable of 
identifying 80 different object classes with high accuracy.

Built with Streamlit, OpenCV, and Ultralytics YOLOv9.
""")

# Model information
st.sidebar.markdown("---")
st.sidebar.header("Model Information")
st.sidebar.markdown("""
- **Model**: YOLOv9-l (large)
- **Classes**: 80 (COCO dataset)
- **Framework**: Ultralytics
""")