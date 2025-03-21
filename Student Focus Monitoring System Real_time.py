import streamlit as st
import cv2
import torch
import numpy as np
import time
import tempfile
import os
from ultralytics import YOLO
from collections import Counter, defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading
import queue

# Set page config
st.set_page_config(
    page_title="Student Focus Monitor",
    page_icon="👩‍🎓",
    layout="wide"
)

# Title and description
st.title("👩‍🎓 Real-Time Student Focus Monitoring System")
st.markdown("""
This application monitors student focus and attention during lectures or study sessions
using computer vision. It detects distraction indicators and provides real-time analytics.
""")

# Sidebar configuration
st.sidebar.header("Monitoring Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Minimum confidence score for detection"
)

distraction_threshold = st.sidebar.slider(
    "Distraction Threshold (seconds)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Number of seconds of detected distraction before alerting"
)

alert_mode = st.sidebar.checkbox(
    "Enable Alerts",
    value=True,
    help="Enable visual and audio alerts when distraction is detected"
)

show_gaze_direction = st.sidebar.checkbox(
    "Show Gaze Direction",
    value=True,
    help="Attempt to estimate and display gaze direction"
)

class_labels = {
    'focused': 'Student is looking at material and appears engaged',
    'phone': 'Student is looking at or using a phone',
    'talking': 'Student appears to be talking to others',
    'sleeping': 'Student appears to be sleeping or very drowsy',
    'looking_away': 'Student is looking away from the learning material',
    'writing': 'Student is writing notes (considered focused)'
}

monitor_classes = {
    'focused': st.sidebar.checkbox('Monitor focused state', value=True),
    'phone': st.sidebar.checkbox('Monitor phone usage', value=True),
    'talking': st.sidebar.checkbox('Monitor talking', value=True),
    'sleeping': st.sidebar.checkbox('Monitor sleeping/drowsy', value=True),
    'looking_away': st.sidebar.checkbox('Monitor looking away', value=True),
    'writing': st.sidebar.checkbox('Monitor writing', value=False)
}

@st.cache_resource
def load_model():
    """Load YOLOv9-l model and cache it"""
    with st.spinner("Loading YOLOv9-l model... This might take a moment."):
        # Load the main YOLO model for person detection
        model = YOLO('yolov9l.pt')  # Will download if not available
        
        # For a real application, we would fine-tune a model for specific attention states
        # Here we're simulating it by using the person detection and adding attention logic
        return model


class FocusDetector:
    def __init__(self, conf_threshold=0.4):
        self.model = load_model()
        self.conf_threshold = conf_threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # For student tracking
        self.student_states = defaultdict(lambda: {
            'state': 'unknown',
            'state_duration': 0,
            'history': [],
            'last_update': time.time(),
            'position': None,
            'face_position': None,
            'gaze_direction': None,
            'distraction_events': 0,
            'total_focused_time': 0,
            'total_distracted_time': 0
        })
        
        # List of phones and devices to detect
        self.device_classes = ['cell phone', 'laptop', 'book']
        
        # Initialize statistics
        self.class_counts = Counter()
        self.focus_timeline = []
        self.last_stats_update = time.time()
        
    def detect_faces_and_eyes(self, frame):
        """Detect faces and eyes in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_gray)
            
            results.append({
                'face': (x, y, w, h),
                'eyes': eyes
            })
            
        return results
    
    def estimate_gaze_direction(self, face_data):
        """Estimate gaze direction based on eye positions"""
        if not face_data or 'eyes' not in face_data or len(face_data['eyes']) < 2:
            return None
        
        # This is a simplified simulation of gaze detection
        # Real gaze detection would require more sophisticated algorithms
        face_x, face_y, face_w, face_h = face_data['face']
        
        # Calculate face center
        face_center_x = face_x + face_w // 2
        face_center_y = face_y + face_h // 2
        
        # Find average eye position
        eye_center_x = 0
        eye_center_y = 0
        for (ex, ey, ew, eh) in face_data['eyes'][:2]:  # Take up to 2 eyes
            eye_center_x += face_x + ex + ew // 2
            eye_center_y += face_y + ey + eh // 2
        
        eye_center_x /= min(len(face_data['eyes']), 2)
        eye_center_y /= min(len(face_data['eyes']), 2)
        
        # Calculate direction vector
        dx = eye_center_x - face_center_x
        dy = eye_center_y - face_center_y
        
        # Normalize
        magnitude = (dx**2 + dy**2)**0.5
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        return (dx, dy)
    
    def detect_focus(self, frame, timestamp):
        """Detect student focus in a frame"""
        # Run YOLO detection for people and devices
        yolo_results = self.model(frame, verbose=False)[0]
        
        # Detect faces for more detailed analysis
        face_results = self.detect_faces_and_eyes(frame)
        
        # Process students detected in the frame
        detections = []
        people_boxes = []
        device_boxes = []
        
        # Extract people and device detections
        for detection in yolo_results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            
            if confidence < self.conf_threshold:
                continue
                
            class_id = int(class_id)
            class_name = yolo_results.names[class_id]
            
            if class_name == 'person':
                people_boxes.append((x1, y1, x2, y2))
            elif class_name in self.device_classes:
                device_boxes.append((x1, y1, x2, y2, class_name))
        
        # Process each person detected
        for i, (x1, y1, x2, y2) in enumerate(people_boxes):
            student_id = f"student_{i}"
            
            # Extract student region for further analysis
            student_region = frame[int(y1):int(y2), int(x1):int(x2)]
            if student_region.size == 0:
                continue
                
            # Check for face in this region
            student_face = None
            student_gaze = None
            
            for face_data in face_results:
                face_x, face_y, face_w, face_h = face_data['face']
                # Check if face is within this person's bounding box
                if (face_x > x1 and face_y > y1 and 
                    face_x + face_w < x2 and face_y + face_h < y2):
                    student_face = face_data
                    student_gaze = self.estimate_gaze_direction(face_data)
                    break
            
            # Determine if student has a device
            has_device = False
            device_type = None
            for dev_x1, dev_y1, dev_x2, dev_y2, dev_type in device_boxes:
                # Check if device overlaps with student
                if (dev_x1 < x2 and dev_x2 > x1 and 
                    dev_y1 < y2 and dev_y2 > y1):
                    has_device = True
                    device_type = dev_type
                    break
            
            # Determine focus state (in a real app, this would be more sophisticated)
            curr_state = 'unknown'
            
            # Simple heuristics for demonstration
            if student_face is None:
                curr_state = 'looking_away'
            elif device_type == 'cell phone':
                curr_state = 'phone'
            elif device_type == 'book' or device_type == 'laptop':
                curr_state = 'focused'
            elif student_gaze:
                dx, dy = student_gaze
                # If looking down significantly
                if dy > 0.2:
                    curr_state = 'writing'
                # If looking significantly to the side
                elif abs(dx) > 0.3:
                    curr_state = 'looking_away'
                else:
                    curr_state = 'focused'
            
            # Update student state
            if student_id in self.student_states:
                # Calculate time since last update
                now = time.time()
                time_diff = now - self.student_states[student_id]['last_update']
                
                # If state is the same, increment duration
                if self.student_states[student_id]['state'] == curr_state:
                    self.student_states[student_id]['state_duration'] += time_diff
                else:
                    # Record the previous state in history
                    prev_state = self.student_states[student_id]['state']
                    prev_duration = self.student_states[student_id]['state_duration']
                    
                    if prev_state != 'unknown':
                        self.student_states[student_id]['history'].append({
                            'state': prev_state,
                            'duration': prev_duration,
                            'timestamp': timestamp
                        })
                    
                    # Reset duration for new state
                    self.student_states[student_id]['state'] = curr_state
                    self.student_states[student_id]['state_duration'] = 0
                
                # Update focused/distracted time counters
                if curr_state in ['focused', 'writing']:
                    self.student_states[student_id]['total_focused_time'] += time_diff
                elif curr_state in ['phone', 'looking_away', 'sleeping', 'talking']:
                    self.student_states[student_id]['total_distracted_time'] += time_diff
                    
                    # Count as distraction event if we just transitioned to distracted state
                    if self.student_states[student_id]['state'] != prev_state:
                        self.student_states[student_id]['distraction_events'] += 1
                
                self.student_states[student_id]['last_update'] = now
            else:
                # Initialize new student
                self.student_states[student_id] = {
                    'state': curr_state,
                    'state_duration': 0,
                    'history': [],
                    'last_update': time.time(),
                    'position': (x1, y1, x2, y2),
                    'face_position': student_face['face'] if student_face else None,
                    'gaze_direction': student_gaze,
                    'distraction_events': 0,
                    'total_focused_time': 0,
                    'total_distracted_time': 0
                }
            
            # Update position and gaze
            self.student_states[student_id]['position'] = (x1, y1, x2, y2)
            self.student_states[student_id]['face_position'] = student_face['face'] if student_face else None
            self.student_states[student_id]['gaze_direction'] = student_gaze
            
            # Add to class counts for statistics
            self.class_counts[curr_state] += 1
            
            # Create detection for visualization and stats
            detections.append({
                'id': student_id,
                'class': curr_state,
                'position': (x1, y1, x2, y2),
                'face_position': student_face['face'] if student_face else None,
                'gaze_direction': student_gaze,
                'confidence': confidence,
                'state_duration': self.student_states[student_id]['state_duration']
            })
        
        # Update focus timeline if enough time has passed
        now = time.time()
        if now - self.last_stats_update >= 1.0:  # Update every second
            states = {id: data['state'] for id, data in self.student_states.items()}
            self.focus_timeline.append({
                'timestamp': timestamp,
                'states': states,
                'focused_count': sum(1 for s in states.values() if s in ['focused', 'writing']),
                'distracted_count': sum(1 for s in states.values() if s in ['phone', 'looking_away', 'sleeping', 'talking'])
            })
            self.last_stats_update = now
            
        return detections
    
    def annotate_frame(self, frame, detections, show_gaze=True):
        """Annotate the frame with detection results"""
        annotated = frame.copy()
        
        # Define state colors
        state_colors = {
            'focused': (0, 255, 0),     # Green
            'phone': (0, 0, 255),       # Red
            'talking': (255, 165, 0),   # Orange
            'sleeping': (255, 0, 255),  # Purple
            'looking_away': (255, 0, 0),# Blue
            'writing': (0, 255, 255),   # Yellow
            'unknown': (128, 128, 128)  # Gray
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['position']
            state = det['class']
            
            # Draw bounding box
            color = state_colors.get(state, (128, 128, 128))
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add state label
            label = f"{state}: {det['state_duration']:.1f}s"
            cv2.putText(annotated, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw face box if detected
            if det['face_position']:
                fx, fy, fw, fh = det['face_position']
                cv2.rectangle(annotated, (int(fx), int(fy)), 
                             (int(fx + fw), int(fy + fh)), (255, 255, 255), 1)
            
            # Draw gaze direction if available and requested
            if show_gaze and det['gaze_direction'] and det['face_position']:
                fx, fy, fw, fh = det['face_position']
                start_x = int(fx + fw // 2)
                start_y = int(fy + fh // 2)
                
                dx, dy = det['gaze_direction']
                # Scale for visualization
                dx *= 50
                dy *= 50
                
                end_x = int(start_x + dx)
                end_y = int(start_y + dy)
                
                cv2.arrowedLine(annotated, (start_x, start_y), (end_x, end_y), 
                               (0, 255, 255), 2)
        
        return annotated
    
    def get_stats(self):
        """Get aggregated statistics"""
        stats = {
            'total_students': len(self.student_states),
            'currently_focused': sum(1 for s in self.student_states.values() 
                                   if s['state'] in ['focused', 'writing']),
            'currently_distracted': sum(1 for s in self.student_states.values() 
                                      if s['state'] in ['phone', 'looking_away', 'sleeping', 'talking']),
            'state_distribution': {k: v for k, v in self.class_counts.items() if v > 0},
            'student_states': {id: {
                'state': data['state'],
                'duration': data['state_duration'],
                'distraction_events': data['distraction_events'],
                'focused_time': data['total_focused_time'],
                'distracted_time': data['total_distracted_time'],
                'focus_ratio': data['total_focused_time'] / 
                              (data['total_focused_time'] + data['total_distracted_time'] + 0.001)
            } for id, data in self.student_states.items()},
            'timeline': self.focus_timeline
        }
        
        return stats


def process_video_feed(video_source, detector, result_queue, stop_event, conf_threshold, show_gaze=True):
    """Process video feed in a separate thread"""
    # Open video capture
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)
        
    if not cap.isOpened():
        result_queue.put(('error', f"Could not open video source: {video_source}"))
        return
        
    # Start time for timestamping
    start_time = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        
        if not ret:
            # For files, loop back to the beginning
            if not isinstance(video_source, int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # Calculate timestamp
        timestamp = time.time() - start_time
        
        # Process frame
        try:
            # Detect focus
            detections = detector.detect_focus(frame, timestamp)
            
            # Annotate frame
            annotated_frame = detector.annotate_frame(frame, detections, show_gaze)
            
            # Get current stats
            stats = detector.get_stats()
            
            # Put results in queue
            result_queue.put(('frame', annotated_frame))
            result_queue.put(('stats', stats))
            
        except Exception as e:
            result_queue.put(('error', f"Error processing frame: {str(e)}"))
            
        # Slightly reduce frame rate to not overwhelm the system
        time.sleep(0.01)
    
    cap.release()
    result_queue.put(('done', None))


def display_focus_stats(stats):
    """Display focus monitoring statistics"""
    if not stats:
        return
    
    st.subheader("Focus Statistics")
    
    # Focus metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Students", stats['total_students'])
    
    with col2:
        focused_pct = stats['currently_focused'] / stats['total_students'] * 100 if stats['total_students'] > 0 else 0
        st.metric("Currently Focused", f"{stats['currently_focused']} ({focused_pct:.1f}%)")
    
    with col3:
        distracted_pct = stats['currently_distracted'] / stats['total_students'] * 100 if stats['total_students'] > 0 else 0
        st.metric("Currently Distracted", f"{stats['currently_distracted']} ({distracted_pct:.1f}%)")
    
    # State distribution
    if 'state_distribution' in stats and stats['state_distribution']:
        st.subheader("Attention State Distribution")
        
        # Convert to DataFrame
        state_data = []
        for state, count in stats['state_distribution'].items():
            state_data.append({
                'State': state,
                'Count': count,
                'Description': class_labels.get(state, 'Unknown state')
            })
        
        state_df = pd.DataFrame(state_data)
        
        if not state_df.empty:
            # Create chart
            fig = px.bar(state_df, x='State', y='Count', 
                        color='State',
                        hover_data=['Description'])
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Student focus timeline if we have enough data
    if 'timeline' in stats and len(stats['timeline']) > 10:
        st.subheader("Focus Timeline")
        
        # Extract timeline data
        times = [entry['timestamp'] for entry in stats['timeline']]
        focused = [entry['focused_count'] for entry in stats['timeline']]
        distracted = [entry['distracted_count'] for entry in stats['timeline']]
        
        # Create DataFrame
        timeline_df = pd.DataFrame({
            'Time': times,
            'Focused Students': focused,
            'Distracted Students': distracted
        })
        
        # Plot
        fig = px.line(timeline_df, x='Time', y=['Focused Students', 'Distracted Students'])
        fig.update_layout(height=300, legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
    
    # Student details
    if 'student_states' in stats and stats['student_states']:
        st.subheader("Individual Student Analysis")
        
        # Create student dataframe
        student_data = []
        for id, data in stats['student_states'].items():
            student_data.append({
                'ID': id,
                'Current State': data['state'],
                'In Current State For': f"{data['duration']:.1f}s",
                'Focus Ratio': f"{data['focus_ratio']*100:.1f}%",
                'Distraction Events': data['distraction_events'],
                'Total Focused Time': f"{data['focused_time']:.1f}s",
                'Total Distracted Time': f"{data['distracted_time']:.1f}s"
            })
        
        student_df = pd.DataFrame(student_data)
        
        if not student_df.empty:
            # Format the dataframe
            st.dataframe(student_df, use_container_width=True)


def main():
    # Create tabs for different sections
    tab_live, tab_recorded, tab_about = st.tabs(["Live Monitoring", "Recorded Session", "About"])
    
    with tab_live:
        st.header("Live Focus Monitoring")
        
        # Camera selection
        camera_options = {
            "Default Camera": 0
        }
        
        selected_camera = st.selectbox(
            "Select Camera",
            list(camera_options.keys())
        )
        
        # Start/stop button for live monitoring
        start_monitoring = st.button("Start Monitoring", key="start_live")
        stop_monitoring = st.button("Stop Monitoring", key="stop_live")
        
        # Create placeholders for video and stats
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Video processing state
        if 'video_thread' not in st.session_state:
            st.session_state.video_thread = None
            st.session_state.stop_event = None
            st.session_state.result_queue = None
        
        # Start monitoring
        if start_monitoring and not st.session_state.video_thread:
            detector = FocusDetector(conf_threshold=confidence_threshold)
            
            # Create thread communication objects
            st.session_state.stop_event = threading.Event()
            st.session_state.result_queue = queue.Queue()
            
            # Start processing thread
            video_source = camera_options[selected_camera]
            st.session_state.video_thread = threading.Thread(
                target=process_video_feed,
                args=(video_source, detector, st.session_state.result_queue, 
                     st.session_state.stop_event, confidence_threshold, show_gaze_direction)
            )
            st.session_state.video_thread.daemon = True
            st.session_state.video_thread.start()
            
            st.success("Monitoring started!")
        
        # Stop monitoring
        if stop_monitoring and st.session_state.video_thread:
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            
            st.session_state.video_thread = None
            st.session_state.stop_event = None
            st.session_state.result_queue = None
            
            st.info("Monitoring stopped.")
        
        # Display results from queue
        if st.session_state.result_queue:
            latest_frame = None
            latest_stats = None
            
            # Get all available results
            while not st.session_state.result_queue.empty():
                try:
                    result_type, data = st.session_state.result_queue.get_nowait()
                    
                    if result_type == 'frame':
                        latest_frame = data
                    elif result_type == 'stats':
                        latest_stats = data
                    elif result_type == 'error':
                        st.error(data)
                except queue.Empty:
                    break
            
            # Display latest frame
            if latest_frame is not None:
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Display latest stats
            if latest_stats is not None:
                with stats_placeholder:
                    display_focus_stats(latest_stats)
    
    with tab_recorded:
        st.header("Analyze Recorded Session")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()
            
            # Start analysis button
            start_analysis = st.button("Start Analysis", key="start_analysis")
            stop_analysis = st.button("Stop Analysis", key="stop_analysis")
            
            # Create placeholders for video and stats
            analysis_video_placeholder = st.empty()
            analysis_stats_placeholder = st.empty()
            
            # Process uploaded video
            if start_analysis:
                detector = FocusDetector(conf_threshold=confidence_threshold)
                
                # Create thread communication objects
                st.session_state.stop_event = threading.Event()
                st.session_state.result_queue = queue.Queue()
                
                # Start processing thread
                st.session_state.video_thread = threading.Thread(
                    target=process_video_feed,
                    args=(video_path, detector, st.session_state.result_queue, 
                         st.session_state.stop_event, confidence_threshold, show_gaze_direction)
                )
                st.session_state.video_thread.daemon = True
                st.session_state.video_thread.start()
                
                st.success("Analysis started!")
            
            # Stop analysis
            if stop_analysis and st.session_state.video_thread:
                if st.session_state.stop_event:
                    st.session_state.stop_event.set()
                
                st.session_state.video_thread = None
                st.session_state.stop_event = None
                st.session_state.result_queue = None
                
                # Clean up temporary file
                os.unlink(video_path)
                
                st.info("Analysis stopped.")
            
            # Display results from queue
            if st.session_state.result_queue:
                latest_frame = None
                latest_stats = None
                
                # Get all available results
                while not st.session_state.result_queue.empty():
                    try:
                        result_type, data = st.session_state.result_queue.get_nowait()
                        
                        if result_type == 'frame':
                            latest_frame = data
                        elif result_type == 'stats':
                            latest_stats = data
                        elif result_type == 'error':
                            st.error(data)
                    except queue.Empty:
                        break
                
                # Display latest frame
                if latest_frame is not None:
                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
                    analysis_video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display latest stats
                if latest_stats is not None:
                    with analysis_stats_placeholder:
                        display_focus_stats(latest_stats)
    
    with tab_about:
        st.header("About This System")
        
        st.markdown("""
        ### How It Works
        
        This system uses computer vision and machine learning techniques to monitor student attention and focus in real-time. The system:
        
        1. **Detects Students**: Identifies each student in the video feed
        2. **Analyzes Behavior**: Monitors for indicators of attention and distraction
        3. **Tracks Focus**: Records focus metrics over time
        4. **Provides Analytics**: Displays real-time statistics and insights
        
        ### Focus Indicators
        
        The system monitors several behavioral indicators:
        
        - **Focused**: Student is looking at learning material, screen, or instructor
        - **Phone Usage**: Student is looking at or using a mobile device
        - **Talking**: Student appears to be engaging in conversation
        - **Sleeping/Drowsy**: Student appears to be sleeping or showing signs of drowsiness
        - **Looking Away**: Student's gaze is directed away from learning material
        - **Writing**: Student is taking notes (considered a focused state)
        
        ### Privacy Considerations
        
        This system is designed for educational settings with appropriate consent. The system:
        
        - Does not store facial identities
        - Only tracks behavioral patterns, not personal information
        - Should only be used with clear disclosure to students
        - Is intended as an educational tool, not for surveillance
        
        ### Limitations
        
        The current implementation has several limitations:
        
        - Simplified focus detection based on facial orientation and positioning
        - Limited gaze estimation accuracy
        - May not account for all legitimate reasons a student might look away
        - Works best in well-lit environments with clear camera views
        """)

# Info section
st.sidebar.title("Info")
st.sidebar.info(
    """
    This app is maintained by mohamed.
    """
)
