import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from deepface import DeepFace

st.title('Human Emotions Recognation APP')
st.write('Upload an Image or video')

options = st.selectbox('Choose an option...',('Image', 'Video'))


def Analyze_Emotion(img_vid):
    try:
        analysis = DeepFace.analyze(img_vid,actions = ['emotion'], enforce_detection = False)
        return analysis[0]['emotion']

    except ValueError as e:
        st.write(f'Error: {e}')
        return None
    
if options == 'Image':
    upload = st.file_uploader('Upload an image, plz: ',type = ['jpg','png','jpeg'])
    if upload is not None:
        img = Image.open(upload)
        img_array = np.array(img)
        st.image(img_array, channels = 'RGB')

        emotion_scores = Analyze_Emotion(img_array)

        if emotion_scores:
            # emotion_scores = {'Happy': 0.7, 'sad': 0.3, 'angry': 0.2}
            detected_emotion = max(emotion_scores, key = emotion_scores.get)
            st.write(f'detected emotion: {detected_emotion}')
        else:
            st.write('No Face Detected in your img')
            
if options == 'Video':
    upload = st.file_uploader('Upload a video, plz: ',type = ['mp4','mov','avi'])
    if upload is not None:
        with tempfile.NamedTemporaryFile(delete = False) as temp_video:
            temp_video.write(upload.read())
            video_path = temp_video.name
        video = cv2.VideoCapture(video_path)

        frame_count = 0 
        farme_rate =50 # control how many frames 


        
        while video.isOpened():
            ret , farme = video.read()
            if not ret:
                break
            frame_count +=1

            if frame_count % farme_rate == 0:
                fram_rgb = cv2.cvtColor(farme,cv2.COLOR_BGR2RGB)
                emotion_scores = Analyze_Emotion(fram_rgb)
                if emotion_scores:
                    # emotion_scores = {'Happy': 0.7, 'sad': 0.3, 'angry': 0.2}
                    detected_emotion = max(emotion_scores, key = emotion_scores.get)
                    cv2.putText(farme, detected_emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
                else:
                    detected_emotion = 'No Face Detected'
                    cv2.putText(farme, detected_emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
                st.image(farme, channels = 'BGR')
        video.release()