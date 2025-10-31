import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Gait Analyzer")
st.title("Video-Based Gait Analyzer (Knee Angle)")
st.markdown("Upload a video of a person walking (side view recommended) to analyze knee joint angles.")

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Helper Functions ---

def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points (in degrees)."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def plot_signal(timestamps, angles, title, yaxis_title="Angle (Degrees)"):
    """Creates an interactive signal plot using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=angles, mode='lines+markers', name='Angle'))
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title=yaxis_title,
        template="plotly_dark"
    )
    return fig

# --- Main Processing Function ---

@st.cache_data
def process_video(video_path):
    """
    Processes the video, extracts landmarks, calculates knee angles, and creates an annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file at: {video_path}")
        return None, None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Change format to WebM with VP9 codec, which is browser-native
    out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'VP90'), fps, (width, height))

    # Lists to store data for plotting
    left_knee_angles = []
    right_knee_angles = []
    timestamps = []
    
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                left_knee_angles.append(left_angle)
                right_knee_angles.append(right_angle)
                timestamps.append(frame_count / fps)
                
                cv2.putText(image, f"Left: {int(left_angle)}", (int(left_knee[0]*width)+10, int(left_knee[1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Right: {int(right_angle)}", (int(right_knee[0]*width)+10, int(right_knee[1]*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            except:
                left_knee_angles.append(None)
                right_knee_angles.append(None)
                timestamps.append(frame_count / fps)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            out.write(image)
            frame_count += 1
            
    cap.release()
    out.release()
    
    return out_video_path, timestamps, left_knee_angles, right_knee_angles

# --- Streamlit UI ---

# --- START OF UI CHANGE ---
EXAMPLE_VIDEO_PATH = "videos/examplevideo.mp4"

st.sidebar.header("Video Input")
use_example = st.sidebar.checkbox("Use Example Video", value=True)

uploaded_file = None
if not use_example:
    uploaded_file = st.sidebar.file_uploader("Upload video file (mp4, avi, mov)", type=["mp4", "avi", "mov"])

video_path = None
temp_video_to_delete = None

if use_example:
    if os.path.exists(EXAMPLE_VIDEO_PATH):
        video_path = EXAMPLE_VIDEO_PATH
    else:
        st.error(f"Example video not found at: {EXAMPLE_VIDEO_PATH}")
        st.stop()
elif uploaded_file is not None:
    original_suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name
        temp_video_to_delete = tfile.name
# --- END OF UI CHANGE ---

if video_path is not None:
    annotated_video_path = None # Define outside try block
    try:
        with st.spinner("Analyzing video... This may take a while for long videos..."):
            annotated_video_path, times, left_angles, right_angles = process_video(video_path)
            
        if annotated_video_path:
            st.success("Analysis complete!")
            
            # --- START OF LAYOUT CHANGE (Smaller Video) ---
            
            # Row 1: Analyzed Video (Centered, 50% width)
            vid_col1, vid_col2, vid_col3 = st.columns([1, 2, 1])
            with vid_col2:
                st.header("Analyzed Video")
                video_file = open(annotated_video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes, format='video/webm')
            
            # --- END OF LAYOUT CHANGE ---
            
            st.divider()

            # Row 2: Graphs
            st.header("Gait Signal (Knee Angle)")
            
            clean_times = [t for i, t in enumerate(times) if left_angles[i] is not None]
            clean_left = [a for a in left_angles if a is not None]
            clean_right = [a for a in right_angles if a is not None]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if clean_left:
                    fig_left = plot_signal(clean_times, clean_left, "Left Knee Angle Over Time")
                    st.plotly_chart(fig_left, use_container_width=True)
                else:
                    st.info("No left leg detected.")
            
            with col2:
                if clean_right:
                    fig_right = plot_signal(clean_times, clean_right, "Right Knee Angle Over Time")
                    st.plotly_chart(fig_right, use_container_width=True)
                else:
                    st.info("No right leg detected.")

    finally:
        # Clean up temp files
        if temp_video_to_delete and os.path.exists(temp_video_to_delete):
            os.remove(temp_video_to_delete)
        if annotated_video_path and os.path.exists(annotated_video_path):
            os.remove(annotated_video_path)
