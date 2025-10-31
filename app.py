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
        st.error("Error opening video file.")
        return None, None, None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- START OF FIX (V3) ---
    # Change format to WebM with VP9 codec, which is browser-native
    out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'VP90'), fps, (width, height))
    # --- END OF FIX (V3) ---

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
                
            # Recolor image to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recolor back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for Left side
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Get coordinates for Right side
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Calculate angles
                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                # Store data
                left_knee_angles.append(left_angle)
                right_knee_angles.append(right_angle)
                timestamps.append(frame_count / fps)
                
                # Visualize angle on the video
                cv2.putText(image, f"Left: {int(left_angle)}", 
                               (int(left_knee[0]*width)+10, int(left_knee[1]*height)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Right: {int(right_angle)}", 
                               (int(right_knee[0]*width)+10, int(right_knee[1]*height)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            except:
                # Append None if landmarks were not detected
                left_knee_angles.append(None)
                right_knee_angles.append(None)
                timestamps.append(frame_count / fps)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            out.write(image)
            frame_count += 1
            
    cap.release()
    out.release()
    
    return out_video_path, timestamps, left_knee_angles, right_knee_angles

# --- Streamlit UI ---

uploaded_file = st.file_uploader("Upload video file (mp4, avi, mov)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Use the original suffix for the input file
    original_suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name

    with st.spinner("Analyzing video... This may take a while for long videos..."):
        annotated_video_path, times, left_angles, right_angles = process_video(video_path)
        
        if annotated_video_path:
            st.success("Analysis complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Analyzed Video")
                video_file = open(annotated_video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes, format='video/webm') # Tell streamlit it's a webm file
            
            with col2:
                st.header("Gait Signal (Knee Angle)")
                # Clean data for plotting (remove None values)
                clean_times = [t for i, t in enumerate(times) if left_angles[i] is not None]
                clean_left = [a for a in left_angles if a is not None]
                clean_right = [a for a in right_angles if a is not None]
                
                if clean_left:
                    fig_left = plot_signal(clean_times, clean_left, "Left Knee Angle Over Time")
                    st.plotly_chart(fig_left, use_container_width=True)
                if clean_right:
                    fig_right = plot_signal(clean_times, clean_right, "Right Knee Angle Over Time")
                    st.plotly_chart(fig_right, use_container_width=True)

    # Clean up temp files
    if 'video_path' in locals() and os.path.exists(video_path):
        os.remove(video_path)
    if 'annotated_video_path' in locals() and os.path.exists(annotated_video_path):
        os.remove(annotated_video_path)
