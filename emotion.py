# app.py
import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

# ---------------------------
# Simple login system
# ---------------------------
# For demo purposes, hard-coded users
users = {"teacher": "password123"}

st.set_page_config(page_title="Classroom Emotion Dashboard", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid credentials")
else:
    st.title("Classroom Emotion Dashboard")

    # ---------------------------
    # Session controls
    # ---------------------------
    if "session_running" not in st.session_state:
        st.session_state.session_running = False

    start_session = st.button("Start Session")
    stop_session = st.button("Stop Session")

    # Placeholders for session info
    session_info_placeholder = st.empty()
    video_placeholder = st.empty()

    if start_session:
        st.session_state.session_running = True
        st.session_state.emotion_history = []
        st.session_state.positions = []

    # ---------------------------
    # Capture loop
    # ---------------------------
    if st.session_state.session_running:
        st.info("Session running... Press Stop Session to finish.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        start_time = time.time()

        while st.session_state.session_running:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y+h, x:x+w]
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                except:
                    emotion = "Unknown"

                st.session_state.emotion_history.append(emotion)
                st.session_state.positions.append((x + w//2, y + h//2))

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Show video frame
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Stop session if button clicked
            if stop_session:
                st.session_state.session_running = False
                break

        cap.release()

    # ---------------------------
    # Generate dashboard after session
    # ---------------------------
    if not st.session_state.session_running and "emotion_history" in st.session_state and st.session_state.emotion_history:
        st.subheader("Session Summary")

        # Emotion counts
        emotion_counts = Counter(st.session_state.emotion_history)
        emotion_df = pd.DataFrame.from_dict(emotion_counts, orient='index', columns=['Count'])
        st.dataframe(emotion_df)

        # Emotion bar chart
        st.bar_chart(emotion_df)

        # Heatmap of face positions
        if st.session_state.positions:
            x_coords, y_coords = zip(*st.session_state.positions)
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=[64, 36], range=[[0, 1280], [0, 720]])
            heatmap = cv2.resize(heatmap.T, (1280, 720))  # transpose to match image coords

            plt.figure(figsize=(12,6))
            plt.imshow(heatmap, cmap='jet', alpha=0.7)
            plt.title("Heatmap of Face Positions")
            plt.axis('off')
            st.pyplot(plt)
