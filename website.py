import streamlit as st
import cv2
import subprocess
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import time

# ----------------------------------------
# 🌟 PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(page_title="Team Gestura 🤟", layout="wide", page_icon="🤟")

st.markdown(
    """
    <style>
        .stApp { background-color: #f8f9fa; }
        h1, h2, h3 { color: #1e1e1e; }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("Black.png", width=150)
st.title("🤟 Team Gestura")
st.markdown("### AI-powered ASL Gesture Recognition using YOLOv8")

# ----------------------------------------
# ⚙️ MODEL LOADING
# ----------------------------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------------------
# 🧭 SIDEBAR NAVIGATION
# ----------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "🤟 Detection", "⭐ Rate Us"])

st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
voice_enabled = st.sidebar.toggle("🔊 Enable Voice", value=True)

# ----------------------------------------
# 🗣️ Helper Functions
# ----------------------------------------
def speak_phrase(phrase):
    if voice_enabled and phrase:
        subprocess.Popen(["say", phrase])

def calculate_fps(start_time, end_time):
    return round(1 / (end_time - start_time), 2) if (end_time - start_time) > 0 else 0

# ----------------------------------------
# 📊 DASHBOARD PAGE
# ----------------------------------------
if page == "📊 Dashboard":
    st.header("📊 Detection Dashboard")
    st.markdown("#### Overview of your ASL gesture detection performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("🕒 Average FPS", "—")
    col2.metric("✋ Total Gestures Detected", "—")
    col3.metric("🧠 Model Confidence", f"{confidence*100:.1f}%")

    st.info("""
    👋 **Welcome to the Team Gestura Dashboard!**

    Once you start detection, live metrics will appear here automatically.
    You can monitor FPS, total detections, and system performance.
    """)

    st.markdown("---")
    st.caption("📈 Real-time dashboard updates when detection is running.")

# ----------------------------------------
# 🤟 DETECTION PAGE
# ----------------------------------------
elif page == "🤟 Detection":
    st.header("🤟 ASL Gesture Detection")
    mode = st.radio("Select Input Mode", ["📷 Webcam", "🖼️ Image", "🎞️ Video"])

    # -------------------
    # 📷 WEBCAM MODE
    # -------------------
    if mode == "📷 Webcam":
        st.markdown("#### 🎥 Real-Time Detection (Webcam)")

        # Use browser camera if no direct webcam access
        if os.environ.get("STREAMLIT_RUNTIME") == "cloud" or not os.path.exists("/dev/video0"):
            st.warning("⚠️ Local webcam access unavailable. Using browser camera instead.")
            img_data = st.camera_input("📸 Capture a gesture")

            if img_data is not None:
                file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)

                start_time = time.time()
                results = model(frame, conf=confidence, verbose=False)
                end_time = time.time()

                annotated_frame = results[0].plot()
                detected_labels = [model.names[int(b.cls)] for b in results[0].boxes]
                fps = calculate_fps(start_time, end_time)
                detection_time = round(end_time - start_time, 3)

                st.image(annotated_frame, channels="BGR", caption=f"Processed in {detection_time}s (FPS: {fps})")

                if detected_labels:
                    phrase = ", ".join(set(detected_labels))
                    st.success(f"Detected: {phrase}")
                    speak_phrase(f"Detected {phrase}")
                else:
                    st.warning("No gestures detected.")
        else:
            # Local webcam (desktop app)
            run_webcam = st.toggle("🎥 Start Real-Time Detection")
            FRAME_WINDOW = st.image([])
            fps_placeholder = st.empty()
            time_placeholder = st.empty()

            if run_webcam:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Could not access webcam.")
                else:
                    st.success("✅ Webcam active! Detecting gestures in real time...")

                last_phrase = ""
                gesture_count = 0
                total_time = 0
                total_frames = 0

                while run_webcam:
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to capture frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    results = model(frame, conf=confidence, verbose=False)
                    annotated_frame = results[0].plot()

                    detected_labels = []
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        detected_labels.append(model.names[cls_id])

                    end_time = time.time()
                    fps = calculate_fps(start_time, end_time)
                    detection_time = round(end_time - start_time, 3)
                    total_frames += 1
                    total_time += detection_time

                    fps_placeholder.markdown(f"**FPS:** {fps}")
                    time_placeholder.markdown(f"**Detection Time:** {detection_time}s")

                    if detected_labels:
                        phrase = ", ".join(set(detected_labels))
                        if phrase != last_phrase:
                            gesture_count += 1
                            speak_phrase(f"Detected {phrase}")
                            last_phrase = phrase
                            st.session_state.last_phrase = phrase

                    FRAME_WINDOW.image(annotated_frame, channels="BGR")

                cap.release()
                avg_fps = round(total_frames / total_time, 2) if total_time > 0 else 0
                st.success(f"✅ Detection finished. Average FPS: {avg_fps}, Total Gestures: {gesture_count}")
            else:
                st.info("☝️ Click 'Start Real-Time Detection' to begin gesture recognition.")

    # -------------------
    # 🖼️ IMAGE MODE
    # -------------------
    elif mode == "🖼️ Image":
        st.markdown("#### 🖼️ Upload an Image for Detection")
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            start_time = time.time()
            results = model(image, conf=confidence, verbose=False)
            end_time = time.time()
            detection_time = round(end_time - start_time, 3)

            annotated = results[0].plot()
            detected_labels = [model.names[int(box.cls)] for box in results[0].boxes]
            st.image(annotated, channels="BGR", caption=f"Processed in {detection_time}s")

            if detected_labels:
                phrase = ", ".join(set(detected_labels))
                st.success(f"Detected: {phrase}")
                speak_phrase(f"Detected {phrase}")
            else:
                st.warning("No gestures detected.")

    # -------------------
    # 🎞️ VIDEO MODE
    # -------------------
    elif mode == "🎞️ Video":
        st.markdown("#### 🎞️ Upload a Video for Gesture Detection")
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            temp_dir = tempfile.NamedTemporaryFile(delete=False)
            temp_dir.write(uploaded_video.read())
            temp_dir.close()

            cap = cv2.VideoCapture(temp_dir.name)
            stframe = st.image([])
            fps_placeholder = st.empty()
            time_placeholder = st.empty()
            st.success("✅ Processing video...")

            detected_labels_all = set()
            total_frames = 0
            total_time = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                frame = cv2.flip(frame, 1)
                results = model(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                end_time = time.time()

                fps = calculate_fps(start_time, end_time)
                detection_time = round(end_time - start_time, 3)
                total_frames += 1
                total_time += detection_time

                fps_placeholder.markdown(f"**FPS:** {fps}")
                time_placeholder.markdown(f"**Frame Detection Time:** {detection_time}s")
                stframe.image(annotated_frame, channels="BGR")

                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    detected_labels_all.add(model.names[cls_id])

            cap.release()
            os.unlink(temp_dir.name)

            avg_fps = round(total_frames / total_time, 2) if total_time > 0 else 0
            st.info(f"🎬 Processed {total_frames} frames | Average FPS: {avg_fps}")

            if detected_labels_all:
                phrase = ", ".join(detected_labels_all)
                st.success(f"Detected gestures: {phrase}")
                speak_phrase(f"Detected {phrase}")
            else:
                st.warning("No gestures detected.")

# ----------------------------------------
# ⭐ RATE US PAGE
# ----------------------------------------
elif page == "⭐ Rate Us":
    st.header("⭐ Rate Team Gestura")
    st.markdown("We’d love your feedback! How was your experience using our ASL detection app?")

    rating = st.slider("Rate us out of 5 ⭐", 1, 5, 5)
    feedback = st.text_area("Your feedback:", placeholder="Tell us what you liked or what we can improve...")
    submitted = st.button("Submit Feedback")

    if submitted:
        st.success(f"✅ Thank you for rating us {rating}/5 stars!")
        if feedback:
            st.info(f"💬 Your feedback: {feedback}")
        speak_phrase("Thank you for your feedback!")

    st.markdown("---")
    st.caption("Developed by Team Gestura 💙")
