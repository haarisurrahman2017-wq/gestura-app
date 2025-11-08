---

#Team Gestura 🤟 – ASL Gesture Detection Web App

**Breaking Barriers, One Gesture at a Time**

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Demo](#demo)
* [Installation](#installation)
* [Usage](#usage)
* [Configuration](#configuration)
* [Supported Input Modes](#supported-input-modes)
* [Voice Feedback](#voice-feedback)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

**Team Gestura** is a web-based application for **real-time ASL (American Sign Language) gesture detection** built with:

* **Python**
* **Streamlit** for the web interface
* **YOLOv8** for gesture detection
* **OpenCV** for image/video processing

It allows users to detect hand gestures from a **webcam, uploaded images, or videos** and provides **voice feedback** using browser-based speech synthesis.

The app is designed to make **communication with the hearing-impaired easier** and to serve as a **practical demonstration of AI-based gesture recognition**.

---

## Features

* **Real-time ASL gesture detection** using YOLOv8
* **Multiple input modes:** webcam, image upload, video upload
* **Dashboard:** displays average FPS, total gestures detected, and model confidence
* **Voice feedback:** automatically announces detected gestures (via browser speech synthesis)
* **User feedback form:** allows users to rate and leave comments about the app
* Fully **web-based**, compatible with **Streamlit Cloud**

---

## Demo

> Live app: [https://gestura-app.streamlit.app](https://gestura-app.streamlit.app)

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/gestura-app.git
cd gestura-app
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the trained YOLOv8 model** (`best.pt`) and place it in the project root.

---

## Usage

1. **Run locally**

```bash
streamlit run website.py
```

2. **Open the URL** shown in your terminal (usually `http://localhost:8501`)

3. **Select input mode** from the sidebar:

   * 📷 Webcam
   * 🖼️ Image Upload
   * 🎞️ Video Upload

4. **Adjust settings**:

   * Confidence threshold
   * Enable/disable voice feedback

5. **View Dashboard** for real-time metrics like FPS and detected gestures.

---

## Configuration

* **Confidence Threshold:** Set a value between 0.0 and 1.0 to filter predictions.
* **Voice Feedback:** Toggle voice output on/off. On Streamlit Cloud, uses **browser speech synthesis**.

---

## Supported Input Modes

* **Webcam:** Capture gestures in real-time (local webcam or browser camera)
* **Image:** Upload an image and detect gestures
* **Video:** Upload a video file for frame-by-frame gesture detection

---

## Voice Feedback

* Uses **browser-based speech synthesis** on Streamlit Cloud.
* Automatically announces detected gestures when enabled.
* Local users (Linux) can optionally use `espeak` if desired.

---

## Contributing

Contributions are welcome! You can:

* Improve gesture detection
* Add new ASL gestures
* Enhance the UI or dashboard
* Fix bugs or optimize performance

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

**Team Gestura 💙 – Breaking barriers, one gesture at a time!**

---
