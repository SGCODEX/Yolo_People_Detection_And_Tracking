# 👁️ Drishti Access: Intelligent Crowd Monitoring
It is a real-time, AI-powered crowd monitoring tool built using YOLOv8 and Streamlit. It enables smart detection of people in videos, highlights crowd density in a customizable Region of Interest (ROI), and visualizes live KPIs and heatmaps. This tool is ideal for public safety teams, surveillance system designers, and event organizers to assess crowd behavior and trigger overcrowding alerts intelligently.

---

## 🚀 Features

- **🧠 Real-Time People Detection with YOLOv8**
  Upload any MP4 video and automatically detect people inside and outside a selected ROI.

- **🌐 ROI-Based Intelligence**
  Customize Region of Interest using sidebar sliders.
  Inside/Outside bounding box tracking.

- **📊 Live KPIs**
  Display total number of people inside/outside ROI.
  Highlight current crowd status (Safe / Overcrowded).

- **🔍 Heatmap Generation**
  Automatically generate a density heatmap of detected people within ROI.

- **🚨 Overcrowding Alerts**
  Real-time warning toast messages when overcrowding is detected.
  Configurable crowd threshold.

- **🔹 Annotated Video Export**
  Download the final annotated video showing detection results and ROI bounding box.

- **📅 Modern UI**
  Clean, accessible, and responsive Streamlit interface with custom theming and controls.

---

## 💡 Topics Covered

- Streamlit UI Design
- OpenCV Video Processing
- YOLOv8 Person Detection (Ultralytics)
- Live Heatmap Generation
- Custom KPI Components with HTML/CSS
- Real-Time Alerting with st.toast()
- Session Management for Alert State

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/drishti-access-people-tracker.git
   cd drishti-access-people-tracker

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Run the app:**
    ```bash
    streamlit run app_updated.py

---
## 📺 Demo

---
**Made by Shivam Gupta for Google Agentic AI Day Hackathon**
