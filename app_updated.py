import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import base64
import time
import matplotlib.pyplot as plt
from PIL import Image

# --- Drishti Access Custom Theming ---
st.set_page_config(page_title="People Detection - Drishti Access", layout="wide", page_icon="👁️")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #F0F8FF !important;
            font-family: 'Segoe UI', sans-serif;
        }
            
        .stApp {
        background-color: #F0F8FF;
        font-family: 'Segoe UI', sans-serif;
            }

        /* Sidebar Background */
        section[data-testid="stSidebar"] {
            background-color: #F0F8FF;
            border-right: 1px solid #d0d7e2;
        }

        .block-container {
            padding: 2rem 3rem;
        }

        .stButton>button {
            background-color: #7B61FF;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }

        .stDownloadButton>button {
            background-color: #7B61FF;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1rem;
        }

        h1, h2, h3, h4 {
            color: #1C1E21;
        }


        .heatmap-title {
            font-size: 1.4rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #1C1E21;
        }

        .heatmap-container {
            background-color: white;
            border: 1px solid #CCC;
            border-radius: 8px;
            padding: 10px;
        }

        .sidebar .sidebar-content {
            background-color: #F0F8FF;
        }
    </style>

    <style>
    .kpi-container {
        display: flex;
        gap: 20px;
        margin: 1.5rem 0;
    }

    .kpi-box {
        flex: 1;
        background-color: white;
        border: 2px solid #d0d7e2;
        border-radius: 12px;
        padding: .5rem 1rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }

    .kpi-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1C1E21;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1C1E21;
    }

    .kpi-value.green {
        color: green;
    }

    .kpi-value.red {
        color: red;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color:#7B61FF;'>👁️ Drishti Access: Intelligent Crowd Monitoring 🔍</h2>", unsafe_allow_html=True)
st.markdown("##### AI-powered Real-Time People Detection in ROI, Risk Alerts, Live KPIs and Heatmap Generation")

#st.title("👁️‍🗨️ People Detection with ROI, KPIs, and Heatmap")
#st.write("Upload a video to detect people inside/outside ROI, monitor crowd status, and visualize heatmap.")

uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

# Sidebar for ROI and threshold settings

# Option to toggle live preview
#st.sidebar.title("🧮 Video Preview")
st.sidebar.write('(Configure these settings before using the app)')
st.sidebar.header('📺 Video Setting')
show_live_preview = st.sidebar.toggle("Show / Hide Preview", value=True)
st.sidebar.header('👥 Threshold Setting')
alert_threshold = st.sidebar.slider("Overcrowd Threshold (people)", 1, 50, 7)
st.sidebar.header('🧮 ROI Settings')
roi_left = st.sidebar.slider("ROI Left (%)", 0, 100, 20)
roi_right = st.sidebar.slider("ROI Right (%)", 0, 100, 80)
roi_top = st.sidebar.slider("ROI Top (%)", 0, 100, 20)
roi_bottom = st.sidebar.slider("ROI Bottom (%)", 0, 100, 80)
st.sidebar.write('- ROI = Region of Interest (Yellow BBox)')
st.sidebar.write('- BBox = Bounding Box')

if uploaded_file:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert % ROI to pixels
    roi_x1, roi_y1 = int(W * roi_left / 100), int(H * roi_top / 100)
    roi_x2, roi_y2 = int(W * roi_right / 100), int(H * roi_bottom / 100)

    # Validate ROI
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        st.error("❌ Invalid ROI selection. Please adjust the ROI sliders in sidebar.")
        st.stop()

    output_path = video_path.replace(".mp4", "_output.mp4")
    model = YOLO("yolov8n.pt")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # KPI Layout
    st.markdown("#### ⭐KPIs")
    kpi_col1, kpi_col2, kpi_col3= st.columns(3)
    with kpi_col1:
        outside_kpi_box = st.empty()
        #inside_kpi = st.empty()
    with kpi_col2:
        inside_kpi_box = st.empty()
        #inside_kpi = st.empty()
    with kpi_col3:
        status_kpi_box = st.empty()
        #status_kpi = st.empty()

    st.markdown("")

    # Optional video preview
    if show_live_preview:
        st.markdown("#### ⭐Video Area")
        #progress_bar = st.progress(0)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎥 Original")
            original_placeholder = st.empty()
        with col2:
            st.markdown("#### 🧠 Annotated")
            annotated_placeholder = st.empty()

    progress_bar = st.progress(0)

    # Spinner placeholder for heatmap section
    st.markdown("---")
    heatmap_placeholder = st.empty()
    spinner_placeholder = st.empty()

    # Create ROI heatmap matrix
    heatmap = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1), dtype=np.float32)

    frame_idx = 0
    last_kpi_update = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()  # 🔧 Save unmodified version before drawing

        results = model(frame, verbose=False)[0]
        inside, outside = 0, 0

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                inside += 1
                color = (0, 255, 0)
                hx, hy = cx - roi_x1, cy - roi_y1
                if 0 <= hy < heatmap.shape[0] and 0 <= hx < heatmap.shape[1]:
                    cv2.circle(heatmap, (hx, hy), 8, 1, -1)  # Draw larger dot in heatmap
            else:
                outside += 1
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # Draw ROI box
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 12)
        cv2.putText(frame, f"Inside ROI: {inside}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Outside ROI: {outside}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(frame)
        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

        # ✅ Show toast if overcrowding is detected
        if inside > alert_threshold:
            st.toast("Alert: Overcrowding detected!", icon="🚨")

        # Update KPIs every second
        # In update loop:
        outside_kpi_box.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">👩‍👩‍👧‍👦 Outside ROI (Red BBox)</div>
                <div class="kpi-value">{outside}</div>
            </div>
        """, unsafe_allow_html=True)    

        inside_kpi_box.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">👥 Inside ROI (Green BBox)</div>
                <div class="kpi-value">{inside}</div>
            </div>
        """, unsafe_allow_html=True)    

        status_text = "✅ Safe" if inside <= alert_threshold else "❌ Overcrowded"
        status_color = "green" if inside <= alert_threshold else "red"

        status_kpi_box.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">📊 ROI Status (Threshold = {alert_threshold})</div>
                <div class="kpi-value {status_color}">{status_text}</div>
            </div>
        """, unsafe_allow_html=True)




        # Optional display of frames
        if show_live_preview:
            original_resized = cv2.resize(original_frame, (720, int(720 * H / W)))
            annotated_resized = cv2.resize(frame, (720, int(720 * H / W)))

            original_placeholder.image(original_resized, channels="BGR")
            annotated_placeholder.image(annotated_resized, channels="BGR")


    cap.release()
    out.release()

    spinner_placeholder.info("⏳ Generating heatmap...")
    time.sleep(1)  # simulate brief loading

    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.set_title("ROI Heatmap (People Density)")
    ax.axis('off')
    spinner_placeholder.empty()  # remove spinner
    heatmap_placeholder.pyplot(fig)

    st.success("✅ Processing complete!")

    with open(output_path, "rb") as file:
        st.download_button("⬇️ Download Annotated Video", file, file_name="annotated_output.mp4")
