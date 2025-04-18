import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import torch
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Talk To Me")

# Load model YOLO
@st.cache_resource
def load_model_cached():
    model = YOLO("SL-V1.pt")  # Ubah ke model yang sesuai
    return model

model = load_model_cached()

# Menyimpan status dan riwayat prediksi
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []

# SIDEBAR
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Setelah tekan 'Start', harap langsung menunjukan tangan kamu ke kamera.")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Remove History")

# Video stream handler dengan callback
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.run = False

    def recv(self, frame):
        if self.run:
            # Ambil frame sebagai gambar
            img = frame.to_image()
            img_rgb = np.array(img)

            # Deteksi dengan model YOLO
            results = model(img)
            labels = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []
            names = model.names

            if len(labels):
                max_confidence_index = np.argmax(confidences)
                hasil = names[int(labels[max_confidence_index])]
                confidence_score = confidences[max_confidence_index] * 100
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Simpan riwayat
                st.session_state.history.append({
                    "predicted_class": hasil,
                    "confidence_score": confidence_score,
                    "timestamp": timestamp
                })

                # Mengupdate tampilan
                frame = cv2.putText(img_rgb, f"{hasil} ({confidence_score:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame

        return frame

# WebRTC stream
webrtc_streamer(
    key="sign-language-app",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]},
)

# Kontrol Start/Stop
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# Riwayat Prediksi
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(f"**{i}.** Prediksi: {item['predicted_class']} dengan Confidence: {item['confidence_score']:.2f}% pada {item['timestamp']}")
