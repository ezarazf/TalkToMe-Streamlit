import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Konfigurasi STUN
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ————————————————————————————————————————————————
st.set_page_config(layout="wide")
st.title("Talk To Me")

# Sidebar controls
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Klik ‘Start’ untuk mulai kamera, ‘Stop’ untuk menghentikan.")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Hapus Riwayat")

# Session state
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# Toggle state
if start:
    st.session_state.run = True
    st.session_state.show_history = True  # Tampilkan riwayat saat start
if stop:
    st.session_state.run = False

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()

class_labels = ["A", "B", "C", "D", "E"]  # Sesuaikan dengan label Anda

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.labels = class_labels
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if st.session_state.run:
            # Preprocessing
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalisasi
            img_normalized = img_rgb / 255.0
            img_normalized = (img_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
            # Convert to tensor
            tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
            
            # Inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            # Get results
            label = self.labels[pred.item()]
            confidence = conf.item() * 100
            
            # Anotasi frame
            cv2.putText(img, 
                       f"{label} {confidence:.1f}%", 
                       (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # Simpan ke riwayat
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append({
                "waktu": ts,
                "label": label,
                "confidence": confidence,
                "frame": img.copy()
            })
            
            # Trigger re-render
            st.experimental_rerun()
        
        return img

# Tampilkan WebRTC
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan hasil di bawah WebRTC
if st.session_state.show_history:
    st.subheader("📜 Hasil Prediksi Real-Time")
    
    # Buat container untuk hasil
    result_container = st.container()
    
    # Tampilkan 3 hasil terbaru secara horizontal
    if st.session_state.history:
        cols = result_container.columns(3)
        for idx, entry in enumerate(reversed(st.session_state.history[-3:])):
            with cols[idx]:
                st.image(entry["frame"], caption=f"🕒 {entry['waktu']} | 👆 {entry['label']} ({entry['confidence']:.1f}%)")

# Hapus riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("Riwayat telah dihapus")
    st.rerun()
