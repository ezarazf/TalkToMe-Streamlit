import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Improved STUN configuration with fallback servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
})

# ————————————————————————————————————————————————
st.set_page_config(layout="wide")
st.title("Talk To Me")

# Session state management
class SessionState:
    def __init__(self):
        self.run = False
        self.history = []
        self.webrtc_ctx = None

def get_state() -> SessionState:
    if "state" not in st.session_state:
        st.session_state.state = SessionState()
    return st.session_state.state

state = get_state()

# Sidebar controls
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Klik ‘Start’ untuk mulai kamera, ‘Stop’ untuk menghentikan.")
    if st.button("▶️ Start"):
        state.run = True
    if st.button("⏹️ Stop"):
        state.run = False
        state.webrtc_ctx = None  # Force cleanup
    clear_history = st.button("🧹 Hapus Riwayat")

# Model loading
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C"]

# Video processor with connection cleanup
class SafeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.active = True
        
    def recv(self, frame):
        if not self.active or not state.run:
            return frame.to_ndarray(format="bgr24")
            
        try:
            img = frame.to_ndarray(format="bgr24")
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalization
            img_normalized = (img_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
            tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            label = class_labels[pred.item()]
            confidence = conf.item() * 100
            
            cv2.putText(img, f"{label} {confidence:.1f}%", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            state.history.append({
                "waktu": datetime.now().strftime("%H:%M:%S"),
                "label": label,
                "confidence": confidence,
                "frame": img.copy()
            })
            
            return img
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            self.active = False
            return frame.to_ndarray(format="bgr24")

    def on_ended(self):
        self.active = False

# WebRTC component management
if state.run:
    state.webrtc_ctx = webrtc_streamer(
        key="sign-language",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=SafeVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    if state.webrtc_ctx:
        state.webrtc_ctx.stop()
    state.webrtc_ctx = None

# Display predictions
if state.history:
    st.subheader("Hasil Prediksi")
    cols = st.columns(3)
    for idx, entry in enumerate(reversed(state.history[-3:])):
        with cols[idx % 3]:
            st.image(entry["frame"], caption=f"{entry['waktu']} - {entry['label']} ({entry['confidence']:.1f}%)")

if clear_history:
    state.history.clear()
    st.rerun()
