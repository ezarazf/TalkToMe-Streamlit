import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Konfigurasi STUN server
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ————————————————————————————————————————————————
st.set_page_config(layout="wide")
st.title("Talk To Me – WebRTC + TorchScript")

# Sidebar controls
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Klik ‘Start’ untuk mulai kamera, ‘Stop’ untuk menghentikan, lalu ‘Remove History’ untuk menghapus riwayat.")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Remove History")

# Session state initialization
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()

# Ganti dengan label yang sesuai
class_labels = ["A", "B", "C", "D", "E"]  # Contoh label

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.labels = class_labels
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if st.session_state.run:
            try:
                # Preprocessing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (224, 224))
                
                # Normalisasi (sesuaikan dengan preprocessing saat training)
                img_normalized = img_resized / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_normalized = (img_normalized - mean) / std
                
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
                
                # Draw prediction
                cv2.putText(img, 
                           f"{label} {confidence:.2f}%", 
                           (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # Save to history
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append({
                    "image": img.copy(),
                    "prediction": label,
                    "confidence": confidence,
                    "timestamp": ts
                })
                
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
                cv2.putText(img, "ERROR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return img

# WebRTC Streamer
webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan riwayat
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    cols = st.columns(3)
    for idx, entry in enumerate(reversed(st.session_state.history)):
        with cols[idx % 3]:
            st.image(entry["image"], use_column_width=True)
            st.caption(f"{entry['timestamp']} - {entry['prediction']} ({entry['confidence']:.2f}%)")

if clear_history:
    st.session_state.history.clear()
    st.rerun()
