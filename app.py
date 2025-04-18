import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Konfigurasi STUN server
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
})

# Set layout halaman
st.set_page_config(layout="wide")
st.title("Talk To Me")

# State session
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

# Sidebar kontrol
with st.sidebar:
    st.title("🎮 Kontrol")
    st.info("💡 Klik ‘Start’ untuk mulai kamera, ‘Stop’ untuk menghentikan.")
    if st.button("▶️ Start"):
        state.run = True
    if st.button("⏹️ Stop"):
        state.run = False
        state.webrtc_ctx = None
    clear_history = st.button("🧹 Hapus Riwayat")

# Load model (TorchScript)
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C", "D", "E"]  # Ganti sesuai label kamu

# Video Processor
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

            # Normalisasi
            img_normalized = (img_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            label = class_labels[pred.item()]
            confidence = conf.item() * 100

            # Tampilkan teks ke frame
            cv2.putText(img, f"{label} {confidence:.1f}%", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Tambahkan ke riwayat hanya jika berbeda dari prediksi sebelumnya
            if not state.history or label != state.history[-1]["label"]:
                state.history.append({
                    "waktu": datetime.now().strftime("%H:%M:%S"),
                    "label": label,
                    "confidence": confidence,
                    "frame": img.copy()
                })
                st.session_state.force_rerun = True

            return img

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            self.active = False
            return frame.to_ndarray(format="bgr24")

    def on_ended(self):
        self.active = False

# Jalankan WebRTC
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

# Sidebar: prediksi terakhir
with st.sidebar:
    if state.history:
        latest = state.history[-1]
        st.success(f"📢 Terakhir: **{latest['label']}** ({latest['confidence']:.1f}%)")

# Rerun otomatis jika ada prediksi baru
if st.session_state.get("force_rerun", False):
    st.session_state.force_rerun = False
    st.experimental_rerun()

# Tampilkan hasil prediksi
if state.history:
    st.subheader("📜 Hasil Prediksi Real-Time")
    cols = st.columns(3)
    for idx, entry in enumerate(reversed(state.history[-3:])):  # 3 prediksi terakhir
        with cols[idx % 3]:
            st.image(entry["frame"], caption=f"{entry['waktu']} - {entry['label']} ({entry['confidence']:.1f}%)")

# Hapus riwayat jika ditekan
if clear_history:
    state.history.clear()
    st.rerun()
