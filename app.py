import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

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

# Load your TorchScript model once
@st.cache_resource
def load_model():
    m = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    m.eval()
    return m

model = load_model()

# Replace these labels with yours
class_labels = ["A", "B", "C", "D", "E"]  

# VideoProcessor untuk setiap frame
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        if st.session_state.run and model is not None:
            # Pre‑processing untuk model
            img_resized = cv2.resize(img_bgr, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )

            # Inference
            with torch.no_grad():
                out = model(tensor)[0]  # asumsikan ini klasifikasi
                probs = torch.nn.functional.softmax(out, dim=0)
                conf, pred = torch.max(probs, 0)
                label = class_labels[pred.item()]
                score = conf.item() * 100

            # Annotate frame
            text = f"{label} {score:.1f}%"
            cv2.putText(
                img_bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Simpan ke history
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.append({
                "input_image": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                "predicted_class": label,
                "confidence_score": score,
                "timestamp": ts
            })

        return img_bgr

# Jalankan WebRTC streamer
webrtc_streamer(
    key="webrtc",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan riwayat
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(
            f"*{i}.* Prediksi: {item['predicted_class']} "
            f"({item['confidence_score']:.2f}%) "
            f"pada {item['timestamp']}"
        )
        st.image(item["input_image"], use_column_width=True)

# Hapus riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("✅ Riwayat prediksi telah dihapus.")
