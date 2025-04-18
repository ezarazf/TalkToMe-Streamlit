import streamlit as st
import torch
import numpy as np
from datetime import datetime
from PIL import Image
import streamlit_webrtc
import cv2

st.set_page_config(layout="wide")
st.title("Talk To Me")

# Sidebar
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Setelah tekan 'Start', harap langsung menunjukan tangan kamu ke kamera.")
    st.info("💡 Klik 'Stop' terlebih dahulu sebelum 'Remove History'")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Remove History")

# Load TorchScript model
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript")
    model.eval()
    return model

model = load_model()

# Inisialisasi session state
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Fungsi deteksi
def detect(model, frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0]  # Asumsikan model output [boxes]

    # Dummy: misalnya output label paling tinggi (ganti sesuai model kamu)
    if output is not None and output.shape[0] > 0:
        label = "Detected"
        confidence = 0.95
        return label, confidence
    else:
        return None, None

# Fungsi untuk menangani stream dari WebRTC
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    if st.session_state.run:
        label, confidence = detect(model, img)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if label:
            prediction_placeholder.success(f"🧠 Prediksi: {label} (Confidence: {confidence*100:.2f}%) - {timestamp}")
            st.session_state.history.append({
                "input_image": img,
                "predicted_class": label,
                "confidence_score": confidence * 100,
                "timestamp": timestamp
            })
        else:
            prediction_placeholder.info("Belum terdeteksi.")
    return img

# Video Stream WebRTC
video_stream = streamlit_webrtc.VideoProcessorBase(callback=video_frame_callback)
webrtc_streamer = streamlit_webrtc.webrtc_streamer(
    key="video-stream",
    video_processor_factory=video_stream,
    media_stream_constraints={"video": True},
)

# Riwayat prediksi
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(f"**{i}.** Prediksi: {item['predicted_class']} dengan Confidence: {item['confidence_score']:.2f}% pada {item['timestamp']}")
        st.image(item['input_image'], caption=f"Gambar {i}", use_container_width=True)

# Hapus riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("✅ Semua riwayat berhasil dihapus.")
