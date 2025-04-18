import streamlit as st
import torch
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(layout="wide")
st.title("Talk To Me - WebRTC Version")

# Sidebar
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Setelah tekan 'Start', harap langsung tunjukkan tangan kamu ke kamera.")
    st.info("💡 Klik 'Stop' terlebih dahulu sebelum 'Remove History'")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Remove History")

# Load TorchScript model
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()

# Session states
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

# Deteksi tangan (dummy example)
def detect(model, frame: np.ndarray):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        out = model(tensor)[0]
    # Misal: jika ada deteksi
    if out is not None and out.shape[0] > 0:
        return "Tangan", 0.98
    else:
        return None, None

# Callback untuk tiap frame WebRTC
def video_frame_callback(frame):
    img_bgr = frame.to_ndarray(format="bgr24")
    if st.session_state.run and model is not None:
        label, conf = detect(model, img_bgr)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if label:
            prediction_placeholder.success(f"🧠 Prediksi: {label} (Confidence: {conf*100:.2f}%) - {timestamp}")
            st.session_state.history.append({
                "input_image": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                "predicted_class": label,
                "confidence_score": conf*100,
                "timestamp": timestamp
            })
        else:
            prediction_placeholder.info("Belum terdeteksi.")
    # kembalikan frame agar muncul di browser
    return frame

# Jalankan WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="video-stream",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Riwayat prediksi
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(f"*{i}.* Prediksi: {item['predicted_class']} "
                  f"dengan Confidence: {item['confidence_score']:.2f}% "
                  f"pada {item['timestamp']}")
        st.image(item['input_image'], use_container_width=True)

# Hapus riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("✅ Semua riwayat berhasil dihapus!")
