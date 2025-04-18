import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Talk To Me - TorchScript Version")

with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Setelah tekan 'Start', harap langsung menunjukan tangan kamu ke kamera.")
    st.info("💡 Klik 'Stop' terlebih dahulu sebelum 'Remove History'")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Remove History")

# Load model
@st.cache_resource
def load_model_cached():
    try:
        model = torch.jit.load("SL-V1.torchscript", map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model_cached()
class_labels = ["A", "B", "C", "D", "E"]  # Ganti sesuai label model

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

# Loop kamera
if st.session_state.run and model is not None:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ Kamera tidak terdeteksi.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(frame_rgb, (224, 224))
        tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            confidence_score, predicted_class = torch.max(probs, 0)
            hasil = class_labels[predicted_class.item()]
            score = confidence_score.item() * 100

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prediction_placeholder.success(f"🧠 Prediksi: {hasil} (Confidence: {score:.2f}%) - {timestamp}")
        frame_placeholder.image(frame_rgb, channels="RGB")

        st.session_state.history.append({
            "input_image": frame_rgb,
            "predicted_class": hasil,
            "confidence_score": score,
            "timestamp": timestamp
        })

        # Delay kecil untuk hindari loop berat
        if not st.button("Stop", key="stop_button_loop"):
            break

    cap.release()

# Tampilkan Riwayat
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(f"**{i}.** Prediksi: {item['predicted_class']} dengan Confidence: {item['confidence_score']:.2f}% pada {item['timestamp']}")
        st.image(item['input_image'], caption=f"Gambar {i}", use_container_width=True)

# Hapus Riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("✅ Semua riwayat prediksi berhasil dihapus!")
