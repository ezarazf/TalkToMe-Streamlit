import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🧠 Talk To Me")

# SIDEBAR
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
        model = YOLO("SL-V1.pt")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.warning("🔁 Menggunakan model cadangan 'yolov8n.pt'")
        try:
            model = YOLO("yolov8n.pt")  # model umum dari Ultralytics
            return model
        except Exception as e:
            st.error(f"Gagal memuat model cadangan: {e}")
            return None

model = load_model_cached()

# State
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []

# Kontrol
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Prediksi
if st.session_state.run and model is not None:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        st.warning("Kamera tidak terdeteksi.")
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Deteksi
        img = Image.fromarray(frame_rgb)
        results = model(img)

        boxes = results[0].boxes
        if boxes is not None and boxes.cls is not None:
            labels = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            names = model.names

            if len(labels):
                max_conf_index = np.argmax(confidences)
                hasil = names[int(labels[max_conf_index])]
                confidence_score = confidences[max_conf_index] * 100
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                prediction_placeholder.success(f"🧠 Prediksi: {hasil} (Confidence: {confidence_score:.2f}%) - {timestamp}")
                st.session_state.history.append({
                    "input_image": frame_rgb,
                    "predicted_class": hasil,
                    "confidence_score": confidence_score,
                    "timestamp": timestamp
                })
            else:
                prediction_placeholder.info("🖐 Belum terdeteksi.")
        else:
            prediction_placeholder.info("📷 Tidak ada hasil deteksi.")

    cap.release()

# Riwayat
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(f"**{i}.** Prediksi: {item['predicted_class']} dengan Confidence: {item['confidence_score']:.2f}% pada {item['timestamp']}")
        st.image(item['input_image'], caption=f"Gambar {i}", use_container_width=True)

# Hapus Riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("✅ Semua riwayat prediksi berhasil dihapus!")
