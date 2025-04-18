import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Talk To Me")

# SIDEBAR
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Setelah tekan 'Start', harap langsung menunjukan tangan kamu ke kamera.")
    st.info("💡 Klik 'Stop' terlebih dahulu sebelum 'Remove History")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Remove History")

# Load model
@st.cache_resource
def load_model_cached():
    model = YOLO("SL-V1.pt")
    return model

model = load_model_cached()

# Menyimpan status dan riwayat prediksi
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []

# Menangani kontrol untuk mulai dan berhenti
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Prediksi
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        st.warning("Kamera tidak terdeteksi.")
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Konversi ke format YOLO (PIL)
        img = Image.fromarray(frame_rgb)
        results = model(img)

        # Ambil label deteksi paling yakin dan confidence score
        labels = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []
        names = model.names
        
        if len(labels):
            # Ambil label dengan confidence tertinggi
            max_confidence_index = np.argmax(confidences)
            hasil = names[int(labels[max_confidence_index])]
            confidence_score = confidences[max_confidence_index] * 100 

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            prediction_placeholder.success(f"🧠 Prediksi: {hasil} (Confidence: {confidence_score:.2f}%) - {timestamp}")

            # Simpan hasil prediksi ke riwayat
            st.session_state.history.append({
                "input_image": frame_rgb,  # Simpan gambar input
                "predicted_class": hasil,   # Simpan hasil prediksi
                "confidence_score": confidence_score,  # Simpan confidence score
                "timestamp": timestamp      # Simpan timestamp
            })
        else:
            prediction_placeholder.info("Belum terdeteksi")

    cap.release()

# riwayat prediksi
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(f"**{i}.** Prediksi: {item['predicted_class']} dengan Confidence: {item['confidence_score']:.2f}% pada {item['timestamp']}")
        st.image(item['input_image'], caption=f"Gambar {i}", use_container_width=True)

# Menghapus semua riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("✅ Semua riwayat prediksi berhasil dihapus!")