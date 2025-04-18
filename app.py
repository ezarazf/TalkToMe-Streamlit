import cv2
import streamlit as st
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import uuid
from pathlib import Path


# Fungsi untuk deteksi tangan
@st.cache_resource
def load_model_cached():
    model = YOLO("SL-V1.torchscript")
    return model

# Fungsi untuk callback video stream
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Mengubah frame ke RGB untuk pemrosesan YOLO
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Model YOLO
    results = model(pil_img)

    labels = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
    confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []
    names = model.names

    if len(labels):
        max_confidence_index = np.argmax(confidences)
        hasil = names[int(labels[max_confidence_index])]
        confidence_score = confidences[max_confidence_index] * 100
        st.session_state.history.append({
            "predicted_class": hasil,
            "confidence_score": confidence_score
        })

        # Tampilkan prediksi pada frame
        cv2.putText(img, f"{hasil} ({confidence_score:.2f}%)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Direktori untuk menyimpan rekaman
RECORD_DIR = Path("./records")
RECORD_DIR.mkdir(exist_ok=True)

# Fungsi recorder untuk input dan output video
def in_recorder_factory() -> MediaRecorder:
    return MediaRecorder(str(RECORD_DIR / f"{uuid.uuid4()}_input.flv"), format="flv")

def out_recorder_factory() -> MediaRecorder:
    return MediaRecorder(str(RECORD_DIR / f"{uuid.uuid4()}_output.flv"), format="flv")


# Streamlit app
def app():
    if "history" not in st.session_state:
        st.session_state.history = []

    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_frame_callback=video_frame_callback,
        in_recorder_factory=in_recorder_factory,
        out_recorder_factory=out_recorder_factory,
    )

    # Menampilkan riwayat prediksi
    if st.session_state.history:
        st.subheader("Riwayat Prediksi: ")
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.write(f"{i}. Prediksi: {item['predicted_class']} dengan Confidence: {item['confidence_score']:.2f}%")


if __name__ == "__main__":
    app()
