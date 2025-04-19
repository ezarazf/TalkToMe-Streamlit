import streamlit as st
import torch
import cv2
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Konfigurasi halaman
st.set_page_config(layout="wide")
st.title("Talk To Me")

# Ganti dengan URL raw GitHub model kamu
MODEL_URL = "https://raw.githubusercontent.com/FiZ18/HSC304-TalkToMe/main/AI%20Model%20v..0.1%20-%20TalkToMe%20.pt"
MODEL_PATH = Path("AI Model v..0.1 - TalkToMe.pt")

@st.cache_resource
def load_model():
    # Download model jika belum ada
    if not MODEL_PATH.exists():
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C"]  # Ganti sesuai kelas modelmu

# Video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.latest_result = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Preprocessing (180x180)
        img_resized = cv2.resize(img, (180, 180))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb / 255.0
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_labels[pred.item()]
        confidence = conf.item() * 100

        # Simpan hasil prediksi
        self.latest_result = {
            "label": label,
            "confidence": confidence,
            "waktu": datetime.now().strftime("%H:%M:%S")
        }

        # Tambahkan kotak dan label ke frame
        box_w, box_h = 200, 200
        x1 = w // 2 - box_w // 2
        y1 = h // 2 - box_h // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} ({confidence:.1f}%)"
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="sign-language-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan hasil prediksi
if webrtc_ctx.video_processor:
    result = webrtc_ctx.video_processor.latest_result
    if result is not None:
        st.markdown("### 🔍 Prediksi Terkini")
        st.info(f"{result['waktu']} – *{result['label']}* ({result['confidence']:.1f}%)")
    else:
        st.info("🧐 Menunggu prediksi...")
