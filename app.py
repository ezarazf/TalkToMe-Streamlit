import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Konfigurasi WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Set layout
st.set_page_config(layout="wide")
st.title("Talk To Me")

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C", "D", "E"]

# Video processor
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.latest_result = None  # tempat simpan hasil prediksi terbaru

    def recv(self, frame):
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

        # Simpan hasil prediksi terbaru
        self.latest_result = {
            "label": label,
            "confidence": confidence,
            "waktu": datetime.now().strftime("%H:%M:%S")
        }

        # Tampilkan teks ke frame
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Jalankan webrtc
ctx = webrtc_streamer(
    key="sign-lang-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan hasil prediksi terbaru
if ctx.video_processor:
    result = ctx.video_processor.latest_result
    if result:
        st.markdown("### 🔤 Hasil Prediksi Terbaru")
        st.success(f"🕒 {result['waktu']} → **{result['label']}** ({result['confidence']:.1f}%)")
