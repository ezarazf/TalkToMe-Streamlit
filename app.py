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
        self.latest_result = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Resize dan normalisasi
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_labels[pred.item()]
        confidence = conf.item() * 100

        # Simpan prediksi
        self.latest_result = {
            "label": label,
            "confidence": confidence,
            "waktu": datetime.now().strftime("%H:%M:%S")
        }

        # Gambar kotak + label di atas frame
        box_w, box_h = 200, 200
        x1 = w // 2 - box_w // 2
        y1 = h // 2 - box_h // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({confidence:.1f}%)"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_x = x1 + (box_w - text_w) // 2
        text_y = y1 - 10

        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# Jalankan WebRTC
ctx = webrtc_streamer(
    key="sign-lang-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan teks prediksi di luar video (opsional)
if ctx.video_processor:
    result = ctx.video_processor.latest_result
    if result:
        st.markdown("### 🧾 Hasil Prediksi Terkini")
        st.info(f"🕒 {result['waktu']} → **{result['label']}** ({result['confidence']:.1f}%)")
