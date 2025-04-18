import streamlit as st
import torch
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Konfigurasi STUN
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C", "D", "E"]  # Sesuaikan dengan label model Anda

class RealTimeProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.model = model
        self.last_prediction = "👋"

    def annotate_frame(self, frame, prediction, confidence):
        """Tambahkan annotasi ke frame"""
        text = f"{prediction} ({confidence:.1f}%)"
        
        # Tambahkan background box
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (20 + text_width, 40 + text_height),
            (0, 0, 0),
            -1
        )
        
        # Tambahkan text
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),  # Warna hijau
            2
        )
        return frame

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Preprocessing
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalisasi
            img_normalized = (img_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
            # Inference
            tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            # Ambil hasil
            label = class_labels[pred.item()]
            confidence = conf.item() * 100
            self.last_prediction = label
            
            # Annotasi frame
            img = self.annotate_frame(img, label, confidence)

        except Exception as e:
            cv2.putText(
                img,
                "ERROR",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),  # Warna merah untuk error
                2
            )
        
        return img

# UI
st.set_page_config(layout="wide")
st.title("Sign Language Recognition - Realtime")

with st.sidebar:
    st.header("Kontrol")
    st.info("Tekan START untuk memulai kamera dan prediksi")

# Streamer WebRTC
webrtc_ctx = webrtc_streamer(
    key="realtime-sign-language",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=RealTimeProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False
    },
    async_processing=True,
)

# Status prediksi di sidebar
if webrtc_ctx.video_processor:
    with st.sidebar:
        st.subheader("Status Prediksi")
        status_placeholder = st.empty()
        status_placeholder.write(f"🔄 Sedang memproses... Prediksi terakhir: {webrtc_ctx.video_processor.last_prediction}")
