import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Setup halaman
st.set_page_config(layout="wide")
st.title("Talk To Me: Pendeteksi Bahasa Isyarat")

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("model/SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C"]

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.model = model
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Resize dan normalisasi
        img_resized = cv2.resize(img, (180, 180))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_labels[pred.item()]
        confidence = conf.item() * 100

        # Simpan hasil ke session state
        st.session_state.latest_result = {
            "label": label,
            "confidence": confidence,
            "waktu": datetime.now().strftime("%H:%M:%S")
        }

        # Gambar bounding box dan label
        box_w, box_h = 200, 200
        x1 = w//2 - box_w//2
        y1 = h//2 - box_h//2
        cv2.rectangle(img, (x1, y1), (x1+box_w, y1+box_h), (0,255,0), 2)
        cv2.putText(img, 
                   f"{label} ({confidence:.1f}%)", 
                   (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (0,255,0), 2)

        return img

# Stream video
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Tampilkan hasil prediksi
if "latest_result" in st.session_state:
    result = st.session_state.latest_result
    st.markdown("### 🔍 Hasil Prediksi Terbaru")
    st.success(f"Waktu: {result['waktu']}")
    st.success(f"Prediksi: {result['label']}")
    st.success(f"Confidence: {result['confidence']:.1f}%")
else:
    st.info("🎥 Silakan mulai kamera untuk melihat prediksi...")
