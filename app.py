import logging
import queue
import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Setup halaman
st.title("Talk To Me: Penerjemah Bahasa Isyarat Real-Time")
st.markdown("""
    **Aplikasi penerjemah bahasa isyarat menggunakan AI**  
    *Arahkan tangan Anda dalam kotak kamera untuk memulai*
""")
st.divider()

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("model/SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C"]

# Video Processor
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
    
    def _preprocess(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (180, 180))
        img_normalized = (img_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing dan inference
        tensor = self._preprocess(img)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, pred = torch.max(probs, 1)
        
        # Return frame asli tanpa overlay
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
)

webrtc_ctx = webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=rtc_config
)
