import av
import cv2
import torch
import streamlit as st
import numpy as np
from datetime import datetime
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

# Video Processor sederhana
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_result = None
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (180, 180))
        tensor = (torch.from_numpy(img_resized).float() / 255 - 0.5) / 0.5
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        self.last_result = {
            "label": class_labels[pred.item()],
            "confidence": conf.item() * 100,
            "time": datetime.now().strftime("%H:%M:%S")
        }
        
        return frame

# Stream video config
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False}
)

# Tampilkan hasil
if ctx.video_processor:
    if ctx.video_processor.last_result:
        result = ctx.video_processor.last_result
        st.markdown(f"""
        ### Hasil Deteksi
        **Waktu**: {result['time']}  
        **Huruf**: {result['label']}  
        **Akurasi**: {result['confidence']:.1f}%
        """)
    else:
        st.info("🔄 Memproses frame pertama...")
else:
    st.info("✅ Silakan aktifkan kamera di atas")
