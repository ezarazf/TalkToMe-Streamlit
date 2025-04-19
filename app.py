import av
import cv2
import torch
import streamlit as st
import numpy as np
from datetime import datetime
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Setup halaman
st.set_page_config(page_title="Talk To Me", layout="wide")
st.title("🎥 Talk To Me: Penerjemah Bahasa Isyarat Real-Time")
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

# Video Processor dengan error handling
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_result = None
    
    def _preprocess(self, img):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (180, 180))
            return (img_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            processed_img = self._preprocess(img)
            if processed_img is not None:
                tensor = torch.from_numpy(processed_img).permute(2, 0, 1).unsqueeze(0).float()
                
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                self.last_result = {
                    "label": class_labels[pred.item()],
                    "confidence": conf.item() * 100,
                    "time": datetime.now().strftime("%H:%M:%S")
                }
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        
        return frame

# Konfigurasi WebRTC
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Tampilkan hasil
if ctx.video_processor:
    if hasattr(ctx.video_processor, 'last_result') and ctx.video_processor.last_result:
        result = ctx.video_processor.last_result
        st.success(f"""
        ### 🎯 Hasil Terbaru
        **Waktu**: {result['time']}  
        **Huruf**: {result['label']}  
        **Tingkat Akurasi**: {result['confidence']:.1f}%
        """)
    else:
        st.info("🔄 Sedang memproses frame pertama...")
else:
    st.info("✅ Silakan klik 'START' di atas untuk memulai kamera")
