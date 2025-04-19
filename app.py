import logging
import queue
import av
import cv2
import numpy as np
import streamlit as st
import torch
from datetime import datetime
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Setup logging
logger = logging.getLogger(__name__)

# Setup halaman
st.set_page_config(page_title="Talk To Me", layout="wide")
st.title("🎥 Talk To Me: Penerjemah Bahasa Isyarat Real-Time")
st.markdown("""
    **Aplikasi penerjemah bahasa isyarat menggunakan AI**  
    *Arahkan tangan Anda dalam kotak kamera untuk menerjemahkan ke bahasa isyarat*
""")
st.divider()

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("model/SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C"]  # Sesuaikan dengan kelas model Anda

# Session state untuk menyimpan hasil
if 'latest_result' not in st.session_state:
    st.session_state.latest_result = None

# Video Processor
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.result_queue = queue.Queue()
    
    def _preprocess(self, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize ke 180x180
        img_resized = cv2.resize(img_rgb, (180, 180))
        
        # Normalisasi
        img_normalized = (img_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Convert ke tensor
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Preprocessing
        tensor = self._preprocess(img)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        label = class_labels[pred.item()]
        confidence = conf.item() * 100
        
        # Gambar bounding box di tengah
        box_size = 180  # Ukuran sesuai input model
        x1 = w//2 - box_size//2
        y1 = h//2 - box_size//2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({confidence:.1f}%)", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Simpan hasil ke queue
        result = {
            "label": label,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        self.result_queue.put(result)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=rtc_config
)

# Tampilkan hasil
if webrtc_ctx.video_processor:
    result_container = st.empty()
    
    while True:
        try:
            result = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
            st.session_state.latest_result = result
            
            # Update tampilan hasil
            result_container.markdown(f"""
            ### Hasil Prediksi Terbaru
            - **Waktu**: `{result['timestamp']}`
            - **Label**: `{result['label']}`
            - **Confidence**: `{result['confidence']:.2f}%`
            """)
            
        except queue.Empty:
            continue
