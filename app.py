import av
import cv2
import torch
import queue
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

# Video Processor dengan penyimpanan hasil
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.result_queue = queue.Queue(maxsize=5)  # Buffer 5 hasil terakhir
    
    def _preprocess(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (180, 180))
        img_normalized = (img_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Preprocessing dan inference
            tensor = self._preprocess(img)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            # Simpan hasil terbaru
            result = {
                "label": class_labels[pred.item()],
                "confidence": conf.item() * 100,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
            self.result_queue.put(result)
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
        
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

# Tampilkan hasil prediksi
if webrtc_ctx.video_processor:
    result_placeholder = st.empty()
    
    # Inisialisasi state untuk hasil
    if 'latest_result' not in st.session_state:
        st.session_state.latest_result = None
    
    # Ambil hasil terbaru
    try:
        latest_result = webrtc_ctx.video_processor.result_queue.get_nowait()
        st.session_state.latest_result = latest_result
    except queue.Empty:
        pass
    
    # Tampilkan hasil
    if st.session_state.latest_result:
        result_placeholder.markdown(f"""
        ### 🎯 Hasil Prediksi Terbaru
        **Waktu**: {st.session_state.latest_result['timestamp']}  
        **Huruf**: {st.session_state.latest_result['label']}  
        **Akurasi**: {st.session_state.latest_result['confidence']:.1f}%
        """)
    else:
        result_placeholder.info("🔄 Menunggu frame pertama...")
else:
    st.info("✅ Silakan aktifkan kamera untuk memulai deteksi")

# Auto-refresh setiap 1 detik
st_autorefresh(interval=1000, key="refresh")
