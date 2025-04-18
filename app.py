import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Konfigurasi STUN server
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Setup halaman
st.set_page_config(layout="wide")
st.title("✋ Talk To Me: Pendeteksi Bahasa Isyarat")

# Load model
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C", "D", "E"]

# Video Processor yang dioptimalkan
class OptimizedVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.latest_result = None
        self.frame_count = 0  # Untuk skip frame
        self.last_prediction = None
        self.last_confidence = 0.0

    def recv(self, frame):
        self.frame_count += 1
        
        # Skip 1 frame untuk mengurangi beban (30fps -> ~15fps)
        if self.frame_count % 2 == 0:
            return frame
        
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # 1. Resize awal ke resolusi lebih kecil
        img_small = cv2.resize(img, (320, 240))
        
        # 2. Preprocessing yang lebih efisien
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # 3. Konversi ke tensor
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

        # 4. Inference model
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_labels[pred.item()]
        confidence = conf.item() * 100

        # Simpan hasil prediksi
        self.last_prediction = label
        self.last_confidence = confidence

        # 5. Overlay yang dioptimalkan
        overlay = img.copy()
        box_w, box_h = 200, 200
        x1 = w//2 - box_w//2
        y1 = h//2 - box_h//2
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # Gambar overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} ({confidence:.1f}%)"
        cv2.putText(overlay, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        # Gabungkan dengan frame asli dengan alpha blending
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        return img

# Stream video dengan konfigurasi yang dioptimalkan
ctx = webrtc_streamer(
    key="optimized_demo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=OptimizedVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},  # Turunkan resolusi
            "height": {"ideal": 480},
            "frameRate": 15  # Turunkan frame rate
        },
        "audio": False
    },
    async_processing=True,  # Wajib diaktifkan untuk performa
    key_frame_interval=2000,  # Untuk koneksi yang lebih stabil
)

# Tampilkan prediksi teks
if ctx.video_processor:
    st.markdown("### 🔍 Prediksi Langsung")
    prediction_placeholder = st.empty()
    
    # Update prediksi dengan polling
    while ctx.state.playing:
        if ctx.video_processor.last_prediction:
            prediction_placeholder.info(
                f"{datetime.now().strftime('%H:%M:%S')} – "
                f"*{ctx.video_processor.last_prediction}* "
                f"({ctx.video_processor.last_confidence:.1f}%)"
            )
        else:
            prediction_placeholder.info("🧐 Menunggu prediksi pertama...")
