import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
import threading
from queue import Queue

# Konfigurasi STUN server
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Setup halaman
st.set_page_config(layout="wide")
st.title("Talk To Me)")

# Load model dengan cache
@st.cache_resource
def load_model():
    model = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    model.eval()
    return model

model = load_model()
class_labels = ["A", "B", "C", "D", "E"]

# System Resource Manager
class ResourceManager:
    def __init__(self):
        self.frame_queue = Queue(maxsize=2)  # Batasi antrian frame
        self.latest_result = None
        self.lock = threading.Lock()
        self.active = True

    def update_result(self, result):
        with self.lock:
            self.latest_result = result

    def get_result(self):
        with self.lock:
            return self.latest_result

resource_manager = ResourceManager()

# Video Processor yang dioptimalkan
class EfficientVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.resolution = (320, 240)  # Resolusi lebih rendah
        self.frame_skip = 2  # Skip 1 dari 2 frame
        self.frame_counter = 0
        
    def recv(self, frame):
        if not resource_manager.active:
            return frame
        
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return frame
            
        try:
            img = frame.to_ndarray(format="bgr24")
            img_small = cv2.resize(img, self.resolution)
            
            # Masukkan frame ke antrian
            if resource_manager.frame_queue.qsize() < 2:
                resource_manager.frame_queue.put(img_small)
                
        except Exception as e:
            st.error(f"Frame processing error: {str(e)}")
            
        return frame

# Background worker untuk inference
def model_worker():
    while resource_manager.active:
        try:
            if resource_manager.frame_queue.empty():
                continue
                
            img = resource_manager.frame_queue.get()
            
            # Preprocessing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_normalized = (img_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

            # Inference
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            # Update hasil
            result = {
                "label": class_labels[pred.item()],
                "confidence": conf.item() * 100,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            resource_manager.update_result(result)
            
        except Exception as e:
            st.error(f"Inference error: {str(e)}")

# Mulai worker thread
threading.Thread(target=model_worker, daemon=True).start()

# Stream video
ctx = webrtc_streamer(
    key="optimized_server",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EfficientVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": 15  # Frame rate lebih rendah
        },
        "audio": False
    },
    async_processing=True,
)

# Tampilkan UI
result_placeholder = st.empty()
server_status = st.sidebar.empty()

# System monitor
while True:
    try:
        # Update tampilan
        result = resource_manager.get_result()
        if result:
            result_placeholder.info(
                f"🕒 {result['timestamp']} - "
                f"**{result['label']}** "
                f"(Confidence: {result['confidence']:.1f}%)"
            )
            
        # System status
        server_status.markdown(f"""
        **🖥️ Server Status**
        - Frame Queue: {resource_manager.frame_queue.qsize()}
        - Active Threads: {threading.active_count()}
        """)
        
    except Exception as e:
        st.error(f"UI update error: {str(e)}")
