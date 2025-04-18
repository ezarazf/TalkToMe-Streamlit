import streamlit as st
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Set halaman Streamlit
st.set_page_config(layout="wide")
st.title("Talk To Me")

# SIDEBAR
with st.sidebar:
    st.title("Kontrol")
    st.info("💡 Setelah tekan 'Start', harap langsung menunjukan tangan kamu ke kamera.")
    st.info("💡 Klik 'Stop' terlebih dahulu sebelum 'Remove History'")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")
    clear_history = st.button("🧹 Remove History")

# Konfigurasi RTC untuk WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Load model YOLO
@st.cache_resource
def load_model_cached():
    model = YOLO("SL-V1.pt")
    return model

model = load_model_cached()

# Menyimpan status dan riwayat prediksi
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []

# Menangani kontrol untuk mulai dan berhenti
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# Placeholder untuk gambar dan prediksi
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Kelas untuk memproses frame video menggunakan WebRTC
class RealTimeProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.active = True
        
    def recv(self, frame):
        if not self.active or not st.session_state.run:
            return frame.to_ndarray(format="bgr24")

        try:
            # Ambil frame dalam format numpy array
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Konversi gambar menjadi PIL
            img_pil = Image.fromarray(img_rgb)
            results = self.model(img_pil)

            # Ambil label deteksi dan confidence
            labels = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []
            names = self.model.names

            if len(labels):
                # Ambil label dengan confidence tertinggi
                max_confidence_index = np.argmax(confidences)
                hasil = names[int(labels[max_confidence_index])]
                confidence_score = confidences[max_confidence_index] * 100

                # Menambahkan teks prediksi ke gambar
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(img, f"{hasil} ({confidence_score:.2f}%)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Menampilkan prediksi di Streamlit
                prediction_placeholder.success(f"🧠 Prediksi: {hasil} (Confidence: {confidence_score:.2f}%) - {timestamp}")

                # Simpan hasil prediksi ke riwayat
                st.session_state.history.append({
                    "input_image": img_rgb,
                    "predicted_class": hasil,
                    "confidence_score": confidence_score,
                    "timestamp": timestamp
                })
            else:
                prediction_placeholder.info("Belum terdeteksi")

            return img

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return frame.to_ndarray(format="bgr24")

    def on_ended(self):
        self.active = False

# WebRTC untuk streaming video
if st.session_state.run:
    webrtc_ctx = webrtc_streamer(
        key="sign-language",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=RealTimeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# Riwayat prediksi
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.write(f"*{i}.* Prediksi: {item['predicted_class']} dengan Confidence: {item['confidence_score']:.2f}% pada {item['timestamp']}")
        st.image(item['input_image'], caption=f"Gambar {i}", use_container_width=True)

# Menghapus semua riwayat
if clear_history:
    st.session_state.history.clear()
    st.success("✅ Semua riwayat prediksi berhasil dihapus!")
