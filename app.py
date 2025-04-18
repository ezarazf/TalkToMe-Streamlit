import streamlit as st
import torch
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(layout="wide")
st.title("Talk To Me – WebRTC Fixed Auto‑Play")

# Sidebar controls
with st.sidebar:
    st.title("Kontrol")
    start = st.button("▶️ Start")
    stop  = st.button("⏹️ Stop")
    clear = st.button("🧹 Remove History")

# State
if "run" not in st.session_state:    st.session_state.run     = False
if "history" not in st.session_state: st.session_state.history = []

if start: st.session_state.run = True
if stop:  st.session_state.run = False

# Load TorchScript model
@st.cache_resource
def load_model():
    m = torch.jit.load("SL-V1.torchscript", map_location="cpu")
    m.eval()
    return m

model = load_model()
class_labels = ["A","B","C","D","E"]  # ganti sesuai labelmu

# frame callback
def video_frame_callback(frame):
    img_bgr = frame.to_ndarray(format="bgr24")
    if st.session_state.run:
        # preprocessing
        img_r = cv2.resize(img_bgr, (224,224))
        img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(img_rgb)
                  .permute(2,0,1)
                  .unsqueeze(0)
                  .float()/255.0)
        # inference
        with torch.no_grad():
            out = model(tensor)[0]
            probs = torch.nn.functional.softmax(out, dim=0)
            conf, pred = torch.max(probs,0)
            label = class_labels[pred.item()]
            score = conf.item()*100
        # annotate
        cv2.putText(img_bgr, f"{label}: {score:.1f}%", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        # simpan history
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "input_image": cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB),
            "predicted_class": label,
            "confidence_score": score,
            "timestamp": ts
        })
    return img_bgr

# jalankan webrtc dengan autoPlay
webrtc_streamer(
    key="webrtc",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video":True,"audio":False},
    async_processing=True,
    video_html_attrs={
        "autoPlay": True,
        "controls": True,
        "playsInline": True,
        "muted": True
    }
)

# tampilkan riwayat
if st.session_state.history:
    st.subheader("Riwayat Prediksi:")
    for i, h in enumerate(reversed(st.session_state.history),1):
        st.write(f"**{i}.** {h['predicted_class']} ({h['confidence_score']:.2f}%) – {h['timestamp']}")
        st.image(h["input_image"], use_column_width=True)

# clear
if clear:
    st.session_state.history.clear()
    st.success("Riwayat terhapus!")
