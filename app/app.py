import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import tempfile
import os
import time

# --- 1. DEFINE THE MODEL ARCHITECTURE ---
class AdvancedLSTM(nn.Module):
    def __init__(self):
        super(AdvancedLSTM, self).__init__()
        # 2-Layer Bi-Directional LSTM
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.4
        )
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 64), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :]
        return self.fc(final_state)

# --- 2. LOAD MODELS (CACHED FOR SPEED) ---
@st.cache_resource
def load_system():
    # Use CPU for the demo to ensure compatibility
    device = torch.device("cpu")
    
    # A. Load ResNet50 ("The Eyes")
    resnet = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    
    # B. Load LSTM ("The Brain")
    model = AdvancedLSTM()
    
    # Check if file exists
    model_path = 'smarthome_phase2_augmented.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully!")
    else:
        st.error(f"❌ CRITICAL ERROR: Could not find '{model_path}'. Please move it to this folder.")
        st.stop()
    
    model.eval()
    return feature_extractor, model, device

# --- 3. VIDEO PROCESSING PIPELINE ---
def analyze_video(video_path, feature_extractor, model):
    SEQ_LEN = 16
    IMG_SIZE = 224
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Smart Frame Sampling
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1: total_frames = 1
    indices = np.linspace(0, total_frames-1, SEQ_LEN).astype(int)
    
    curr = 0
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if curr in indices:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_frame = preprocess(rgb_frame)
            frames.append(t_frame)
            if len(frames) >= SEQ_LEN: break
        curr += 1
    cap.release()
    
    while len(frames) < SEQ_LEN:
        frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        
    input_tensor = torch.stack(frames).unsqueeze(0)
    
    # --- INFERENCE ---
    with torch.no_grad():
        features = feature_extractor(input_tensor.view(-1, 3, IMG_SIZE, IMG_SIZE))
        features = features.view(1, SEQ_LEN, -1)
        prediction_score = model(features).item()
        
    return prediction_score

# --- 4. THE USER INTERFACE ---
st.set_page_config(page_title="Smart House Anomaly Detection", layout="centered")

# Professional Title (No heavy emojis)
st.markdown("<h1 style='text-align: center;'>Smart House Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Automated Monitoring System for Safety & Security</b></p>", unsafe_allow_html=True)
st.divider()

# Load the brain
feature_extractor, model, device = load_system()

# Sidebar Controls
st.sidebar.header("Control Panel")

# --- PRESENTATION MODE: UPLOAD ONLY ---
# We force the mode to "Upload Custom Video" 
mode = "Upload Custom Video" 

video_path = None
THRESHOLD = 0.6774

if mode == "Upload Custom Video":
    st.sidebar.info("Upload surveillance footage (MP4) to begin analysis.")
    uploaded_file = st.sidebar.file_uploader("Browse Files...", type=["mp4"])
    
    if uploaded_file is not None:
        # Save uploaded file to temp directory so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

# Main Display Logic
if video_path:
    st.video(video_path)
    
    if st.button("Analyze Footage", type="primary", use_container_width=True):
        with st.spinner("Processing video stream..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            score = analyze_video(video_path, feature_extractor, model)
            
        # Results Display
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Confidence", f"{score:.4f}")
            st.caption(f"Threshold: {THRESHOLD}")
            
        with col2:
            if score >= THRESHOLD:
                st.error("ANOMALY DETECTED")
                st.write("**Assessment:** Irregular motion detected (Fall / Accident / Intrusion).")
            else:
                st.success("NORMAL ACTIVITY")
                st.write("**Assessment:** Routine activity. No threats detected.")
                
        # Visual Bar
        st.write("---")
        st.write("### Threat Assessment Scale")
        st.progress(int(score * 100))
        if score >= THRESHOLD:
             st.caption("High confidence of anomalous event.")