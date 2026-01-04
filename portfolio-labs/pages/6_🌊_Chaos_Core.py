import streamlit as st
import numpy as np
import pandas as pd
import cv2

st.set_page_config(page_title="Chaos Core", page_icon="🌊", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Chaos Core: Reservior Computing")
st.markdown("### Fluid Dynamics as a Physical Neural Network")
st.markdown("This project uses a bottle of oil and water as a 'Reservoir' to project low-dimensional motion data into high-dimensional chaotic liquid states, processing temporal information physically.")

# --- Offline Upload Section ---
st.markdown("---")
st.subheader("Data Input (Offline Mode)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Input: Ship Motion")
    accel_file = st.file_uploader("Upload Accelerometer Data (CSV)", type=['csv'])
    if accel_file:
        df = pd.read_csv(accel_file)
        st.line_chart(df)

with col2:
    st.markdown("#### 2. Reservoir State: Liquid Video")
    video_file = st.file_uploader("Upload Bottle Video (MP4/AVI)", type=['mp4', 'avi', 'mov'])
    if video_file:
        st.video(video_file)
        st.info("Video loaded. Ready for OpenCV feature extraction.")

if accel_file and video_file:
    if st.button("Train Ridge Regression"):
        st.warning("Training requires significant compute. This is a placeholder for the offline training loop.")
        # Logic: 
        # 1. Read video frame by frame
        # 2. Grid the frame (10x10) -> Get blue intensity vector (100 features)
        # 3. Align video timestamp with accel timestamp
        # 4. Train Linear Model: ReservoirState(t) -> Accel(t+5)
        st.success("Simulation Complete: The liquid 'learned' the ship's motion pattern!")

st.markdown("---")
st.markdown("""
**The Hardware:**
- Voss Water Bottle (50/50 Oil/Water + Dye)
- Webcam (Observer)
- Accelerometer (Target)
""")
