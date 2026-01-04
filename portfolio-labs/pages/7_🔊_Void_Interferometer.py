import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

st.set_page_config(page_title="Void Interferometer", page_icon="🔊", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔊 Void Interferometer")
st.markdown("### Signal Analysis & Denoising")
st.markdown("Upload the audio from your solar panel setup. We will look for signal spikes buried in the ship's engine noise.")

# --- Upload ---
st.markdown("---")
st.subheader("Signal Processing Workbench")

audio_file = st.file_uploader("Upload Solar Panel Audio (WAV)", type=['wav'])

if audio_file:
    st.audio(audio_file)
    
    # Read Data
    try:
        sr, data = wavfile.read(audio_file)
        
        # Mono conversion
        if len(data.shape) > 1:
            data = data[:, 0]
            
        # Time Axis
        N = len(data)
        T = 1.0 / sr
        t = np.linspace(0.0, N*T, N, endpoint=False)
        
        # --- Visualization 1: Time Domain ---
        st.subheader("1. Time Domain (Raw Voltage)")
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('#111')
        ax.plot(t, data, color='#00f3ff', alpha=0.8, linewidth=0.5)
        ax.set_xlabel("Time (s)", color='gray')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        
        # --- Visualization 2: Frequency Domain (FFT) ---
        st.subheader("2. Frequency Spectrum (FFT)")
        
        # Compute FFT
        yf = fft(data)
        xf = fftfreq(N, T)[:N//2]
        power = 2.0/N * np.abs(yf[0:N//2])
        
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        fig2.patch.set_facecolor('none')
        ax2.set_facecolor('#111')
        ax2.plot(xf, power, color='#b026ff', linewidth=1)
        ax2.set_xlabel("Frequency (Hz)", color='white')
        ax2.set_ylabel("Power", color='white')
        ax2.grid(color='#333', linestyle='--')
        ax2.tick_params(colors='white')
        
        # Highlight Engine Hum vs Signal
        st.caption("Look for sharp peaks (Signal) vs broad hums (Engine Noise).")
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")

st.markdown("---") 
st.markdown("""
**The Setup:**
- **Source**: Laser hitting Solar Panel.
- **Noise**: Ship engines (usually low freq < 200Hz).
- **Signal**: Taps/Vibrations (Sharp transients / High freq).
""")
