import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import random
import os

matplotlib.use('Agg')

st.set_page_config(page_title="Laika AI", page_icon="🐶", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("🐶 Laika: SETI Scout")
st.markdown("### Green Bank Telescope Event Classifier")
st.markdown("Laika is a CNN trained to sniff out **Technosignatures** (narrowband drifting signals) in radio spectrograms.")

# --- Simulation ---
def generate_spectrogram(has_signal=False):
    # Dimensions: Time x Frequency
    H, W = 64, 64
    data = np.random.exponential(1.0, (H, W)) # Background Noise
    
    if has_signal:
        # Doppler Drift Signal
        start_f = np.random.randint(10, 54)
        drift = np.random.choice([-1, 0, 1]) * np.random.randint(1, 3)
        
        for t in range(H):
            f = int(start_f + (drift * (t/H) * 5))
            f = np.clip(f, 0, W-1)
            # Signal intensity
            data[t, f] += np.random.uniform(5.0, 10.0)
            
    return data

# --- AI Model ---
class LaikaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dim
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- UI ---
col_sim, col_laika = st.columns(2)

with col_sim:
    st.subheader("📡 Green Bank Data Stream")
    if 'laika_model' not in st.session_state:
        st.info("Waking up Laika...")
        st.session_state.laika_model = LaikaNet()
        st.session_state.laika_trained = False
        
    if st.button("Train Laika (Simulated Data)"):
        model = st.session_state.laika_model
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()
        
        prog = st.progress(0)
        
        for step in range(100):
            # Batch
            batch_img = []
            batch_lbl = []
            for _ in range(16):
                is_sig = random.random() > 0.5
                img = generate_spectrogram(is_sig)
                batch_img.append(img)
                batch_lbl.append(1.0 if is_sig else 0.0)
            
            X = torch.tensor(np.array(batch_img), dtype=torch.float32)
            Y = torch.tensor(np.array(batch_lbl), dtype=torch.float32).unsqueeze(1)
            
            pred = model(X)
            loss = loss_fn(pred, Y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if step % 10 == 0: prog.progress((step+1)/100)
            
        st.session_state.laika_trained = True
        st.success("Laika is ready to hunt! 🦴")

with col_laika:
    st.subheader("🐶 Laika's Findings")
    
    data_source = st.radio("Data Source", ["Simulated Drift", "Real Voyager 1 (GBT)"])
    
    if st.button("Analyze Signal ☄️"):
        if st.session_state.get('laika_trained', False):
            
            if data_source == "Real Voyager 1 (GBT)":
                # Load Real Data
                # Path
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                npy_path = os.path.join(base_dir, "data", "voyager_waterfall.npy")
                
                if os.path.exists(npy_path):
                    full_spec = np.load(npy_path)
                    # It's big. Let's take a chunk or resize.
                    # Resize to 64x64 for the model?
                    # Or just show the big one and crop a 64x64 patch for classification.
                    
                    st.success("Loaded Actual Voyager 1 Signal!")
                    
                    # Show Full
                    fig_real, axr = plt.subplots(figsize=(8,4))
                    axr.imshow(full_spec, aspect='auto', cmap='viridis', vmin=np.percentile(full_spec, 5), vmax=np.percentile(full_spec, 99))
                    axr.set_title("Full Recording (Voyager Carrier)")
                    st.pyplot(fig_real)
                    plt.close(fig_real)
                    
                    # Crop center for AI
                    center_f = full_spec.shape[1] // 2
                    center_t = full_spec.shape[0] // 2
                    
                    # Downsample/Crop to 64x64
                    # This is just a demo, so we'll just take a slice where the signal is likely (center)
                    # Assuming signal is centered by FFT Shift.
                    sample = full_spec[center_t-32:center_t+32, center_f-32:center_f+32]
                    
                    # Normalize for AI
                    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
                    
                else:
                    st.error("Voyager Data not found. Run 'tools/fetch_voyager.py' first!")
                    sample = generate_spectrogram(True) # Fallback
            else:
                # Sim
                is_real = random.random() > 0.7 
                sample = generate_spectrogram(is_real)
            
            # Show Analysis Target
            fig, ax = plt.subplots()
            ax.imshow(sample, aspect='auto', cmap='viridis', origin='lower')
            ax.set_title("AI Input Window")
            col_sim.pyplot(fig) 
            plt.close(fig)
            
            # Predict
            model = st.session_state.laika_model
            inp = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = model(inp).item()
            
            st.metric("Signal Probability", f"{prob*100:.1f}%")
            
            if prob > 0.8:
                st.success("BARK! Found a Technosignature!")
                st.balloons()
            elif prob < 0.2:
                st.info("Quiet. Just noise.")
            else:
                st.warning("Growling... Unsure.")
                
        else:
            st.error("Train Laika first!")
