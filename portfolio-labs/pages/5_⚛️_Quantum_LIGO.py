import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from utils.quantum_utils import generate_noisy_strain, whiten_data, run_classifier, quantum_circuit, n_qubits
import h5py
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import os

st.set_page_config(page_title="Quantum LIGO Detector", page_icon="⚛️", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚛️ Quantum Gravitational Wave Detector")
st.markdown("### Variational Quantum Classifier (VQC) for LIGO Data")
st.markdown("*Architecture: Spectral Feature Map (4 Bands) -> Angle Embedding -> Basic Entangler -> Z-Measurement*")

# --- Initialize Weights in Session ---
if 'weights' not in st.session_state:
    # Random initialization (Untrained)
    st.session_state.weights = np.random.uniform(0, np.pi, (2, 4)) # 2 layers, 4 qubits
    st.session_state.is_trained = False

# --- Data Source Section ---
st.sidebar.header("1. Data Source")
mode = st.sidebar.radio("Input Mode", ["Simulation", "Local Data (data/)", "Upload (.hdf5)"])

# Shared variables
t = None
raw_strain = None
whitened_strain = None
inject = False
fs_val = 4096.0 # Default

if mode == "Simulation":
    st.sidebar.subheader("Simulation Settings")
    inject = st.sidebar.checkbox("Inject Gravitational Wave", value=True)
    snr = st.sidebar.slider("Signal-to-Noise Ratio (SNR)", 0.1, 5.0, 1.5)
    
    t_duration = 1.0
    fs_val = 4096.0 # Match real data rate for better spectral sim
    t, raw_strain, true_signal = generate_noisy_strain(t_duration, int(fs_val), inject, snr)
    whitened_strain = whiten_data(raw_strain, fs=fs_val)

elif mode == "Local Data (data/)":
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.sidebar.warning(f"Created {data_dir}. Place HDF5 files here.")
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5') or f.endswith('.h5')]
    selected_file = st.sidebar.selectbox("Select File", files)
    
    if selected_file:
        file_path = os.path.join(data_dir, selected_file)
        try:
            with h5py.File(file_path, "r") as f:
                # GWOSC standard path
                if 'strain' in f:
                    data = f['strain']['Strain'][()]
                elif 'data' in f:
                    data = f['data'][()] 
                else:
                    data = np.zeros(1000)
                    st.error("Could not find 'strain' group.")

                slice_size = 16384 # 4 seconds at 4kHz
                start_idx = st.sidebar.number_input("Start Index", 0, len(data)-slice_size, 0)
                raw_strain = data[start_idx : start_idx+slice_size]
                
                ts = 1.0/4096
                fs_val = 4096.0
                t = np.arange(len(raw_strain)) * ts
                whitened_strain = whiten_data(raw_strain, fs=fs_val)
                st.sidebar.success(f"Loaded {slice_size} pts from {selected_file}")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.sidebar.info("No HDF5 files found in data/")

else: # Upload
    uploaded_file = st.sidebar.file_uploader("Upload O3 Event File (.hdf5)", type=['hdf5', 'h5'])
    if uploaded_file:
        with open("temp_ligo.hdf5", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with h5py.File("temp_ligo.hdf5", "r") as f:
            if 'strain' in f:
                data = f['strain']['Strain'][()]
                raw_strain = data[:10000]
                ts = 1.0/4096
                fs_val = 4096.0
                t = np.arange(len(raw_strain)) * ts
                whitened_strain = whiten_data(raw_strain, fs=fs_val)
    
# --- Helper for Spectral Extraction for Training ---
def get_spectral_features(chunk, fs):
     # Same logic as utils but for batch processing
     freqs = np.fft.rfftfreq(len(chunk), d=1/fs)
     mag = np.abs(np.fft.rfft(chunk))
     bands = [(20, 80), (80, 150), (150, 300), (300, 600)]
     features = []
     for low, high in bands:
        idx = np.where((freqs >= low) & (freqs < high))[0]
        val = np.mean(mag[idx]) if len(idx) > 0 else 0.0
        features.append(val)
     features = np.array(features)
     features = np.log1p(features)
     return (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-9) * np.pi

# --- Main Logic ---

if raw_strain is not None:
    
    tab_viz, tab_train, tab_infer = st.tabs(["1. Visualize Data", "2. Train Model", "3. Run Detector"])
    
    with tab_viz:
        st.subheader("Signal Analysis")
        col_raw, col_white = st.columns(2)

        with col_raw:
            st.markdown("#### Raw Strain")
            fig_raw, ax = plt.subplots(figsize=(8, 3))
            fig_raw.patch.set_facecolor('none')
            ax.set_facecolor('#111')
            ax.plot(t, raw_strain, color='#444', alpha=0.8, label='Strain')
            ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
            ax.tick_params(colors='white')
            st.pyplot(fig_raw)

        with col_white:
            st.markdown("#### Whitened Data")
            fig_white, ax = plt.subplots(figsize=(8, 3))
            fig_white.patch.set_facecolor('none')
            ax.set_facecolor('#111')
            ax.plot(t, whitened_strain, color='#b026ff', label='Whitened')
            ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
            ax.tick_params(colors='white')
            st.pyplot(fig_white)
            
    with tab_train:
        st.subheader("Quantum Model Training")
        
        col_circ, col_param = st.columns([1, 2])
        
        with col_circ:
            st.markdown("**Ansatz**")
            n_layers = st.number_input("Layers", 1, 5, 2)
            n_qubits = 4
            dummy_drawer_weights = np.random.uniform(0, np.pi, (n_layers, n_qubits))
            drawer = qml.draw(quantum_circuit)
            st.code(drawer(np.zeros(n_qubits), dummy_drawer_weights), language="text")

        with col_param:
            st.markdown("**Hyperparameters**")
            epochs = st.number_input("Epochs", 1, 50, 20)
            lr = st.number_input("Learning Rate", 0.001, 0.5, 0.05)
            
            if st.button("START TRAINING (Spectral Mode)"):
                st.info("Generating Training Dataset (Spectral Features)...")
                
                # Training Window: 0.1s = ~400 samples at 4kHz
                t_win = 0.1 
                win_pts = int(t_win * fs_val)
                
                X_train = []
                Y_train = []
                
                # Create "Signal" examples (High energy in chirp bands)
                # We simulate chirp chunks specifically for training
                # This ensures the model learns what a chirp LOOKS like spectrally
                for _ in range(25):
                    # Sim Chirp
                    _, c = generate_noisy_strain(t_win, int(fs_val), inject_signal=True, snr=2.0)
                    feat = get_spectral_features(whiten_data(c, fs_val), fs_val)
                    X_train.append(feat)
                    Y_train.append(1.0) # Signal
                    
                # Create "Noise" examples (Raw noise)
                for _ in range(25):
                    _, n = generate_noisy_strain(t_win, int(fs_val), inject_signal=False)
                    feat = get_spectral_features(whiten_data(n, fs_val), fs_val)
                    X_train.append(feat)
                    Y_train.append(-1.0) # Noise
                
                # Shuffle
                indices = np.arange(50)
                np.random.shuffle(indices)
                X_train = np.array(X_train)[indices]
                Y_train = np.array(Y_train)[indices]
                
                X_tensor = torch.tensor(X_train, dtype=torch.float32)
                Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
                
                # Define Model
                weight_shapes = {"weights": (n_layers, n_qubits)}
                qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
                opt = Adam(qlayer.parameters(), lr=lr)
                criterion = MSELoss()
                
                progress = st.progress(0)
                losses = []
                
                for epoch in range(epochs):
                    opt.zero_grad()
                    predictions = qlayer(X_tensor) 
                    pred_sum = torch.sum(predictions, dim=1)
                    loss = criterion(pred_sum, Y_tensor)
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                    progress.progress((epoch+1)/epochs)
                
                st.success(f"Training Complete! Final Loss: {losses[-1]:.4f}")
                st.line_chart(losses)
                
                with torch.no_grad():
                    st.session_state.weights = qlayer.weights.detach().numpy()
                    st.session_state.is_trained = True

    with tab_infer:
        st.subheader("Inference / Detection")
        
        status_color = "green" if st.session_state.is_trained else "red"
        status_text = "TRAINED" if st.session_state.is_trained else "UNTRAINED"
        st.markdown(f"**Model Status:** :{status_color}[{status_text}]")
        
        det_thresh = st.slider("Detection Threshold (Qubit Z-Exp)", -1.0, 1.0, 0.0)
        
        col_ctrl, col_plot = st.columns([1,3])
        with col_ctrl:
            scan_btn = st.button("SCAN DATA NOW")
            
        with col_plot:
            if scan_btn:
                st.write("Scanning...")
                current_weights = st.session_state.weights
                
                # Window: 0.1s
                t_win = 0.1
                win_pts = int(t_win * fs_val)
                step = win_pts // 2 # 50% overlap
                
                scan_scores = []
                scan_times = []
                
                # Limit scan to ~200 windows for perf
                total_samples = len(whitened_strain)
                max_windows = 200
                
                # If too long, just scan a portion
                if total_samples // step > max_windows:
                    st.caption(f"Data too long. Scanning first {max_windows} windows...")
                    iterator = range(max_windows)
                else:
                    iterator = range((total_samples - win_pts) // step)
                
                progress_infer = st.progress(0)
                
                for i in iterator:
                    start = i * step
                    end = start + win_pts
                    chunk = whitened_strain[start:end]
                    t_point = t[start + win_pts//2]
                    
                    # Pass fs to utils
                    val = run_classifier(chunk, current_weights, fs=fs_val)
                    scan_scores.append(val)
                    scan_times.append(t_point)
                    
                    progress_infer.progress((i+1)/len(iterator))
                
                fig_det, ax = plt.subplots(figsize=(10, 3))
                fig_det.patch.set_facecolor('none')
                ax.set_facecolor('#111')
                ax.plot(scan_times, scan_scores, color='#00f3ff', linewidth=2, label='Quantum Output')
                ax.axhline(det_thresh, color='red', linestyle='--', label='Threshold')
                ax.set_title("Detection Trace", color='white')
                ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
                ax.tick_params(colors='white')
                st.pyplot(fig_det)
                
else:
    st.info("Awaiting Data...")
