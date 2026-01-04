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

if mode == "Simulation":
    st.sidebar.subheader("Simulation Settings")
    inject = st.sidebar.checkbox("Inject Gravitational Wave", value=True)
    snr = st.sidebar.slider("Signal-to-Noise Ratio (SNR)", 0.1, 5.0, 1.5)
    
    t_duration = 1.0
    fs = 200 # Hz
    t, raw_strain, true_signal = generate_noisy_strain(t_duration, fs, inject, snr)
    whitened_strain = whiten_data(raw_strain)

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
                # Basic Discovery
                if 'strain' in f:
                    # GWOSC standard path
                    data = f['strain']['Strain'][()]
                elif 'data' in f:
                    data = f['data'][()] # Generic fallback
                else:
                    # Try to find any dataset
                    def find_dataset(name, node):
                        if isinstance(node, h5py.Dataset) and node.shape[0] > 1000:
                            return node
                        return None
                    # This is complex, let's stick to standard strain
                    data = np.zeros(1000)
                    st.error("Could not find 'strain' group. Check HDF5 structure.")

                # Load a chunk
                slice_size = 10000 
                start_idx = st.sidebar.number_input("Start Index", 0, len(data)-slice_size, 0)
                raw_strain = data[start_idx : start_idx+slice_size]
                
                ts = 1.0/4096
                t = np.arange(len(raw_strain)) * ts
                whitened_strain = whiten_data(raw_strain)
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
                t = np.arange(len(raw_strain)) * ts
                whitened_strain = whiten_data(raw_strain)
            else:
                st.error("Invalid File Structure (Missing 'strain')")


# --- Main Logic ---

if raw_strain is not None:
    
    # --- Tabbed Interface for Workflow ---
    tab_viz, tab_train, tab_infer = st.tabs(["1. Visualize Data", "2. Train Model", "3. Run Detector"])
    
    with tab_viz:
        st.subheader("Signal Processing Phase")
        col_raw, col_white = st.columns(2)

        with col_raw:
            st.markdown("#### Raw Strain")
            fig_raw, ax = plt.subplots(figsize=(8, 3))
            fig_raw.patch.set_facecolor('none')
            ax.set_facecolor('#111')
            ax.plot(t, raw_strain, color='#444', alpha=0.8, label='Strain')
            if inject:
                ax.plot(t, true_signal, color='cyan', alpha=0.6, label='True Injection')
            ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
            ax.tick_params(colors='white')
            st.pyplot(fig_raw)

        with col_white:
            st.markdown("#### Whitened Data (VQC Input)")
            fig_white, ax = plt.subplots(figsize=(8, 3))
            fig_white.patch.set_facecolor('none')
            ax.set_facecolor('#111')
            ax.plot(t, whitened_strain, color='#b026ff', label='Whitened Strain')
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
            epochs = st.number_input("Epochs", 1, 50, 15)
            lr = st.number_input("Learning Rate", 0.001, 0.5, 0.05)
            
            st.markdown("---")
            if st.button("START TRAINING (Hybrid PyTorch Loop)"):
                st.info("Initializing Loop...")
                
                # --- Create Dataset (Windowed) ---
                # We need to make this harder. 
                # Task: Distinguish "Quiet" noise from "Loud" noise (Signal Proxy)
                # But we add label noise so it's not trivial.
                
                window_size = 4 
                X_train = []
                Y_train = []
                
                # Sample 50 windows
                for _ in range(50):
                    idx = np.random.randint(0, len(whitened_strain) - window_size)
                    chunk = whitened_strain[idx:idx+window_size]
                    
                    # Norm
                    chunk_norm = (chunk - np.min(chunk))/(np.max(chunk) - np.min(chunk) + 1e-9) * np.pi
                    
                    # Label Logic: Energy Threshold
                    energy = np.sum(chunk**2)
                    thresh = np.mean(whitened_strain**2) * 2.5
                    
                    label = 1.0 if energy > thresh else -1.0
                    
                    # Add Noise to Data so it's not perfect
                    chunk_noisy = chunk_norm + np.random.normal(0, 0.1, len(chunk))
                    
                    X_train.append(chunk_noisy)
                    Y_train.append(label)
                
                X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
                Y_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32)
                
                # --- Define Model ---
                weight_shapes = {"weights": (n_layers, n_qubits)}
                qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
                
                # Initialize with current session weights if compatible, else random
                # (Ignoring for simplicity to allow architecture change)
                
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
                
                # Update Session State
                with torch.no_grad():
                    trained_weights = qlayer.weights.detach().numpy()
                    st.session_state.weights = trained_weights
                    st.session_state.is_trained = True

    with tab_infer:
        st.subheader("Inference / Detection")
        
        status_color = "green" if st.session_state.is_trained else "red"
        status_text = "TRAINED Model Configured" if st.session_state.is_trained else "UNTRAINED (Random Weights)"
        st.markdown(f":{status_color}[**Status: {status_text}**]")
        
        col_ctrl, col_plot = st.columns([1,3])
        
        with col_ctrl:
            det_thresh = st.slider("Detection Threshold", -2.0, 2.0, 0.5)
            window_size = 20
            step = 10
            
            run_btn = st.button("SCAN DATA NOW")
            
        with col_plot:
            if run_btn:
                # Use current weights
                current_weights = st.session_state.weights
                
                # Must match trained layer shape logic (if trained)
                # If weights shape mismatch (due to changing layer count in train tab), warn user
                if current_weights.shape != (st.session_state.weights.shape[0], 4):
                     st.warning("Weight shape mismatch! Did you change layers without retraining? Using random.")
                     current_weights = np.random.uniform(0, np.pi, (n_layers, 4))
                
                scan_scores = []
                scan_times = []
                
                num_windows = (len(whitened_strain) - window_size) // step
                if num_windows > 200: num_windows = 200 # Cap
                
                progress_infer = st.progress(0)
                
                for i in range(num_windows):
                    start = i * step
                    end = start + window_size
                    chunk = whitened_strain[start:end]
                    t_point = t[start + window_size//2]
                    
                    val = run_classifier(chunk, current_weights)
                    scan_scores.append(val)
                    scan_times.append(t_point)
                    
                    progress_infer.progress((i+1)/num_windows)
                
                # Plot
                fig_det, ax = plt.subplots(figsize=(10, 3))
                fig_det.patch.set_facecolor('none')
                ax.set_facecolor('#111')
                
                ax.plot(scan_times, scan_scores, color='#00f3ff', linewidth=2, label='Quantum Activity <Z>')
                ax.axhline(det_thresh, color='red', linestyle='--', label='Threshold')
                
                # Highlight Detections
                scan_scores_np = np.array(scan_scores)
                # Determine "Detected" regions
                # Simple boolean mask
                
                ax.set_title(f"Inference Result (Weights: {'Trained' if st.session_state.is_trained else 'Random'})", color='white')
                ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
                ax.tick_params(colors='white')
                st.pyplot(fig_det)

else:
    st.info("Awaiting Data...")
