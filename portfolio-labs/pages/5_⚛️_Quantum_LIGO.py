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
st.markdown("""
**The Concept**: Classical neural networks struggle with specific noise profiles in spacetime strain data. 
We use a **Quantum Kernel** — mapping classical data into a high-dimensional Hilbert space ($2^N$ dimensions) — to "unwind" this noise and detect the "chirp" of black hole mergers.
""")

# --- Sidebar ---
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Input Mode", ["Simulation", "Local Data (data/)", "Upload (.hdf5)"])

# Shared variables
t = None
raw_strain = None
whitened_strain = None
inject = False

if mode == "Simulation":
    st.sidebar.header("Injection Settings")
    inject = st.sidebar.checkbox("Inject Gravitational Wave", value=True)
    snr = st.sidebar.slider("Signal-to-Noise Ratio (SNR)", 0.1, 5.0, 1.5)
    
    # Data Sim
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
                if 'strain' in f:
                    data = f['strain']['Strain'][()]
                    raw_strain = data[:10000] # Slice for performance
                    ts = 1.0/4096
                    t = np.arange(len(raw_strain)) * ts
                    whitened_strain = whiten_data(raw_strain)
                    st.success(f"Loaded {selected_file}")
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
                st.success(f"Loaded {len(raw_strain)} pts.")

st.sidebar.header("Circuit Parameters")
n_layers = st.sidebar.slider("Entanglement Layers", 1, 5, 2)
st.sidebar.caption(f"Qubits: {n_qubits} (Simulated on default.qubit)")


# --- Main Logic ---

if raw_strain is not None:
    # --- Visualization 1: Data ---
    st.subheader("1. Signal Processing Phase")
    col_raw, col_white = st.columns(2)

    with col_raw:
        st.markdown("#### Raw Strain (Time Series)")
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
        st.markdown("#### Whitened Data (Input to VQC)")
        fig_white, ax = plt.subplots(figsize=(8, 3))
        fig_white.patch.set_facecolor('none')
        ax.set_facecolor('#111')
        ax.plot(t, whitened_strain, color='#b026ff', label='Whitened Strain')
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
        ax.tick_params(colors='white')
        st.pyplot(fig_white)


    # --- Quantum Processing ---
    st.subheader("2. Quantum Processing Unit (QPU)")

    col_circ, col_res = st.columns([1, 2])

    # Weights for the circuit
    if 'weights' not in st.session_state:
        st.session_state.weights = np.random.uniform(0, np.pi, (n_layers, n_qubits))

    with col_circ:
        st.markdown("**Variational Ansatz**")
        drawer = qml.draw(quantum_circuit)
        dummy_input = np.random.uniform(0, np.pi, n_qubits)
        st.code(drawer(dummy_input, st.session_state.weights), language="text")
        
        st.markdown("---")
        st.markdown("**Training Control**")
        
        # Training Parameters
        epochs = st.number_input("Epochs", 1, 100, 10)
        lr = st.number_input("Learning Rate", 0.001, 1.0, 0.1)
        
        if st.button("TRAIN VQC (PyTorch)"):
            st.info("Initializing Hybrid Training Loop...")
            
            # 1. Prepare Training Data (Self-Supervised / Auto-Encoder style or Dummy Labels)
            # For this demo, let's assume we want to distinguish signal from noise.
            # We'll create small windows of "Noise" (from quiet parts) and "signal" (injected or high amplitude)
            # Or simpler: Minimize energy of output for noise?
            
            # Let's generate synthetic labels for the current data for demonstration
            # Label = 1 if signal injected (amplitude > threshold), 0 otherwise
            threshold = np.std(whitened_strain) * 2.0
            
            # Creating dataset from the whitened strain
            window_size = n_qubits 
            X_train = []
            Y_train = []
            
            # Sample 20 windows
            for _ in range(20):
                idx = np.random.randint(0, len(whitened_strain) - window_size)
                chunk = whitened_strain[idx:idx+window_size]
                
                # Simple "Ground Truth" logic for the toy training
                # If we injected signal, we know where it is. If real data, we usually don't know without a template.
                # Here we will train it to detect "High Energy" events as a proxy
                energy = np.sum(chunk**2)
                label = 1.0 if energy > np.mean(whitened_strain**2)*5 else -1.0
                
                # Normalize chunk to [0, pi]
                chunk_norm = (chunk - np.min(chunk))/(np.max(chunk) - np.min(chunk) + 1e-9) * np.pi
                X_train.append(chunk_norm)
                Y_train.append(label)
                
            X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
            Y_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32)
            
            # 2. Define QNode Interface
            weight_shapes = {"weights": (n_layers, n_qubits)}
            qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
            
            # 3. Optimizer
            opt = Adam(qlayer.parameters(), lr=lr)
            criterion = MSELoss()
            
            # 4. Loop
            progress = st.progress(0)
            losses = []
            
            for epoch in range(epochs):
                opt.zero_grad()
                predictions = qlayer(X_tensor) # Output is list of expvals
                
                # We want a single scalar output for classification.
                # Let's sum the PauliZ expectations (Z_tot)
                # Output of qlayer is [batch, n_qubits]
                pred_sum = torch.sum(predictions, dim=1)
                
                loss = criterion(pred_sum, Y_tensor)
                loss.backward()
                opt.step()
                
                losses.append(loss.item())
                progress.progress((epoch+1)/epochs)
            
            st.success(f"Training Complete! Final Loss: {losses[-1]:.4f}")
            st.line_chart(losses)
            
            # Update session weights (Extract from torch layer)
            # TorchLayer stores weights in .weights parameter usually
            # But extracting them back to numpy for the pure qnode utils is tricky as shape might flatten
            # We will adhere to use the trained torch layer for inference below? 
            # Or just update the dummy weights visually.
            
            # Sync weights back for "Live Detection" which uses numpy QNode
            with torch.no_grad():
                trained_weights = qlayer.weights.detach().numpy()
                st.session_state.weights = trained_weights

    with col_res:
        st.markdown("**Live Detection (Scanning Window)**")
        
        if st.button("RUN DETECTOR"):
            window_size = 20 # points
            step = 10
            detection_scores = []
            scan_times = []
            
            progress = st.progress(0)
            
            num_windows = (len(whitened_strain) - window_size) // step
            if num_windows > 100: num_windows = 100 # Cap for performance
            
            # Use current weights (Trained if button clicked)
            current_weights = st.session_state.weights
            
            for i in range(num_windows):
                start = i * step
                end = start + window_size
                chunk = whitened_strain[start:end]
                time_point = t[start + window_size//2]
                
                exp_vals = run_classifier(chunk, current_weights)
                detection_scores.append(exp_vals)
                scan_times.append(time_point)
                progress.progress((i+1) / num_windows)
                
            fig_det, ax = plt.subplots(figsize=(10, 3))
            fig_det.patch.set_facecolor('none')
            ax.set_facecolor('#111')
            ax.plot(scan_times, detection_scores, color='#00f3ff', linewidth=2, label='QNode Output <Z>')
            ax.axhline(0.0, color='gray', linestyle='--')
            
            ax.set_title("Quantum Classifier Output", color='white')
            ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
            ax.tick_params(colors='white')
            st.pyplot(fig_det)
else:
    st.warning("No data loaded. Select input mode.")

st.info("Offline Protocol: Place .hdf5 files in `portfolio-web/data/` for auto-detection.")
