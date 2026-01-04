import numpy as np
import pennylane as qml
from scipy import signal
from scipy.signal import windows

# --- Quantum Circuit ---
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    """
    Variational Quantum Circuit for classification.
    """
    # Encoding Classical Data (The "Feature Map")
    # We expect inputs to be scaled to [0, 2pi] usually, or normalized
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # Variational Layer (The "Brain")
    # Weights shape: (n_layers, n_qubits)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # Measurement
    # Returning expectation of PauliZ on each wire
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

def run_classifier(data_slice, weights):
    """
    Run the VQC on a slice of data.
    """
    # Pre-process: Resize data to match n_qubits (dimensionality reduction)
    # Simple strategy: Downsample or average pooling to get 4 features
    if len(data_slice) > n_qubits:
        # Resample to 4 points
        pooled_data = signal.resample(data_slice, n_qubits)
    else:
        pooled_data = np.pad(data_slice, (0, n_qubits - len(data_slice)))
    
    # Normalize to [0, pi] for AngleEmbedding
    # Map min/max to 0, pi
    norm_data = (pooled_data - np.min(pooled_data)) / (np.max(pooled_data) - np.min(pooled_data) + 1e-9) * np.pi
    
    # Run Circuit
    expectations = quantum_circuit(norm_data, weights)
    
    # Return mean expectation
    return np.mean(expectations)

# --- Data Simulation ---
def generate_chirp_signal(t_duration, fs, t_event=0.5, amplitude=1.0):
    """
    Generates a synthetic 'chirp' gravitational wave signal.
    """
    t = np.linspace(0, t_duration, int(t_duration * fs))
    # Chirp: Frequency increases with time
    # localized around t_event
    
    # Create a window
    # FIX: Use windows.gaussian for recent SciPy versions
    window = windows.gaussian(len(t), std=int(fs*0.05))
    
    # Shift window to t_event
    shift = int((t_event - t_duration/2) * fs)
    window = np.roll(window, shift)
    
    # Chirp formulation
    chirp = signal.chirp(t, f0=20, t1=t_duration, f1=100)
    
    gw_signal = chirp * window * amplitude
    return t, gw_signal

def generate_noisy_strain(t_duration=1.0, fs=1024, inject_signal=False, snr=1.0):
    """
    Generates noisy strain data, optionally with a signal.
    """
    t, pure_signal = generate_chirp_signal(t_duration, fs, t_event=0.5)
    
    # Coloured noise simulation (approximating LIGO PSD roughly with 1/f)
    noise = np.random.normal(0, 1.0, len(t))
    
    if inject_signal:
        # Scale signal by SNR relative to noise std
        scaled_signal = pure_signal * snr
        data = noise + scaled_signal
    else:
        data = noise
        scaled_signal = np.zeros_like(t)
        
    return t, data, scaled_signal

def whiten_data(data):
    """
    Toy whitening: Normalize spectral density.
    """
    sos = signal.butter(4, [0.05, 0.45], btype='bandpass', output='sos')
    whitened = signal.sosfilt(sos, data)
    return whitened
