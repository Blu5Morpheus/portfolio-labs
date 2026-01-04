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

def run_classifier(data_slice, weights, fs=4096):
    """
    Run the VQC on a slice of data using Spectral Features.
    Mapping: 4 Qubits representing 4 Frequency Bands.
    """
    # Feature Extraction: Band Power
    # We want to capture the "Chirp" energy distribution.
    # Bands: [20-80Hz], [80-150Hz], [150-300Hz], [300-600Hz] (Approx chirp sweep)
    
    # 1. Compute FFT
    if len(data_slice) < 4: return 0 # Too small
    
    freqs = np.fft.rfftfreq(len(data_slice), d=1/fs)
    mag = np.abs(np.fft.rfft(data_slice))
    
    # Define bands (Indices)
    bands = [
        (20, 80),
        (80, 150),
        (150, 300),
        (300, 600)
    ]
    
    features = []
    for low, high in bands:
        # Find indices
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) > 0:
            avg_power = np.mean(mag[idx])
        else:
            avg_power = 0.0
        features.append(avg_power)
        
    # Resize to n_qubits if needed (though we defined 4 bands for 4 qubits)
    features = np.array(features[:n_qubits])
    
    # Normalize features to [0, pi] for Angle Embedding
    # Log scale is often better for power spectra
    features = np.log1p(features) 
    
    # Min-Max Scaling to [0, pi]
    if np.max(features) - np.min(features) > 1e-9:
        norm_features = (features - np.min(features)) / (np.max(features) - np.min(features)) * np.pi
    else:
        norm_features = np.zeros_like(features)
    
    # Run Circuit
    expectations = quantum_circuit(norm_features, weights)
    
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

def whiten_data(data, fs=4096):
    """
    Whitens data using a bandpass filter tuned to the sampling rate.
    LIGO Sensitive Band: ~20Hz to ~2000Hz.
    """
    nyquist = fs / 2
    
    # Desired band: 30Hz - 1000Hz
    # Safe limits for generic bandpass
    low = 30.0 / nyquist
    high = 1000.0 / nyquist
    
    # Safety checks
    if low <= 0: low = 0.01
    if high >= 1: high = 0.99
    
    sos = signal.butter(4, [low, high], btype='bandpass', output='sos')
    whitened = signal.sosfilt(sos, data)
    return whitened
