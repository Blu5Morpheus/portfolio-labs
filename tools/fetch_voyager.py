import requests
import gzip
import numpy as np
import os
import matplotlib.pyplot as plt

# URL for Voyager 1 recording (SDR-Radio / Daniel Estevez)
# ~50MB compressed, contains the Carrier.
URL = "https://www.sdr-radio.com/dl/voyager1-LHCP-2015-12-30-HSR.raw.gz"
SAVE_DIR = "data"
RAW_PATH = os.path.join(SAVE_DIR, "voyager1.raw.gz")
NPY_PATH = os.path.join(SAVE_DIR, "voyager_waterfall.npy")

def fetch_and_process():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Downloading Voyager 1 Data from {URL}...")
    try:
        # Download
        if not os.path.exists(RAW_PATH):
            with requests.get(URL, stream=True) as r:
                r.raise_for_status()
                with open(RAW_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download Complete.")
        else:
            print("Using cached raw file.")
            
        # Process
        print("Decompressing & Processing IQ Data...")
        with gzip.open(RAW_PATH, 'rb') as f:
            # Read first chunk to detect format
            header_bytes = f.read(1024 * 64)
            
            # Try float32 (complex64)
            data_test = np.frombuffer(header_bytes, dtype=np.complex64)
            data_max = np.max(np.abs(data_test))
            
            # If max is huge (e.g. > 100), it's probably int16, not normalized float
            # But SDR-Radio usually uses int16 or uint8.
            # Let's try int16
            data_int16 = np.frombuffer(header_bytes, dtype=np.int16)
            # Interleaved I/Q
            
            print(f"Format Check: Float Max={data_max:.2f}")
            
            # It's likely int16 (SDRSharp default) or Int8.
            # Given the source (HSR = High Sampling Rate?), likely int16.
            # Let's assume Int16 interleaved.
            dtype_use = np.int16
            
            f.seek(0)
            raw_data = f.read()
            # Interleaved I Q
            data_int = np.frombuffer(raw_data, dtype=dtype_use)
            # Combine to complex
            # I = even, Q = odd
            I = data_int[0::2].astype(np.float32)
            Q = data_int[1::2].astype(np.float32)
            complex_data = I + 1j*Q
            
            print(f"loaded {len(complex_data)} samples.")
            
            # Compute Spectrogram
            # 1M sample chunk
            if len(complex_data) > 4000000:
                complex_data = complex_data[:4000000]
                
            # FFT
            fft_size = 1024
            num_rows = len(complex_data) // fft_size
            
            waterfall = np.zeros((num_rows, fft_size))
            
            for i in range(num_rows):
                chunk = complex_data[i*fft_size : (i+1)*fft_size]
                spec = np.abs(np.fft.fftshift(np.fft.fft(chunk)))
                waterfall[i, :] = spec
            
            # Log scale
            waterfall = np.log10(waterfall + 1e-9)
            
            # Save
            np.save(NPY_PATH, waterfall)
            print(f"Saved Spectrogram to {NPY_PATH}")
            print("Run the Laika App to see the Voyager signal!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_process()
