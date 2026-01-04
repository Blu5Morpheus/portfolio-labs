import requests
import os

# Target: GW150914 (First Detection) - H1 Detector, 4kHz, 32s
# GPS Time: 1126259462
# We want a file that covers this time.
# Using GWOSC public API pattern.

URL = "https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5"
SAVE_DIR = "data"
SAVE_PATH = os.path.join(SAVE_DIR, "GW150914-H1.hdf5")

def download_event():
    print(f"Targeting Event: GW150914 (The 'Golden' Event)...")
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Downloading from {URL}...")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        
        with open(SAVE_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Done! Saved to {SAVE_PATH}")
        print("This file contains the first ever detected gravitational wave. Robust signal.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_event()
