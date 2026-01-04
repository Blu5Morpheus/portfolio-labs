import requests
import os

# GW190412 (Black Hole Merger) - H1 Detector, 4kHz, 32s
# This is a robust event for testing.
URL = "https://www.gw-openscience.org/eventapi/html/GWTC-2/GW190412/v3/H-H1_GWOSC_4KHZ_R1-1239082247-32.hdf5"
SAVE_PATH = "data/GW190412-H1.hdf5"

def download_file():
    print(f"Attempting to download GW190412 to {SAVE_PATH}...")
    try:
        if not os.path.exists("data"):
            os.makedirs("data")

        response = requests.get(URL, stream=True)
        response.raise_for_status()
        
        with open(SAVE_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download Complete! You are ready to hunt black holes.")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Please manually download an O3a file from gwosc.org and place it in the 'data/' folder.")

if __name__ == "__main__":
    download_file()
