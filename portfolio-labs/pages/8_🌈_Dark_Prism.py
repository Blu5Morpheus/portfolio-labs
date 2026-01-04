import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dark Prism", page_icon="🌈", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌈 Dark Prism: Spectrometer")
st.markdown("### Chemical Analysis via Diffraction")
st.markdown("Using a Cereal Box + CD-R diffraction grating. Upload your spectrum photo to extract the fingerprint.")

st.markdown("---")
st.subheader("Spectral Analysis")

spec_img = st.file_uploader("Upload Spectrum Photo", type=['jpg', 'png', 'jpeg'])

if spec_img:
    image = Image.open(spec_img)
    st.image(image, caption="Raw Spectrum", use_container_width=True)
    
    # Conversion to Array
    img_array = np.array(image)
    
    # Check if RGB or Grayscale
    if len(img_array.shape) == 3:
        # Convert to Grayscale for intensity (simple average)
        # or separate channels if we want to be fancy
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    if st.button("Extract Spectrum"):
        st.write("Processing Data...")
        
        # 1. Spatial Integration
        # We assume the spectrum is horizontal. We sum columns to get intensity profile.
        # If the spectrum is vertical, user might need to rotate image or we check orientation.
        # Simple heuristic: longer dimension is the spectral axis?
        # Let's assume standard landscape photo where spectrum is Left-Right.
        
        # Sum vertically (collapse Y)
        intensity_profile = np.sum(gray, axis=0)
        
        # Normalize
        intensity_profile = (intensity_profile - np.min(intensity_profile)) / (np.max(intensity_profile) - np.min(intensity_profile))
        
        x_pixels = np.arange(len(intensity_profile))
        
        # Plot
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**1D Intensity Profile**")
            fig, ax = plt.subplots(figsize=(5,3))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('#111')
            ax.plot(x_pixels, intensity_profile, color='white', linewidth=1)
            ax.set_xlabel("Pixel Position (Blue -> Red)", color='white')
            ax.set_ylabel("Normalized Intensity", color='white')
            ax.tick_params(colors='white')
            st.pyplot(fig)
            
        with col2:
            st.markdown("**Auto-Peak Detection**")
            # Simple thresholding
            peaks = np.where(intensity_profile > 0.8)[0]
            if len(peaks) > 0:
                st.metric("Dominant Line Position", f"Pixel {int(np.mean(peaks))}")
            else:
                st.write("No strong peaks detected.")
            
            st.info("To Calibrate: Map Pixel 0->N to Wavelengths (nm) using known sources like a fluorescent bulb.")

st.markdown("---")
st.markdown("""
**Hardware Protocol:**
1.  **Capture**: Dark room, flashlight through liquid, look through CD slit.
2.  **Upload**: Take photo, crop to just the rainbow bar, upload here.
3.  **Result**: The graph above shows the absorption/emission lines.
""")
