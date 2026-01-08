import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# Add project root to path to import sdk
# pages -> portfolio-labs -> portfolio-web (root)
current_dir = os.path.dirname(os.path.abspath(__file__))
labs_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(labs_dir) # Go up one more to portfolio-web
sys.path.append(root_dir)

from sdk.quantum_lock import QuantumLock

matplotlib.use('Agg')

st.set_page_config(page_title="Quantum Lock", page_icon="🔐", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("🔐 Quantum Lock SDK")
st.markdown("### Post-Quantum Encryption (Lattice LWE)")
st.markdown("An implementation of **Learning With Errors**, the 'Untangling' technique used to secure data against Quantum Computers.")

col_viz, col_sdk = st.columns([2, 1])

# Initialize SDK
if 'ql' not in st.session_state:
    st.session_state.ql = QuantumLock(n=64, q=256) # Low dims for fast demo demo
    st.session_state.ql.keygen()

with col_sdk:
    st.subheader("📦 Download SDK")
    st.markdown("Get the `quantum_lock.py` python module to secure your own apps.")
    
    sdk_path = os.path.join(base_path, "sdk", "quantum_lock.py")
    if os.path.exists(sdk_path):
        with open(sdk_path, "rb") as f:
            st.download_button(
                label="Download SDK (.py)",
                data=f,
                file_name="quantum_lock.py",
                mime="text/x-python"
            )
    else:
        st.error("SDK file not found!")
        
    st.markdown("---")
    st.subheader("🔑 Key Management")
    if st.button("Rotate Keys 🔄"):
        st.session_state.ql.keygen()
        st.success("New Lattice Keys Generated!")

with col_viz:
    st.subheader("Encryption Visualizer")
    
    msg_in = st.text_input("Message to Encrypt", "Secrets of the Universe")
    
    if st.button("Encrypt & Untangle"):
        ql = st.session_state.ql
        msg_bytes = msg_in.encode('utf-8')
        
        # 1. Encrypt
        ciphertext = ql.encrypt(msg_bytes)
        u_vecs, v_vals = ciphertext
        
        # 2. Visualize the "Noisy Lattice"
        # We'll plot the 'v' values (masked message) vs the Decoded values
        
        st.info(f"Encrypted {len(msg_bytes)} bytes into {len(v_vals)} lattice points.")
        
        # Decrypt logic for viz
        decrypted_vals = []
        noise_vals = []
        
        sk = ql.sk
        q = ql.q
        
        for u, v in zip(u_vecs, v_vals):
            # Noisy value: v - s.u
            noisy = (v - np.dot(sk, u)) % q
            
            # Centered
            val_centered = noisy
            if val_centered > q//2: val_centered -= q
            
            noise_vals.append(val_centered)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('#111')
        
        # Colors: Blue = 0, Red = 1 (based on if it's closer to 0 or q/2)
        colors = []
        threshold = q // 4
        for val in noise_vals:
            if abs(val) > threshold:
                colors.append('#ff0055') # 1
            else:
                colors.append('#00f3ff') # 0
                
        x_pts = range(len(noise_vals))
        ax.scatter(x_pts, noise_vals, c=colors, s=50, alpha=0.8)
        
        # Bounds
        ax.axhline(0, color='white', linestyle='--', alpha=0.3, label='Bit 0 Center')
        ax.axhline(q//2, color='white', linestyle='--', alpha=0.3, label='Bit 1 Center')
        ax.axhline(-q//2, color='white', linestyle='--', alpha=0.3)
        
        ax.set_ylim(-q, q)
        ax.set_title("Lulu's Lattice: Untangling Signal from Noise", color='white')
        ax.set_ylabel("Mod q Space", color='white')
        
        st.pyplot(fig)
        plt.close(fig)
        
        # 3. Decrypt Result
        dec_bytes = ql.decrypt(ciphertext)
        st.success(f"Decrypted: {dec_bytes.decode('utf-8', errors='ignore')}")
        
    st.markdown("""
    **How it works (LWE)**:
    1. The message bit (0 or 1) is scaled to `0` or `q/2`.
    2. We add it to a random point on the lattice `As` plus some noise `e`.
    3. The ciphertext looks like random noise to anyone without the private key `s`.
    4. With `s`, we subtract the lattice point. The remaining value is `Message + Noise`.
    5. Since Noise is small, we can round to the nearest bit (0 or q/2).
    """)
