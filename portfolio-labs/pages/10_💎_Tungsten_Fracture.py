import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Tungsten Fracture", page_icon="💎", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("💎 Tungsten Quantum Fracture")
st.markdown("### Tight-Binding Lattice Simulation")
st.markdown("Simulating the breakdown of a Tungsten crystal lattice under strain by solving the 1D Schrödinger Equation (Tight-Binding Model).")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Control")
    strain = st.slider("Applied Strain (%)", 0.0, 20.0, 0.0)
    
    # Physics Parameters
    # As strain increases, atoms move apart -> Hopping integral (t) decreases
    t0 = 1.0 # Base hopping energy (eV)
    # Exponential decay of overlap integral with distance
    hopping = t0 * np.exp(-0.2 * strain) 
    
    st.metric("Hopping Energy (t)", f"{hopping:.4f} eV")
    
    # Lattice Constant 'a' stretches
    a0 = 3.16 # Angstroms (Tungsten roughly)
    a = a0 * (1 + strain/100)
    st.metric("Lattice Scale", f"{a:.2f} Å")

with col2:
    st.subheader("Energy Band Structure")
    
    # Tight Binding Dispersion: E(k) = -2t * cos(ka)
    k = np.linspace(-np.pi/a, np.pi/a, 200)
    E = -2 * hopping * np.cos(k * a)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#111')
    
    ax.plot(k, E, color='#00f3ff', linewidth=3, label='Conduction Band')
    
    # Visualizing the "Gap" or Stability
    # For a metal like Tungsten, it's about the density of states and bond strength
    # Toy fracture logic: If hopping energy drops below threshold, bond breaks
    fracture_threshold = 0.3
    
    if hopping < fracture_threshold:
        st.error("⚠️ CRITICAL FAILURE: LATTICE FRACTURE DETECTED")
        ax.set_title("FRACTURED STATE", color='red')
        ax.plot(k, E, color='red', linestyle='--')
    else:
        st.success("Structure Stable")
        ax.set_title(f"Electronic Band Structure (Strain: {strain}%)", color='white')
    
    ax.set_xlabel("Wavevector (k)", color='white')
    ax.set_ylabel("Energy (eV)", color='white')
    ax.tick_params(colors='white')
    ax.grid(color='#333', linestyle='--')
    
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.markdown("""
**The Physics**:
Strain increases the distance between atoms ($a$). This reduces the electron orbital overlap, decreasing the hopping integral ($t$). 
When $t$ drops too low, the metallic cohesion fails, simulating a micro-fracture event at the quantum level.
""")
