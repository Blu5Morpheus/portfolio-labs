import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pennylane as qml
from pennylane import numpy as pnp

matplotlib.use('Agg')

st.set_page_config(page_title="VQE Tungsten", page_icon="💎", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("💎 Toy Spin-Lattice (Tungsten-Inspired)")
st.markdown("### Variational Quantum Eigensolver")
st.markdown("Finding the Ground State Energy of a 1D toy lattice model inspired by Tungsten bonding.")

st.info("""
**Physical Sandbox Disclaimer**:
Real material fracture requires **Density Functional Theory (DFT)** and many-body electron simulations.
This model uses a simplified **Tight-Binding Hamiltonian** mapped to qubits to demonstrate the *principle* of using VQE for material properties.
""")

# --- VQE Setup ---
# Toy Hamiltonian for 2 interacting atoms (1D dimer unit cell)
# H = J * (Z0 Z1) + h * (X0 + X1)
# J represents the bond strength (decreases with strain)
# h represents transverse field / tunneling

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params, J, h):
    # Ansatz: Hardware Efficient
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    
    # Measure Hamiltonian Expectation
    # H = J Z0 Z1 + h X0 + h X1
    # We return the expectation of terms
    # Using PennyLane's Hamiltonian structure is cleaner but manual is explicit
    
    return qml.expval(qml.Hamiltonian(
        [J, h, h], 
        [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0), qml.PauliX(1)]
    ))

# Wrapper for Gradient Descent
def cost_fn(params, J, h):
    return circuit(params, J, h)

# --- UI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Lattice Parameters")
    strain_range = np.linspace(0, 0.2, 10) 
    st.write("Simulating range: 0% - 20% Strain")
    
    # New Physics: Temperature
    temp = st.slider("Temperature (Kelvin)", 0, 3000, 300, help="Thermal energy excites phonons, blurring the energy bands.")
    kB = 8.617e-5 # eV/K (Boltzman)
    thermal_E = kB * temp
    st.metric("Thermal Energy (kT)", f"{thermal_E:.3f} eV")
    
    if st.button("Run VQE Sweep ⚛️"):
        st.info("Optimizing Quantum Circuit for each strain point...")
        
        energies = []
        params = pnp.random.random(4)
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        
        progress = st.progress(0)
        
        for i, s in enumerate(strain_range):
            J_val = 1.0 * np.exp(-5.0 * s)
            h_val = 0.5 
            
            for _ in range(20):
                params = opt.step(lambda p: cost_fn(p, J_val, h_val), params)
            
            E_ground = cost_fn(params, J_val, h_val)
            
            # Physics: Thermal Fluctuations
            # VQE finds E0. Real material has E ~ E0 + Thermal Noise
            E_real = E_ground + np.random.normal(0, thermal_E)
            
            energies.append(E_real)
            progress.progress((i+1)/len(strain_range))
        
        st.success("Sweep Complete!")
        st.session_state.vqe_energies = energies
        st.session_state.strain_axis = strain_range

with col2:
    st.subheader("Binding Energy Curve")
    if 'vqe_energies' in st.session_state:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('#111')
        
        s_axis = st.session_state.strain_axis * 100 # %
        E_axis = st.session_state.vqe_energies
        
        ax.plot(s_axis, E_axis, 'o-', color='#00f3ff', label='VQE Ground State')
        
        # Fracture Threshold
        # If Energy rises (becomes less negative) too much, bond breaks
        # Base energy is usually negative (bound). 0 is free.
        
        ax.axhline(-0.8, color='red', linestyle='--', label='Fracture Limit')
        
        ax.set_xlabel("Strain (%)", color='white')
        ax.set_ylabel("Ground State Energy <H>", color='white')
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
        ax.tick_params(colors='white')
        
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Run the VQE sweep to see the physics.")

st.markdown("""
**Technical Detail**:
We minimize $\\langle \\psi(\\theta) | H(strain) | \\psi(\\theta) \\rangle$ using a hardware-efficient ansatz.
As strain increases, the coupling term $J Z_0 Z_1$ weakens, raising the ground state energy towards zero (unbound/fractured state).
""")
