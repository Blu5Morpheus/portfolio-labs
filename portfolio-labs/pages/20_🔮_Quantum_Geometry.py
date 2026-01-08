import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('Agg')

st.set_page_config(page_title="Quantum Geometry", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Quantum State Geometry")
st.markdown("### Hilbert Space & The Bloch Sphere")
st.markdown("""
Visualizing **Mixed States** (Density Matrices $\\rho$) and **Decoherence**. 
A pure state $|\psi\\rangle$ lives on the surface. A mixed state lives inside.
""")

col_ctrl, col_viz = st.columns([1, 2])

with col_ctrl:
    st.subheader("State Control")
    
    # Angles
    theta = st.slider("Theta (Polar)", 0.0, np.pi, np.pi/2)
    phi = st.slider("Phi (Azimuthal)", 0.0, 2*np.pi, 0.0)
    
    # Purity / Decoherence
    st.markdown("---")
    st.subheader("Decoherence")
    purity = st.slider("Purity (Radius)", 0.0, 1.0, 1.0, help="1.0 = Pure State, 0.0 = Maximally Mixed")
    
    # Calculate Bloch Vector
    rx = purity * np.sin(theta) * np.cos(phi)
    ry = purity * np.sin(theta) * np.sin(phi)
    rz = purity * np.cos(theta)
    
    st.code(f"Bloch Vector:\n[{rx:.2f}, {ry:.2f}, {rz:.2f}]")
    
    # Density Matrix
    # rho = 1/2 (I + r . sigma)
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    rho = 0.5 * (I + rx*X + ry*Y + rz*Z)
    
    st.markdown("**Density Matrix $\\rho$**:")
    st.write(np.round(rho, 3))
    
    p_trace = np.trace(rho @ rho)
    st.metric("Purity Tr($\\rho^2$)", f"{p_trace.real:.3f}")

with col_viz:
    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor('none')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('none')
    
    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)
    
    # Axis lines
    ax.plot([-1, 1], [0, 0], [0, 0], 'w--', alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], 'w--', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1, 1], 'w--', alpha=0.3)
    
    # Labels
    ax.text(1.1, 0, 0, "|+>", color='white')
    ax.text(-1.1, 0, 0, "|->", color='white')
    ax.text(0, 0, 1.1, "|0>", color='white')
    ax.text(0, 0, -1.1, "|1>", color='white')
    
    # The State Vector
    ax.quiver(0, 0, 0, rx, ry, rz, color='#00f3ff', linewidth=3, arrow_length_ratio=0.1)
    
    # Trace/History (Decoherence path or Rotation path?)
    # Just show volume projection
    
    # View
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    
    st.pyplot(fig)
    plt.close(fig)
    
    if purity < 0.99:
        st.info(f"State is Mixed. Information loss likely due to entanglement with environment.")
    else:
        st.success("State is Pure.")
