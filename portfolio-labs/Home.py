import streamlit as st

st.set_page_config(
    page_title="Raven Physics Labs",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌌 Raven Physics Labs")

st.markdown("""
### Welcome to the Reality Engine

This is a collection of interactive physics, mathematics, and geometry demonstrations.
Select a simulation from the sidebar to begin.

#### Available Modules:

- **🦋 Lorenz Chaos Lab**: Explore the butterfly effect, divergence, and prediction horizons in strange attractors.
- **📐 Clifford Phase Space**: Visualize local trajectory folding using Geometric Algebra bivectors.
- **💠 Amplituhedron Toy**: A conceptual explorer for positive geometry and scattering amplitudes.
- **🔢 Linear Algebra Playground**: Intuitive visualization of 2D matrix transformations and eigenvectors.
- **⚛️ Quantum LIGO Detector**: Variational Quantum Classifier (VQC) detecting gravitational waves in noisy strain data.

---
*Built with Python & Streamlit*
""")

st.sidebar.success("Select a demo above.")
