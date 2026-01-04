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
- **🔥 PINN Heat Solver**: Physics-Informed Neural Network solving thermodynamics without a mesh.
- **💎 Tungsten Fracture**: Quantum tight-binding simulation of crystal lattice breakdown.
- **⚫ Black Hole Raytracer**: Schwarzschild geodesic integrator for gravitational lensing.
- **🤖 RL Ship Docker**: Deep Q-Network agent (DQN) with adversarial wind.
- **☢️ Nuclear Forensics**: AI Yield Estimator via Sedov-Taylor blast waves.
- **🤿 Scuba AI**: Neural Buoyancy Control agent fighting Boyle's Law.
- **📝 Engineering Blog**: Integrated viewer for technical design specifications (e.g. Radio Telescopes).
- **🛠️ Clifford General**: (Under Construction) Generalized Geometric Algebra toolkit.

---
*Built with Python & Streamlit*
""")

st.sidebar.success("Select a demo above.")
