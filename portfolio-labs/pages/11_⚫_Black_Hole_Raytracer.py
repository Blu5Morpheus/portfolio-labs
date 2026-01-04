import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint

matplotlib.use('Agg')

st.set_page_config(page_title="Black Hole Raytracer", page_icon="⚫", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚫ Schwarzschild Raytracer")
st.markdown("### Gravitational Lensing Simulation")
st.markdown("Tracing photon geodesics around a non-rotating Black Hole. Light bends due to spacetime curvature.")

# Schwarzschild Radius Rs = 2GM/c^2
# We work in units where Rs = 1

st.sidebar.header("Photon Parameters")
b_impact = st.sidebar.slider("Impact Parameter (b)", 0.0, 5.0, 2.8, help="Distance of the light ray from the center at infinity.")
st.sidebar.caption("Critical limit: b = 2.6 (Photon Capture)")

# --- Physics (Geodesic Equation) ---
# Second order ODe for u = 1/r as function of phi
# d^2u/dphi^2 + u = 1.5 * u^2 (in units Rs=1? No, usually slightly different scaling)
# Let's use standard u'' + u = 3GM/c^2 * u^2
# In units M=1, G=1, c=1 -> Rs = 2.
# Equation: u'' + u = 3M * u^2 = 1.5 * Rs * u^2?
# Let's define Rs=1 unit. Then 3M = 1.5.
# Equation: u'' + u = 1.5 * u^2

def geodesic(y, phi):
    u, dudphi = y
    d2udphi2 = 1.5 * u**2 - u
    return [dudphi, d2udphi2]

if st.button("Trace Ray"):
    # Initial Conditions (Far away)
    # r -> infinity, so u = 1/r -> 0
    # b is impact parameter.
    # u(0) = 0
    # du/dphi(0) = 1/b (Geometric derivation)
    
    y0 = [0.0, 1.0/b_impact]
    
    # Angle range (Beam coming from right, passing BH, going left)
    phi = np.linspace(0, 3*np.pi, 1000)
    
    # Integrate
    try:
        sol = odeint(geodesic, y0, phi)
        u = sol[:, 0]
        
        # Convert back to r = 1/u
        # Filter singularities (u=0 implies r=inf)
        valid = (u > 0.01) & (u < 5.0) # Cap near horizon to avoid div
        u_valid = u[valid]
        phi_valid = phi[valid]
        r = 1.0 / u_valid
        
        # Cartesian for plot
        x = r * np.cos(phi_valid)
        y = r * np.sin(phi_valid)
        
        # Event Horizon (Rs=1)
        theta = np.linspace(0, 2*np.pi, 100)
        xh = 1.0 * np.cos(theta)
        yh = 1.0 * np.sin(theta)
        
        # Photon Sphere (1.5 Rs)
        xp = 1.5 * np.cos(theta)
        yp = 1.5 * np.sin(theta)
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('#000')
        
        # Black Hole
        ax.fill(xh, yh, color='black', zorder=10)
        ax.plot(xh, yh, color='white', linewidth=1, label='Event Horizon')
        ax.plot(xp, yp, color='orange', linestyle='--', alpha=0.5, label='Photon Sphere')
        
        # Ray
        ax.plot(x, y, color='#b026ff', linewidth=2, label='Photon Path')
        
        # Decor
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
        
        st.pyplot(fig)
        plt.close(fig)
        
        if np.min(r) < 1.1:
            st.error("Photon Captured by Event Horizon!")
        else:
            st.success("Photon Escaped (Lensed)")
            
    except:
        st.warning("Integration Unstable (Singularity Reached)")

st.markdown("""
**The Math**:
We solve the null geodesic equation in Schwarzschild metric:
$\\frac{d^2u}{d\\phi^2} + u = \\frac{3}{2} R_s u^2$
where $u = 1/r$. The non-linear term $u^2$ is responsible for the extreme light bending (lensing) that doesn't exist in Newtonian gravity.
""")
