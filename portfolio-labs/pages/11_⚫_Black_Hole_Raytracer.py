import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint

matplotlib.use('Agg')

st.set_page_config(page_title="Black Hole QFT", page_icon="⚫", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚫ Quantum Black Hole")
st.markdown("### Hawking Radiation & Geodesics")
st.markdown("Simulating **Quantum Field Theory (QFT)** effects near the Event Horizon.")

tab_geo, tab_qft = st.tabs(["Classic Raytracer", "Hawking Radiation (QFT)"])

with tab_geo:
    st.subheader("Schwarzschild Geodesics")
    b_impact = st.slider("Impact Parameter (b)", 0.0, 6.0, 2.8, key="geo_b")
    
    if st.button("Trace Ray", key="geo_btn"):
         # Solves u'' + u = 1.5 u^2
        def geodesic(y, phi):
            u, dudphi = y
            return [dudphi, 1.5*u**2 - u]
        
        # Initial: Far away (u~0), incoming
        y0 = [0.0, 1.0/b_impact]
        phi = np.linspace(0, 3*np.pi, 500)
        
        try:
            sol = odeint(geodesic, y0, phi)
            u = sol[:,0]
            valid = (u > 0.01) & (u < 5.0)
            r = 1.0/u[valid]
            p = phi[valid]
            x = r*np.cos(p)
            y = r*np.sin(p)
            
            fig, ax = plt.subplots(figsize=(6,6))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('black')
            ax.fill(np.cos(phi), np.sin(phi), color='black', zorder=10) # BH radius 1 in these units
            ax.plot(np.cos(phi), np.sin(phi), color='white') # Horizon
            ax.plot(x, y, color='#b026ff', linewidth=2)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            st.pyplot(fig)
            plt.close(fig)
        except:
             st.error("Photon Captured.")

with tab_qft:
    st.subheader("Virtual Pair Production & Tunneling")
    st.markdown("Visualizing $e^+ e^-$ pairs spawning from the vacuum near $R_s$.")
    
    if st.button("Simulate Quantum Vacuum ⚛️"):
        st.info("Calculating Tunneling Probabilities (Parikh-Wilczek)...")
        
        # Simulation
        # Spawn N particles near R=1
        N = 50
        thetas = np.random.uniform(0, 2*np.pi, N)
        # Pairs: Inner (just inside 1), Outer (just outside 1)
        # R_inner = 0.95, R_outer = 1.05
        
        # Energies (Boltzmann dist for Temperature T_H)
        # T_H ~ 1/M. Let M=1.
        energies = np.random.exponential(scale=1.0, size=N)
        
        # Tunneling Probability P ~ exp(-E)
        tunnel_probs = np.exp(-energies)
        escaped = np.random.rand(N) < tunnel_probs
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8,8))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('black')
        
        # Horizon
        circle = plt.Circle((0,0), 1.0, color='white', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        for i in range(N):
            th = thetas[i]
            E = energies[i]
            
            # Base pos
            x0 = np.cos(th)
            y0 = np.sin(th)
            
            # Anti-particle (Falling in)
            ax.plot([x0, 0.8*x0], [y0, 0.8*y0], color='red', alpha=0.5, linewidth=1)
            
            # Particle (Tunneling out?)
            if escaped[i]:
                # Escapes!
                dist = 2.0 + E # More energy, further reach visual
                ax.plot([x0, dist*x0], [y0, dist*y0], color='#00f3ff', alpha=0.8, linewidth=E*2)
                ax.scatter([dist*x0], [dist*y0], color='#00f3ff', s=E*20, marker='*')
            else:
                # Re-annihilates or trapped
                ax.scatter([1.05*x0], [1.05*y0], color='gray', s=10, alpha=0.3)
                
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f"Hawking Radiation ({np.sum(escaped)} particles tunneled)", color='white')
        
        st.pyplot(fig)
        plt.close(fig)
        
        st.metric("Total Radiated Energy", f"{np.sum(energies[escaped]):.4f} units")

st.markdown("""
**Parikh-Wilczek Mechanism**:
Particles tunnel through the horizon barrier as a semi-classical WKB approximation.
The radiation spectrum follows a Black Body distribution with temperature $T_H = \\frac{1}{8\\pi M}$.
""")
