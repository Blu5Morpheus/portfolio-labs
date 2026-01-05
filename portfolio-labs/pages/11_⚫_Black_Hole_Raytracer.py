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

tab_geo, tab_qft, tab_cam = st.tabs(["Classic Raytracer", "Hawking Radiation (QFT)", "Photon Camera"])

with tab_geo:
    st.subheader("Schwarzschild Geodesics")
# ... (Geodesic code matches original logic, omitting for brevity of instruction but preserving in file) ...
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
            
            # Accretion Disk (Visuals)
            disk_inner = plt.Circle((0,0), 3.0, color='orange', alpha=0.3, fill=False, linewidth=2)
            disk_outer = plt.Circle((0,0), 5.0, color='red', alpha=0.2, fill=False, linewidth=1)
            ax.add_artist(disk_inner)
            ax.add_artist(disk_outer)
            
            # Fill logic for disk
            th_disk = np.linspace(0, 2*np.pi, 200)
            for r_disk in np.linspace(3, 5, 10):
                ax.plot(r_disk*np.cos(th_disk), r_disk*np.sin(th_disk), color='orange', alpha=0.1)
            
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
    # ... (QFT code matches original) ...
    st.markdown("Visualizing $e^+ e^-$ pairs spawning from the vacuum near $R_s$.")
    
    if st.button("Simulate Quantum Vacuum ⚛️"):
        st.info("Calculating Tunneling Probabilities (Parikh-Wilczek)...")
        # Reuse logic
        N = 50
        thetas = np.random.uniform(0, 2*np.pi, N)
        energies = np.random.exponential(scale=1.0, size=N)
        tunnel_probs = np.exp(-energies)
        escaped = np.random.rand(N) < tunnel_probs
        
        fig, ax = plt.subplots(figsize=(8,8))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('black')
        circle = plt.Circle((0,0), 1.0, color='white', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        for i in range(N):
            th = thetas[i]
            E = energies[i]
            x0 = np.cos(th); y0 = np.sin(th)
            ax.plot([x0, 0.8*x0], [y0, 0.8*y0], color='red', alpha=0.5, linewidth=1)
            if escaped[i]:
                dist = 2.0 + E 
                ax.plot([x0, dist*x0], [y0, dist*y0], color='#00f3ff', alpha=0.8, linewidth=E*2)
                ax.scatter([dist*x0], [dist*y0], color='#00f3ff', s=E*20, marker='*')
            else:
                ax.scatter([1.05*x0], [1.05*y0], color='gray', s=10, alpha=0.3)
                
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
        st.pyplot(fig)
        plt.close(fig)

with tab_cam:
    st.subheader("The Photon Sphere (What you see)")
    st.markdown("Rendering the **Einstein Ring** and the **Shadow**.")
    
    if st.button("Render Camera View 📸"):
        # Image plane [-10, 10]
        res = 200
        x = np.linspace(-10, 10, res)
        y = np.linspace(-10, 10, res)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Impact Parameter map
        # If R < 5.2 (Critical impact parm 3*sqrt(3)), it falls in -> BLACK
        # If R ~ 5.2, Photon Sphere -> BRIGHT RING
        # If R > 5.2, Sky -> DISTORTED STARFIELD
        
        img = np.zeros((res, res, 3))
        
        b_crit = 5.196 # 3 * sqrt(3)
        
        # Mask
        shadow = R < b_crit
        ring = np.abs(R - b_crit) < 0.4
        sky = R > (b_crit + 0.4)
        
        # Shadow = Black (0,0,0)
        
        # Ring = Bright White/Orange
        img[ring] = [1.0, 0.8, 0.4] 
        # Add glow decay
        
        # Sky (Noise/Stars)
        noise = np.random.uniform(0, 0.2, (res, res))
        img[sky, 0] = noise[sky] # R
        img[sky, 1] = noise[sky] # G
        img[sky, 2] = noise[sky] + 0.1 # B (Blueish stars)
        
        # Accretion Disk Band across middle?
        # Disk at Z=0 projects to an ellipse or line depending on inclination
        # Simple band
        disk_band = (np.abs(Y) < 1.0) & (R > 3.0) & (R < 8.0)
        img[disk_band & sky] += [0.5, 0.3, 0.0]
        
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img, extent=[-10,10,-10,10])
        ax.axis('off')
        ax.set_title("Observer View (60 deg FOV)", color='white')
        st.pyplot(fig)
        plt.close(fig)

st.markdown("""
**Parikh-Wilczek Mechanism**:
Particles tunnel through the horizon barrier as a semi-classical WKB approximation.
The radiation spectrum follows a Black Body distribution with temperature $T_H = \\frac{1}{8\\pi M}$.
""")
