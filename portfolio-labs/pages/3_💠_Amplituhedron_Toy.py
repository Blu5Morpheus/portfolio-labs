import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.positivity_utils import sample_kinematics, calculate_boundary_distance

st.set_page_config(page_title="Amplituhedron Kinematics", page_icon="💠", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("💠 Amplituhedron 'Kinematics'")
st.markdown("### A Toy Model of Positive Geometry")
st.markdown("""
This is a conceptual toy inspired by the **Amplituhedron**: collision probabilities (scattering amplitudes) 
are volumes of geometric objects defined by "positivity" constraints.
Here, we carve out an **allowed region** in a 2 parameter kinematic space.
""")

# --- Controls ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Constraints")
    curved = st.checkbox("Enable 'Loop' Curvature", value=False, help="Adds non-linear constraints (u*v > k)")
    k_threshold = st.slider("Positivity Threshold (k)", 0.0, 0.1, 0.02, step=0.005)
    
    st.divider()
    st.subheader("Probe Point")
    u_slider = st.slider("Energy Split (u)", 0.0, 1.0, 0.33)
    v_slider = st.slider("Angle Proxy (v)", 0.0, 1.0, 0.33)

with col2:
    # --- Visualization ---
    
    # 1. Generate Background Map
    # We'll use a dense grid for a smooth heatmap/contour
    grid_res = 100
    u_grid = np.linspace(0, 1, grid_res)
    v_grid = np.linspace(0, 1, grid_res)
    U, V = np.meshgrid(u_grid, v_grid)
    
    # Vectorized calculation for grid would be faster, but loop is fine for MVP 100x100
    Z = np.zeros_like(U)
    for i in range(grid_res):
        for j in range(grid_res):
            Z[i,j] = calculate_boundary_distance(U[i,j], V[i,j], k_threshold, curved)
            
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#111')
    
    # Contour / Heatmap of allowed region
    # 'inferno' or 'viridis' - bright means deep inside, black means forbidden
    contour = ax.imshow(Z, origin='lower', extent=[0,1,0,1], cmap='magma', vmin=0, vmax=1.0, alpha=0.9)
    
    # Plot the Probe Point
    ax.scatter([u_slider], [v_slider], color='#00FFFF', s=100, edgecolors='white', label='Probe (u,v)', zorder=10)
    
    # Draw Analytic Boundaries (Simplex)
    ax.plot([0, 1], [1, 0], color='cyan', linestyle='--', alpha=0.5, label='Simplex Boundary')
    ax.plot([0, 0], [0, 1], color='cyan', linestyle='--', alpha=0.3)
    ax.plot([0, 1], [0, 0], color='cyan', linestyle='--', alpha=0.3)
    
    # Draw Curved Boundary if active
    if curved and k_threshold > 0:
        # u*v = k => v = k/u
        us = np.linspace(k_threshold, 1, 100)
        vs = k_threshold / us
        # Clip to plot
        mask = vs <= 1
        ax.plot(us[mask], vs[mask], color='magenta', linestyle='-', linewidth=2, label='Positivity Bound')

    ax.set_title("Allowed Kinematic Region", color='white')
    ax.set_xlabel("u (Energy Split)", color='white')
    ax.set_ylabel("v (Angle Proxy)", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white', loc='upper right')
    
    st.pyplot(fig)

# --- Analysis ---
probe_score = calculate_boundary_distance(u_slider, v_slider, k_threshold, curved)
status = "ALLOWED" if probe_score > 0 else "FORBIDDEN"
status_color = "green" if probe_score > 0 else "red"

st.markdown(f"""
### Status: <span style="color:{status_color}">{status}</span>
- **Distance to Boundary**: {probe_score:.3f}
- **Interpretation**: { "Points inside contribute to the amplitude." if probe_score > 0 else "Points outside violate physical principles (in this toy model)." }
""", unsafe_allow_html=True)
