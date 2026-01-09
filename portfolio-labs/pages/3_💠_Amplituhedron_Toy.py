import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.positivity_utils import check_positivity, compute_volume_form, check_polygon_containment

# ...

matplotlib.use('Agg')

st.set_page_config(page_title="Amplituhedron Toy", page_icon="💠", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("💠 Amplituhedron Toy Model")
st.markdown("### Positive Geometries & Scattering Amplitudes")

st.warning("""
**Pedagogical Disclaimer**: 
This is a **simplified toy model** inspired by the concept of *Positive Geometries*. 
It demonstrates the core idea: defining a volume via inequality constraints ($Z > 0$) rather than Feynman diagrams.
It does **NOT** use Momentum Twistors or full Grassmanian geometry ($Gr(k, n)$).
""")

st.markdown("""
**Reference**: Arkani-Hamed, N., & Trnka, J. (2014). *The Amplituhedron*. Journal of High Energy Physics.
""")

# --- Setup ---
st.sidebar.header("Geometry Settings")
num_particles = st.sidebar.slider("Number of Particles (Vertices)", 3, 6, 4)
randomness = st.sidebar.slider("Vertex Randomness", 0.0, 1.0, 0.2)

# ... (Rest of logic unchanged, just framing updated) ...

# 1. Define Positive Region (The Polygon)
# In the toy model, we just define a Convex Polygon in 2D
# This represents the "Amplituhedron" A_{n,k,m} projected down

angles = np.linspace(0, 2*np.pi, num_particles, endpoint=False)
# Add noise to angles to make it irregular but ordered (Cyclic)
angles += np.random.normal(0, 0.1 * randomness, num_particles)
angles = np.sort(angles)

radii = 1.0 + np.random.normal(0, 0.2 * randomness, num_particles)
x_poly = radii * np.cos(angles)
y_poly = radii * np.sin(angles)

vertices = np.column_stack((x_poly, y_poly))

# --- Visualization ---

col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("The Geometry $A_{n,k}$")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#111')
    
    # Draw Polygon (The "Positive Region")
    # Close the loop
    x_plot = np.append(x_poly, x_poly[0])
    y_plot = np.append(y_poly, y_poly[0])
    
    ax.plot(x_plot, y_plot, color='#b026ff', linewidth=2, label='Boundary')
    ax.fill(x_plot, y_plot, color='#b026ff', alpha=0.1)
    
    # Scatter Vertices
    ax.scatter(x_poly, y_poly, color='#00f3ff', s=100, zorder=5)
    for i in range(num_particles):
        ax.text(x_poly[i]*1.1, y_poly[i]*1.1, f"Z_{i+1}", color='white', fontsize=12)
        
    # Probe Point
    st.markdown("**Probe the Amplitude Form**")
    px = st.slider("Probe X", -1.5, 1.5, 0.0)
    py = st.slider("Probe Y", -1.5, 1.5, 0.0)
    
    ax.scatter([px], [py], color='yellow', marker='x', s=150, label='Probe Y')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    ax.axis('off')
    
    st.pyplot(fig)
    plt.close(fig)

with col_info:
    st.subheader("Positivity Check")
    
    # In P^2 (Projective), being "Positive" means inside the convex hull
    # We check if Probe is inside the polygon
    
    is_inside, probe_score = check_polygon_containment(vertices, np.array([px, py]))
    
    if is_inside:
        st.success("INSIDE (Positive)")
        st.markdown(f"**Contribution**: $\\Omega \\sim 1/(\\text{{vol}})$")
    else:
        st.error("OUTSIDE")
        st.markdown("**Contribution**: $0$")
        
    st.write(f"Convexity Metric: {probe_score:.3f}")

    st.markdown("""
    **Theory Note**:
    The volume form $\\Omega$ has logarithmic singularities on the boundaries.
    $\\Omega = \\sum \\frac{1}{\\det(Z_i, Z_{i+1}, Y)}$
    """)

st.markdown("---")
st.subheader("📍 LHC/CERN Collision Event")
st.write("Simulating a scattering amplitude in the **Geometric Sector** of the detector.")

if st.button("TRIGGER EVENT (4-Gluon Scattering)"):
    # Visualize Feynman-like outgoing legs inside a Detector
    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#000000')
    
    # Detector Rings (CMS Style)
    # Tracker (Silicon)
    ax.add_artist(plt.Circle((0,0), 0.5, color='#333', fill=False, linestyle='--', linewidth=1))
    # ECAL (Green)
    ax.add_artist(plt.Circle((0,0), 1.0, color='green', alpha=0.3, fill=False, linewidth=5))
    # HCAL (Orange)
    ax.add_artist(plt.Circle((0,0), 1.4, color='orange', alpha=0.3, fill=False, linewidth=8))
    # Muon Chambers (Red blocks)
    ax.add_artist(plt.Circle((0,0), 1.8, color='red', alpha=0.2, fill=False, linewidth=2))
    
    # Center
    ax.scatter([0], [0], color='yellow', s=300, marker='*', zorder=10, label="Interaction Point")
    
    # 4 Legs (Momentum vectors)
    angles = np.linspace(0, 2*np.pi, 5)[:-1] + np.random.normal(0, 0.2, 4)
    for i, th in enumerate(angles):
        # Jet cones
        ax.arrow(0, 0, np.cos(th)*1.5, np.sin(th)*1.5, head_width=0.1, color='#00f3ff', linewidth=2, zorder=5)
        ax.text(np.cos(th)*1.7, np.sin(th)*1.7, f"Jet {i+1}", color='white', fontsize=10)
        
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title("Transverse Plane View (x-y)", color='white')
    st.pyplot(fig)
    plt.close(fig)
