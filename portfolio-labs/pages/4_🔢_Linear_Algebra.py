import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.la_utils import get_matrix_properties, get_preset_matrix, interpolate_matrices

st.set_page_config(page_title="Linear Algebra Playground", page_icon="🔢", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔢 Linear Algebra Playground")
st.markdown("### Visualizing 2D Transformations")
st.markdown("See how a matrix **A** transforms space, bases, and specific vectors.")

# --- Controls ---
col_ctrl, col_viz = st.columns([1, 2])

with col_ctrl:
    st.subheader("Matrix Control")
    
    mode = st.radio("Input Mode", ["Manual", "Presets", "Singularity Slider"])
    
    if mode == "Manual":
        col_m1, col_m2 = st.columns(2)
        a = col_m1.number_input("a (0,0)", value=1.0, step=0.1)
        b = col_m2.number_input("b (0,1)", value=0.0, step=0.1)
        c = col_m1.number_input("c (1,0)", value=0.0, step=0.1)
        d = col_m2.number_input("d (1,1)", value=1.0, step=0.1)
        M = np.array([[a,b],[c,d]])
        
    elif mode == "Presets":
        preset = st.selectbox("Choose Transformation", 
                             ["Identity", "Rotation (45°)", "Shear (X)", 
                              "Scaling (2x)", "Reflection (Y-axis)", "Singular (Projection X)"])
        a, b, c, d = get_preset_matrix(preset)
        st.info(f"Loaded: [[{a:.2f}, {b:.2f}], [{c:.2f}, {d:.2f}]]")
        M = np.array([[a,b],[c,d]])
        
    else: # Singularity Slider
        st.markdown("Interpolate between Identity and a Singular Matrix")
        t_sing = st.slider("Singularity Mix", 0.0, 1.0, 0.0)
        M_ident = np.eye(2)
        M_sing = np.array([[1.0, 1.0], [1.0, 1.0]]) # Det = 0
        M = interpolate_matrices(M_ident, M_sing, t_sing)
        a, b, c, d = M[0,0], M[0,1], M[1,0], M[1,1]
    
    st.markdown("---")
    st.subheader("Test Vector v")
    vx = st.slider("v_x", -2.0, 2.0, 1.0)
    vy = st.slider("v_y", -2.0, 2.0, 0.5)
    v = np.array([vx, vy])

# --- Calculations ---
props = get_matrix_properties(a,b,c,d)
det = props['det']
Av = M @ v

# --- Visualization ---
with col_viz:
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#111')
    
    # Grid setup
    limit = 3.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    ax.axhline(0, color='white', alpha=0.5)
    ax.axvline(0, color='white', alpha=0.5)
    
    # 1. Unit Square (Original)
    # Points: (0,0), (1,0), (1,1), (0,1), (0,0)
    square_x = [0, 1, 1, 0, 0]
    square_y = [0, 0, 1, 1, 0]
    ax.plot(square_x, square_y, 'w--', alpha=0.5, label='Original Unit Square')
    
    # 2. Transformed Square
    # Apply M to each point
    sq_pts = np.array([square_x, square_y])
    trans_sq = M @ sq_pts
    ax.plot(trans_sq[0,:], trans_sq[1,:], 'cyan', linewidth=2, label='Transformed Space')
    ax.fill(trans_sq[0,:], trans_sq[1,:], 'cyan', alpha=0.1)
    
    # 3. Vectors
    # Origin
    origin = np.array([0, 0])
    
    # Helper to plot vector
    def plot_vector(vec, origin, color, label, alpha=1.0):
        ax.quiver(origin[0], origin[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color, label=label, alpha=alpha)

    # Basis vectors i_hat, j_hat (Transformed)
    i_new = M @ np.array([1, 0])
    j_new = M @ np.array([0, 1])
    plot_vector(i_new, origin, '#FF00FF', "i' (Transformed)", alpha=0.8)
    plot_vector(j_new, origin, '#FFFF00', "j' (Transformed)", alpha=0.8)
    
    # Input Vector v
    plot_vector(v, origin, 'white', 'v (Input)')
    
    # Output Vector Av
    plot_vector(Av, origin, '#00FF00', 'Av (Result)')
    
    # Eigenvectors (if real)
    if props['is_real_eigen']:
        for i in range(len(props['eig_vals'])):
            val = props['eig_vals'][i]
            vec = props['eig_vecs'][:, i]
            # Scale for visibility
            plot_vector(vec * val, origin, 'red', f'Eigen {i+1} (λ={val:.2f})', alpha=0.5)
    
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white', loc='upper left')
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        
    st.pyplot(fig)

# --- Properties Display ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Determinant", f"{det:.3f}", help="Area scaling factor. 0 means collapse.")
col2.metric("Condition No.", f"{props['cond']:.2f}", help="Sensitivity to numerical error. Huge = near singular.")
col3.metric("Trace", f"{props['trace']:.2f}", help="Sum of diagonal elements.")
col4.metric("Real Eigenvalues?", "Yes" if props['is_real_eigen'] else "No", help="If No, involves rotation.")

st.markdown("---")
st.subheader("3. The Jacobian & Determinant")
show_jacobian = st.checkbox("Show Jacobian Interpretation 🧠")

if show_jacobian:
    st.markdown(f"""
    **The Determinant is the Expansion Factor.**
    
    *   **Input Area**: 1.0 (Unit Square)
    *   **Output Area**: {np.abs(det):.4f} (Parallelogram)
    *   **Ratio**: {np.abs(det):.4f}
    
    In calculus (change of variables), this $|\\det(J)|$ tells us how local volume scales.
    """)
    
    # Area verification
    v1 = M[:, 0]
    v2 = M[:, 1]
    # Cross product Z-component for 2D vectors
    area = np.abs(v1[0]*v2[1] - v1[1]*v2[0])
    st.caption(f"Numerically Calculated Area: {area:.4f} (Matches |det|)")

st.subheader("4. Eigenvector Residual Check")
st.markdown("Verifying $A v = \\lambda v$ numerically.")

if props['is_real_eigen']:
    vals = props['eig_vals']
    vecs = props['eig_vecs']
    
    for i in range(len(vals)):
        lam = vals[i]
        v_eig = vecs[:, i]
        
        # Av
        Av_res = M @ v_eig
        # lam * v
        lam_v = lam * v_eig
        
        # Residual ||Av - lam*v||
        residual = np.linalg.norm(Av_res - lam_v)
        
        col_res1, col_res2 = st.columns([3, 1])
        with col_res1:
            st.code(f"v{i+1}: {v_eig}\nλ{i+1}: {lam:.3f}")
        with col_res2:
            st.metric(f"Residual {i+1}", f"{residual:.2e}")
            
    if props['cond'] > 100:
         st.warning("⚠️ Matrix is ill-conditioned. Numerical stability at risk.")
else:
    st.write("Complex Eigenvalues (Rotation detected). Visual residuals skipped.")

st.markdown("---")
st.markdown("""
### What to look for:
- **Determinant = 0**: The square collapses into a line or point.
- **Negative Determinant**: space flips (like a mirror).
- **Eigenvectors**: The red arrows (if they exist) stay on their own span; they just stretch/shrink by λ.
""")
