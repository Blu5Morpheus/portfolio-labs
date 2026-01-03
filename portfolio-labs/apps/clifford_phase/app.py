import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from ga_utils import generate_lorenz_trajectory, compute_tangents, compute_wedge_magnitude

st.set_page_config(page_title="Clifford Algebra Phase Space", page_icon="📐", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("📐 Clifford Algebra Phase Space")
st.markdown("### Quantifying Trajectory Folding with Geometric Algebra")
st.markdown("We analyze the local geometry of the Lorenz attractor using the **wedge product** magnitude $||v_t \\wedge v_{t+1}||$, which quantifies local turning and folding complexity.")

# --- Sidebar ---
st.sidebar.header("Lorenz Parameters")
sigma = st.sidebar.slider("Sigma (σ)", 0.1, 50.0, 10.0)
rho = st.sidebar.slider("Rho (ρ)", 0.1, 100.0, 28.0)
beta = st.sidebar.slider("Beta (β)", 0.1, 20.0, 2.6667)

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")
T = st.sidebar.slider("Duration", 10.0, 100.0, 50.0)
dt = st.sidebar.slider("Time Step", 0.001, 0.05, 0.01)

# --- Computation ---
with st.spinner('Calculating wedge products...'):
    x0 = np.array([1.0, 1.0, 1.0])
    t, traj = generate_lorenz_trajectory(x0, T, dt, sigma, rho, beta)
    
    # Tangents: length N-1
    tangents = compute_tangents(traj)
    # Wedge Mags: length N-2
    wedges = compute_wedge_magnitude(tangents)
    
    # Pad wedges for plotting consistency
    # (Since we lose 2 points in calc, let's pad the start with the first value twice or 0)
    wedges_padded = np.pad(wedges, (2, 0), 'edge')

# --- Metrics ---
avg_wedge = np.mean(wedges)
max_wedge = np.max(wedges)

col1, col2 = st.columns(2)
col1.metric("Avg Wedge Magnitude", f"{avg_wedge:.2e}", help="Average local folding intensity")
col2.metric("Max Wedge Magnitude", f"{max_wedge:.2e}", help="Peak folding intensity")

# --- Visualizations ---
tab1, tab2 = st.tabs(["Phase Space Geometry", "Wedge Magnitude vs Time"])

with tab1:
    fig = go.Figure()
    
    # We want to color the line by wedge magnitude.
    # Plotly lines don't support gradients easily in one trace.
    # We use Scatter3d with marker mode for gradient (heavy) or segments (complicated).
    # Easier approximation: Scatter3d with markers size=0, line color based on a dimension? 
    # Actually, Plotly Scatter3d line color can be a constant, or we can use a marker trick if points are dense.
    # Let's try the Marker trick (mode='lines+markers') with small markers colored by wedge.
    
    # Subsample for performance
    step = 2 if len(t) > 2000 else 1
    
    fig.add_trace(go.Scatter3d(
        x=traj[::step, 0], 
        y=traj[::step, 1], 
        z=traj[::step, 2],
        mode='lines', # Just lines for shape clarity first? 
        # Actually proper gradient lines in 3D is hard in single trace. 
        # Let's do a Scatter3d with 'markers' only, sized small, colored by wedge.
        # It looks like a glowing particle stream.
        marker=dict(
            size=2,
            color=wedges_padded[::step],
            colorscale='Inferno', 
            colorbar=dict(title="||v ∧ v'||"),
            opacity=0.8
        ),
        line=dict(color='rgba(0,0,0,0)'), # Hide line, show dots
        name='Trajectory'
    ))
    
    # Add a thin line underneath to guide the eye
    fig.add_trace(go.Scatter3d(
        x=traj[::step, 0], y=traj[::step, 1], z=traj[::step, 2],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.1)', width=1),
        hoverinfo='skip'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333", showbackground=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333", showbackground=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333", showbackground=False),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Wedge Magnitude Time Series")
    st.markdown("Peaks correspond to sharp turns or 'folds' in the phase space trajectory.")
    
    fig_w, ax = plt.subplots(figsize=(10, 4))
    fig_w.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    ax.plot(t, wedges_padded, color='#ff9900', linewidth=1)
    
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("||v ∧ v'||", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.grid(True, alpha=0.1)
    
    st.pyplot(fig_w)

st.markdown("---")
st.markdown("### The Math")
st.latex(r"v_t = p_{t+1} - p_t")
st.latex(r"\text{Wedge Magnitude} = ||v_t \wedge v_{t+1}|| = ||v_t \times v_{t+1}||")
st.markdown("In Geometric Algebra, the **bivector** $v_t \wedge v_{t+1}$ represents the oriented plane segment swept by the tangent vector turning. Its magnitude quantifies the 'amount of turning' or local curvature.")
