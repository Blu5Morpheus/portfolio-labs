import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from utils.lorenz_utils import simulate_with_divergence, calculate_horizon, estimate_lyapunov

st.set_page_config(page_title="Lorenz Chaos Lab", page_icon="🦋", layout="wide")

# Custom CSS for that cyberpunk feel (can be expanded later)
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦋 Lorenz Chaos Lab")
st.markdown("### Explore the Butterfly Effect")
st.markdown("Adjust parameters to see how tiny differences in initial conditions lead to completely different outcomes.")

# --- Sidebar Controls ---
st.sidebar.header("System Parameters")
sigma = st.sidebar.slider("Sigma (σ)", 0.1, 50.0, 10.0, help="Prandtl number")
rho = st.sidebar.slider("Rho (ρ)", 0.1, 100.0, 28.0, help="Rayleigh number")
beta = st.sidebar.slider("Beta (β)", 0.1, 20.0, 2.6667, help="Geometric factor")

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")
T = st.sidebar.slider("Duration (T)", 10.0, 100.0, 40.0)
dt = st.sidebar.slider("Time Step (dt)", 0.001, 0.05, 0.01, format="%.3f")
epsilon = st.sidebar.select_slider(
    "Epsilon (Initial Separation)", 
    options=[1e-9, 1e-7, 1e-5, 1e-3, 1e-1], 
    value=1e-5,
    help="Distance between the two starting points"
)

st.sidebar.markdown("---")
st.sidebar.header("Analysis")
delta = st.sidebar.slider("Prediction Horizon Threshold (δ)", 0.1, 20.0, 5.0, help="Distance at which prediction 'fails'")


# --- Simulation ---
with st.spinner('Running simulation...'):
    x0 = np.array([1.0, 1.0, 1.0])
    t, traj1, traj2, divergence = simulate_with_divergence(x0, epsilon, T, dt, sigma, rho, beta)
    horizon = calculate_horizon(t, divergence, delta)
    lyapunov = estimate_lyapunov(t, divergence)

# --- Metrics ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Prediction Horizon", 
              f"{horizon:.2f} s" if horizon else "> Max T", 
              help=f"Time until divergence > {delta}")

with col2:
    st.metric("Lyapunov Estimate", 
              f"{lyapunov:.2f}",
              help="Positive means chaotic ( exponential divergence )")

with col3:
    st.metric("Epsilon", f"{epsilon:.1e}")


# --- Visualizations ---
tab1, tab2 = st.tabs(["3D Attractor", "Divergence Analysis"])

with tab1:
    # 3D Plot
    fig = go.Figure()
    
    # Plot Trajectory 1
    # We will plot the whole thing, maybe subsample for performance if T is huge
    step_skip = 5 if len(t) > 5000 else 1
    
    fig.add_trace(go.Scatter3d(
        x=traj1[::step_skip,0], y=traj1[::step_skip,1], z=traj1[::step_skip,2], 
        mode='lines', name='Trajectory 1',
        line=dict(color='#00FFFF', width=2),
        opacity=0.8
    ))
    
    # Plot Trajectory 2 (only if they diverge visibly, otherwise it Z-fights)
    # Let's just plot it in a different color, maybe magenta
    fig.add_trace(go.Scatter3d(
        x=traj2[::step_skip,0], y=traj2[::step_skip,1], z=traj2[::step_skip,2], 
        mode='lines', name='Trajectory 2',
        line=dict(color='#FF00FF', width=2),
        opacity=0.5
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X', 
            yaxis_title='Y', 
            zaxis_title='Z',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0), 
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Trajectory Divergence ||Δ(t)||")
    st.markdown("Log-scale divergence shows exponential growth (linear slope) characteristic of chaos.")
    
    # Matplotlib for precise log plot control
    fig_div, ax = plt.subplots(figsize=(10, 4))
    fig_div.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Plot log divergence
    # Add small epsilon to avoid log(0)
    log_div_vals = np.log(divergence + 1e-20)
    ax.plot(t, log_div_vals, color='#00ff00', linewidth=1.5, label='Log Divergence')
    
    # Threshold line
    ax.axhline(np.log(delta), color='red', linestyle='--', alpha=0.7, label=f'Threshold (δ={delta})')
    
    # Horizon line
    if horizon:
        ax.axvline(horizon, color='yellow', linestyle=':', alpha=0.8, label=f'Horizon (t={horizon:.2f})')
        # Fill area after horizon
        ax.axvspan(horizon, T, color='red', alpha=0.1)

    ax.set_xlabel("Time (t)", color='white')
    ax.set_ylabel("Log Divergence", color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.2)
    
    st.pyplot(fig_div)
    
st.markdown("---")
st.markdown("""
**Definitions:**
- **Prediction Horizon**: The time until the error between our prediction (Trajectory 1) and reality (Trajectory 2) becomes unacceptably large (Threshold).
- **Lyapunov Exponent**: A measure of the rate of separation of infinitesimally close trajectories. Positive λ implies chaos.
""")
