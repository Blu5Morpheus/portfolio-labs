import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

st.set_page_config(page_title="Numerical PDE Explorer", page_icon="💥", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("💥 Numerical PDE & Stability Explorer")
st.markdown("### The Finite Difference Method & CFL Condition")
st.markdown("""
Solving the **1D Heat Equation** $\\frac{\\partial u}{\\partial t} = \\alpha \\frac{\\partial^2 u}{\\partial x^2}$.
Numerical solvers are approximate. If the timestep $\\Delta t$ is too large relative to space step $\\Delta x$, the solution **explodes**.
""")

# --- Setup ---
st.sidebar.header("Solver Parameters")
N = st.sidebar.slider("Grid Points (N)", 10, 100, 50)
alpha = 1.0 # Thermal diffusivity

L = 1.0
dx = L / (N - 1)
x = np.linspace(0, L, N)

st.sidebar.markdown("---")
st.sidebar.subheader("Timestep Control")
dt = st.sidebar.number_input("Time Step (dt)", value=0.0005, format="%.5f", step=0.0001)

# CFL Calculation
# Stability for Explicit Euler: r <= 0.5
r = alpha * dt / (dx**2)

st.sidebar.markdown(f"**Courant Ratio (r):** {r:.4f}")
if r > 0.5:
    st.sidebar.error("⚠️ UNSTABLE (r > 0.5)")
else:
    st.sidebar.success("✅ STABLE (r <= 0.5)")

# Initial Condition
# Gaussian pulse
u0 = np.exp(-100 * (x - 0.5)**2)

# Solver: FTCS (Forward Time Centered Space)
# u_new[i] = u[i] + r * (u[i+1] - 2u[i] + u[i-1])

steps = st.slider("Simulation Steps", 0, 500, 100)

u = u0.copy()
history = [u.copy()]

# Run Simulation
for _ in range(steps):
    u_new = np.zeros_like(u)
    # Boundaries fixed at 0 (Dirichlet)
    for i in range(1, N-1):
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new
    history.append(u.copy())

# --- Viz ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Solution Field $u(x,t)$")
    
    # Heatmap
    hist_arr = np.array(history)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#111')
    
    # Log scale if unstable to show explosion
    if r > 0.55:
        # Just show magnitude issues
        im = ax.imshow(hist_arr, aspect='auto', cmap='inferno', origin='lower', 
                       extent=[0, L, 0, steps*dt])
        st.caption("Showing magnitude (solution exploding!)")
    else:
        im = ax.imshow(hist_arr, aspect='auto', cmap='inferno', origin='lower',
                       extent=[0, L, 0, steps*dt])
        
    ax.set_xlabel("Position (x)", color='white')
    ax.set_ylabel("Time (t)", color='white')
    ax.set_title("Heat Diffusion History", color='white')
    ax.tick_params(colors='white')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Stability Analysis")
    
    # Profile at final step
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor('none')
    ax2.set_facecolor('#111')
    
    ax2.plot(x, history[0], color='gray', linestyle='--', label='Initial')
    ax2.plot(x, history[-1], color='#00f3ff', label=f'Step {steps}')
    
    if r > 0.5:
        ax2.set_title("⚠️ Numerical Instability Detected!", color='red')
    else:
        ax2.set_title("Stable Diffusion", color='green')
        
    ax2.legend()
    ax2.tick_params(colors='white')
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.info("""
    **Von Neumann Stability Analysis**:
    The error growth factor is $G = 1 - 4r \\sin^2(k \\Delta x / 2)$.
    If $|G| > 1$, errors grow exponentially.
    For Explicit Euler Heat Equation, this requires $r \\le 0.5$.
    """)
