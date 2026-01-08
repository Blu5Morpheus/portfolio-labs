import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import pdist, squareform

matplotlib.use('Agg')

st.set_page_config(page_title="Topology & Data Shape", page_icon="🍩", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("🍩 Topology & Shape of Data")
st.markdown("### Persistent Homology (Simplicial Complexes)")
st.markdown("""
Data has shape. Topology measures features (Components, Loops, Voids) that persist across scales.
We simulate a **Vietoris-Rips Filtration** by growing balls around points and detecting when components merge (Betti-0).
""")

# --- Setup ---
st.sidebar.header("Data Generator")
shape_type = st.sidebar.selectbox("Shape", ["Noisy Circle", "Two Clusters", "Random Cloud"])
n_points = st.sidebar.slider("N Points", 20, 100, 50)
noise_level = st.sidebar.slider("Noise", 0.0, 0.5, 0.1)

# Generate Data
if shape_type == "Noisy Circle":
    t = np.linspace(0, 2*np.pi, n_points)
    x = np.cos(t) + np.random.normal(0, noise_level, n_points)
    y = np.sin(t) + np.random.normal(0, noise_level, n_points)
elif shape_type == "Two Clusters":
    x = np.concatenate([np.random.normal(0, 0.2, n_points//2), np.random.normal(2, 0.2, n_points//2)])
    y = np.concatenate([np.random.normal(0, 0.2, n_points//2), np.random.normal(2, 0.2, n_points//2)])
else:
    x = np.random.normal(0, 1, n_points)
    y = np.random.normal(0, 1, n_points)
    
points = np.column_stack((x, y))

# --- TDA Logic (Manual Betti-0) ---
# We compute Connected Components for a range of epsilons (filtration)
# This gives us the Betti-0 Barcode

dists = squareform(pdist(points))
epsilons = np.linspace(0, 2.0, 50)
betti_0_counts = []

for eps in epsilons:
    # Adjacency
    adj = dists < eps
    # BFS/DFS for components
    visited = np.zeros(n_points, dtype=bool)
    n_components = 0
    for i in range(n_points):
        if not visited[i]:
            n_components += 1
            # Visit all reachable
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                neighbors = np.where(adj[node])[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
    betti_0_counts.append(n_components)

# --- Visualization ---
col_viz, col_barcode = st.columns(2)

with col_viz:
    st.subheader("Data Point Cloud & Simplex")
    
    current_eps = st.slider("Current Epsilon (Connectivity Radius)", 0.0, 2.0, 0.5)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#111')
    
    # Draw points
    ax.scatter(x, y, color='#00f3ff', s=50, zorder=10)
    
    # Draw connections (Simplex 1-skeleton)
    # Brute force line drawing for small N
    for i in range(n_points):
        for j in range(i+1, n_points):
            if dists[i, j] < current_eps:
                ax.plot([x[i], x[j]], [y[i], y[j]], color='#b026ff', alpha=0.3)
                
    ax.set_title(f"Connectivity at ε = {current_eps:.2f}", color='white')
    ax.axis('equal')
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

with col_barcode:
    st.subheader("Persistent Homology (Betti-0)")
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor('none')
    ax2.set_facecolor('#111')
    
    ax2.plot(epsilons, betti_0_counts, color='orange', drawstyle='steps-post', linewidth=2)
    ax2.set_xlabel("Filtration Radius (ε)", color='white')
    ax2.set_ylabel("Number of Components (β0)", color='white')
    ax2.set_title("Betti-0 Curve", color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # Indicator for current epsilon
    ax2.axvline(current_eps, color='white', linestyle='--', label='Current ε')
    
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.info("""
    **Interpretation**:
    - **Long plateaus** indicate persistent features (robust clusters).
    - **Short drops** indicate noise.
    - We look for the "Lifespan" of topological features.
    """)
