import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="PINN Heat Solver", page_icon="🔥", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔥 PINN Heat Solver")
st.markdown("### Physics-Informed Neural Network")
st.markdown(r"Solving the 1D Heat Equation: $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$ using Deep Learning.")

# --- Hyperparameters ---
st.sidebar.header("Simulation Parameters")
alpha = st.sidebar.slider("Thermal Diffusivity (alpha)", 0.01, 1.0, 0.1)
epochs = st.sidebar.slider("Training Epochs", 100, 2000, 500)

# --- The Network ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x, t):
        # Concatenate x and t
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

if st.button("Train PINN"):
    st.info(f"Training for {epochs} epochs...")
    
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Collocation Points (Physics Domain)
    # Random points in x=[-1, 1], t=[0, 1]
    x_phy = torch.rand(1000, 1) * 2 - 1
    t_phy = torch.rand(1000, 1)
    x_phy.requires_grad = True
    t_phy.requires_grad = True
    
    # Boundary Conditions (t=0)
    # Initial Condition: Sinusoidal
    x_bc = torch.rand(200, 1) * 2 - 1
    t_bc = torch.zeros(200, 1)
    u_bc = torch.sin(np.pi * x_bc)
    
    progress = st.progress(0)
    loss_hist = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Physics Loss (PDE Residual)
        u = model(x_phy, t_phy)
        
        # Gradients
        u_x = torch.autograd.grad(u, x_phy, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_phy, torch.ones_like(u_x), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_phy, torch.ones_like(u), create_graph=True)[0]
        
        residual = u_t - alpha * u_xx
        loss_phy = torch.mean(residual**2)
        
        # 2. Boundary Loss
        u_pred_bc = model(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        # Total Loss
        loss = loss_phy + loss_bc
        loss.backward()
        optimizer.step()
        
        loss_hist.append(loss.item())
        if epoch % 10 == 0:
            progress.progress((epoch+1)/epochs)
            
    st.success("Training Complete!")
    st.line_chart(loss_hist)
    
    # --- Visualize Solution ---
    st.subheader("Temperature Heatmap u(x, t)")
    
    # Grid for Plotting
    x_eval = np.linspace(-1, 1, 100)
    t_eval = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_eval, t_eval)
    
    X_torch = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    T_torch = torch.tensor(T.flatten()[:, None], dtype=torch.float32)
    
    with torch.no_grad():
        U_pred = model(X_torch, T_torch).numpy().reshape(100, 100)
        
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('none')
    c = ax.contourf(T, X, U_pred, 50, cmap='inferno')
    plt.colorbar(c)
    ax.set_xlabel("Time (t)", color='white')
    ax.set_ylabel("Space (x)", color='white')
    ax.tick_params(colors='white')
    ax.set_title(f"Heat Diffusion (alpha={alpha})", color='white')
    st.pyplot(fig)
    plt.close(fig)

st.markdown("""
**Concept**:
Instead of a grid solver (like Finite Difference), we train a neural network $u(x,t)$ to satisfy the PDE derivatives directly. 
This allows "Mesh-Free" simulation!
""")
