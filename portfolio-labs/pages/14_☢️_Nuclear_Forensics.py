import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from io import BytesIO

matplotlib.use('Agg')

st.set_page_config(page_title="Nuclear Forensics", page_icon="☢️", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .report-box {
        border: 1px solid #00f3ff;
        padding: 20px;
        background-color: #000;
        font-family: 'Courier New', monospace;
        color: #00f3ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("☢️ Nuclear Yield Forensics")
st.markdown("### AI Analysis of Blast Wave Dynamics")
st.markdown("Using the **Sedov-Taylor Solution** ($R \sim t^{2/5}$) to reverse-engineer energy yield from fireball expansion data.")

# --- Physics Engine ---
def sedov_radius(t, E_kt, rho=1.2):
    # E in Joules. 1 kT = 4.184e12 J
    E_joules = E_kt * 4.184e12
    # R(t) = C * (E * t^2 / rho)^(1/5)
    # C is approx 1.03
    return 1.0 * (E_joules * t**2 / rho)**0.2

# --- AI Model (The estimator) ---
class BlastNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Sequence of Radius points (normalized) -> Output: Yield
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Feature: Training
if 'blast_model' not in st.session_state:
    st.session_state.blast_model = BlastNet()
    st.session_state.model_trained = False

col_sim, col_ai = st.columns(2)

with col_sim:
    st.subheader("1. Scenario Generator")
    real_yield = st.number_input("True Yield (Kilotons)", 1.0, 100.0, 20.0)
    
    # Generate Synthetic Data
    t_points = np.linspace(0.01, 1.0, 10) # 10 ms to 1s?
    # Trinity test was ~100m in 0.1s. 
    # Let's say t in seconds.
    # Yield 20kT -> Huge radius.
    # Sedov valid for Early times.
    
    # Add noise
    noise_level = st.slider("Measurement Noise", 0.0, 0.1, 0.02)
    
    r_pure = sedov_radius(t_points, real_yield)
    r_noisy = r_pure * (1 + np.random.normal(0, noise_level, len(t_points)))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#111')
    ax.scatter(t_points, r_noisy, color='#b026ff', label='Observed Data')
    ax.plot(t_points, r_pure, color='gray', linestyle='--', alpha=0.5, label='Theoretical Fit')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Radius (m)", color='white')
    ax.loglog()
    ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white', which='both')
    st.pyplot(fig)
    plt.close(fig)

with col_ai:
    st.subheader("2. AI Estimator")
    
    if st.button("Train AI on Synthetic Curves"):
        st.info("Training on 1000 randomized blasts...")
        opt = torch.optim.Adam(st.session_state.blast_model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        
        loss_h = []
        prog = st.progress(0)
        
        for ep in range(200):
            # Batch
            batch_size = 32
            yields = torch.rand(batch_size, 1) * 100 # 0-100 kT
            
            # Generate Curves
            radii_batch = []
            for y in yields:
                 r = sedov_radius(t_points, y.item())
                 # Normalize radius: Divide by R(t=1, E=50) approx for scaling?
                 # Simple scaling: Divide by 1000m
                 r = r / 1000.0
                 radii_batch.append(r)
            
            X = torch.tensor(np.array(radii_batch), dtype=torch.float32)
            Y = yields
            
            pred = st.session_state.blast_model(X)
            loss = loss_fn(pred, Y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_h.append(loss.item())
            if ep % 50 == 0: prog.progress((ep+1)/200)
            
        st.session_state.model_trained = True
        st.success("AI Trained!")
        st.line_chart(loss_h)
        
    if st.session_state.model_trained:
        # Predict current scenario
        # Prepare input
        x_in = r_noisy / 1000.0
        x_tensor = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_yield = st.session_state.blast_model(x_tensor).item()
            
        st.metric("AI Predicted Yield", f"{pred_yield:.2f} kT", delta=f"{pred_yield - real_yield:.2f} kT")
        
        error = abs(pred_yield - real_yield) / real_yield * 100
        if error < 10:
            st.success("ACCURATE ESTIMATION")
        else:
            st.warning("HIGH DEVIATION")
    else:
        st.warning("Train Model First")

# --- Report Generator ---
st.markdown("---")
st.subheader("📁 Classified Forensics Report")
if st.button("Generate Report"):
    report_text = f"""
    TOP SECRET // RAVEN LABS FORENSICS
    ==================================
    EVENT ANALYSIS REPORT
    DATE: 2026-01-04
    
    DATA ANALYSIS: SEDOV-TAYLOR SOLVER
    ----------------------------------
    OBSERVED TIMESTEPS: {len(t_points)}
    MAX RADIUS: {np.max(r_noisy):.2f} m
    
    AI ESTIMATION RESULT
    --------------------
    PREDICTED YIELD: {pred_yield if 'pred_yield' in locals() else 'N/A'} kT
    CONFIDENCE: {'HIGH' if 'pred_yield' in locals() and error < 10 else 'LOW'}
    
    NOTES: 
    Blast wave expansion follows t^(2/5) power law.
    Atmospheric density assumed nominal (1.2 kg/m^3).
    
    // END REPORT
    """
    st.code(report_text)
