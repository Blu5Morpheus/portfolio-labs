import streamlit as st
import numpy as np
import time
import torch
import torch.nn as nn
import random
from collections import deque

st.set_page_config(page_title="Deep RL Docker", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Deep RL Ship Docker")
st.markdown("### Deep Q-Network (DQN) + Adversarial Wind")
st.markdown("The agent uses a Neural Network to interpret the grid state and land the ship 🚀.")

# --- Config ---
GRID_SIZE = 15
TARGET = GRID_SIZE // 2

# --- Environment ---
class DeepDockEnv:
    def __init__(self, wind_mode=False):
        self.state = 0 # Pos
        self.wind_mode = wind_mode
    
    def reset(self):
        self.state = random.randint(0, GRID_SIZE-1)
        return self.get_obs()
    
    def get_obs(self):
        # One-hot encoding for NN
        obs = np.zeros(GRID_SIZE, dtype=np.float32)
        obs[self.state] = 1.0
        return obs
        
    def step(self, action):
        # 0: Left, 1: Stay, 2: Right
        move = action - 1
        
        # Wind!
        if self.wind_mode and random.random() < 0.3:
            wind = random.choice([-1, 1])
            move += wind
            
        self.state = np.clip(self.state + move, 0, GRID_SIZE-1)
        
        done = False
        reward = -1
        
        if self.state == TARGET:
            reward = 50
            done = True
            
        # Distance penalty
        reward -= abs(self.state - TARGET) * 0.1
        
        return self.get_obs(), reward, done

# --- DQN ---
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Q-Values for L, S, R
        )
    def forward(self, x):
        return self.net(x)

# --- UI ---
col_param, col_screen = st.columns([1, 2])

with col_param:
    wind_on = st.checkbox("Adversarial Wind 💨", value=False)
    episodes = st.slider("Episodes", 10, 200, 50)
    
    if st.button("Train DQN 🧠"):
        env = DeepDockEnv(wind_mode=wind_on)
        policy_net = DQN()
        target_net = DQN()
        target_net.load_state_dict(policy_net.state_dict())
        
        opt = torch.optim.Adam(policy_net.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        replay = deque(maxlen=500)
        gamma = 0.9
        epsilon = 0.5
        
        progress = st.progress(0)
        rewards_h = []
        
        for ep in range(episodes):
            state = env.reset()
            total_r = 0
            
            for t in range(20):
                # E-Greedy
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        q = policy_net(torch.tensor(state).unsqueeze(0))
                        action = torch.argmax(q).item()
                
                next_state, r, done = env.step(action)
                
                replay.append((state, action, r, next_state, done))
                state = next_state
                total_r += r
                
                if done: break
                
                # Train Step
                if len(replay) > 32:
                    batch = random.sample(replay, 32)
                    b_s = torch.tensor([x[0] for x in batch])
                    b_a = torch.tensor([x[1] for x in batch]).unsqueeze(1)
                    b_r = torch.tensor([x[2] for x in batch]).unsqueeze(1)
                    b_ns = torch.tensor([x[3] for x in batch])
                    b_d = torch.tensor([x[4] for x in batch]).unsqueeze(1)
                    
                    q_curr = policy_net(b_s).gather(1, b_a)
                    with torch.no_grad():
                        q_next = target_net(b_ns).max(1)[0].unsqueeze(1)
                        q_targ = b_r + gamma * q_next * (~b_d)
                    
                    loss = loss_fn(q_curr, q_targ)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            # Decay Epsilon
            epsilon *= 0.95
            if ep % 10 == 0: target_net.load_state_dict(policy_net.state_dict())
            
            rewards_h.append(total_r)
            progress.progress((ep+1)/episodes)
        
        st.session_state.dqn = policy_net
        st.success(f"DQN Trained! Wind: {wind_on}")
        st.line_chart(rewards_h)

with col_screen:
    st.subheader("📺 Neural Nav System")
    screen = st.empty()
    
    if st.button("Run Simulation"):
        if 'dqn' in st.session_state:
            env = DeepDockEnv(wind_mode=wind_on)
            state = env.reset()
            
            model = st.session_state.dqn
            
            for _ in range(20):
                # Vis
                # Create a visual grid
                grid_viz = ["⬜"] * GRID_SIZE
                grid_viz[TARGET] = "🛑"
                
                # Agent pos from one-hot
                curr_pos = np.argmax(state)
                grid_viz[curr_pos] = "🚀"
                
                # Q-Values (What is the brain thinking?)
                with torch.no_grad():
                    q_vals = model(torch.tensor(state).unsqueeze(0)).numpy()[0]
                
                screen_html = f"""
                <div style="background-color: #000; padding: 20px; border-radius: 10px; border: 2px solid #00f3ff; text-align: center;">
                    <div style="font-size: 30px; letter-spacing: 5px;">{''.join(grid_viz)}</div>
                    <hr style="border-color: #333;">
                    <div style="display: flex; justify-content: space-around; color: #00f3ff; font-family: monospace;">
                        <div>LEFT: {q_vals[0]:.2f}</div>
                        <div>STAY: {q_vals[1]:.2f}</div>
                        <div>RIGHT: {q_vals[2]:.2f}</div>
                    </div>
                </div>
                """
                screen.markdown(screen_html, unsafe_allow_html=True)
                
                # Act
                action = np.argmax(q_vals)
                state, _, done = env.step(action)
                
                if done:
                    st.balloons()
                    break
                
                time.sleep(0.3)
        else:
            st.warning("Train the Brain first!")
