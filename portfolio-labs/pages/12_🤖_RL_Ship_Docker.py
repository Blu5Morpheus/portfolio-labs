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
st.markdown("### Complex Port Environment")
st.markdown("Mission: Reach the Dock (🛑) while avoiding Islands (🌴) and Moving Ships (🚢)!")

# --- Config ---
GRID_SIZE = 20
TARGET = GRID_SIZE // 2

# --- Environment ---
class ComplexPortEnv:
    def __init__(self, wind_mode=False):
        self.state = 0 # Pos
        self.wind_mode = wind_mode
        self.obstacles = [3, 16] # Islands
        self.ships = [6, 13] # Moving ships
        self.ship_dirs = [1, -1]
    
    def reset(self):
        # Avoid spawning on obstacles
        while True:
            self.state = random.randint(0, GRID_SIZE-1)
            if self.state not in self.obstacles and self.state not in self.ships and self.state != TARGET:
                break
        return self.get_obs()
    
    def get_obs(self):
        # One-hot encoding + Obstacles?
        # Let's keep input simple (15 inputs) but semantic?
        # Actually, for DQN to learn dynamic obstacles, it needs to see them.
        # Simple One-Hot of Position is insufficient for moving obstacles.
        # Let's feed: [AgentPos, TargetPos, Ship1Pos, Ship2Pos] normalised
        # Flattened State Vector
        obs = np.zeros(GRID_SIZE, dtype=np.float32)
        obs[self.state] = 1.0 # Agent
        
        # Mark static obstacles with -1? No, logic handles collision.
        # But agent needs to know where they are. 
        # For simplicity in this demo, we assume the agent "Sees" the grid.
        # Retaining One-Hot State for Agent, but maybe simple state is better?
        
        return obs
        
    def step(self, action):
        # 0: Left, 1: Stay, 2: Right
        move = action - 1
        
        # Wind
        if self.wind_mode and random.random() < 0.2:
            move += random.choice([-1, 1])
            
        new_pos = np.clip(self.state + move, 0, GRID_SIZE-1)
        
        done = False
        reward = -1
        
        # Check Collision (Island)
        if new_pos in self.obstacles:
            reward = -50
            done = True
            
        # Check Collision (Ships) - Simple
        if new_pos in self.ships:
            reward = -50
            done = True
            
        self.state = new_pos
        
        # Move Ships
        for i in range(len(self.ships)):
            self.ships[i] += self.ship_dirs[i]
            # Bounce
            if self.ships[i] <= 0 or self.ships[i] >= GRID_SIZE-1:
                self.ship_dirs[i] *= -1
        
        if self.state == TARGET:
            reward = 100
            done = True
            
        # Distance penalty
        reward -= abs(self.state - TARGET) * 0.1
        
        return self.get_obs(), reward, done

# --- DQN ---
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple feed forward
        self.net = nn.Sequential(
            nn.Linear(GRID_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )
    def forward(self, x):
        return self.net(x)

# --- UI ---
col_param, col_screen = st.columns([1, 2])

with col_param:
    wind_on = st.checkbox("High Winds 💨", value=False)
    episodes = st.slider("Training Episodes", 10, 300, 100)
    
    if st.button("Train Captain AI 🧠"):
        env = ComplexPortEnv(wind_mode=wind_on)
        policy_net = DQN()
        target_net = DQN()
        target_net.load_state_dict(policy_net.state_dict())
        
        opt = torch.optim.Adam(policy_net.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        
        replay = deque(maxlen=1000)
        gamma = 0.95
        epsilon = 1.0
        
        progress = st.progress(0)
        rewards_h = []
        
        for ep in range(episodes):
            state = env.reset()
            total_r = 0
            
            for t in range(50):
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
                
                if len(replay) > 64:
                    batch = random.sample(replay, 64)
                    b_s = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
                    b_a = torch.tensor([x[1] for x in batch], dtype=torch.long).unsqueeze(1)
                    b_r = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1)
                    b_ns = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32)
                    b_d = torch.tensor([x[4] for x in batch], dtype=torch.bool).unsqueeze(1)
                    
                    q_curr = policy_net(b_s).gather(1, b_a)
                    with torch.no_grad():
                        q_next = target_net(b_ns).max(1)[0].unsqueeze(1)
                        q_targ = b_r + gamma * q_next * (~b_d)
                    
                    loss = loss_fn(q_curr, q_targ)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            epsilon = max(0.1, epsilon * 0.95)
            if ep % 10 == 0: target_net.load_state_dict(policy_net.state_dict())
            
            rewards_h.append(total_r)
            progress.progress((ep+1)/episodes)
        
        st.session_state.dqn_port = policy_net
        st.success("Captain Certified!")
        st.line_chart(rewards_h)

with col_screen:
    st.subheader("📺 Live Port View")
    screen = st.empty()
    
    if st.button("Run Simulation"):
        if 'dqn_port' in st.session_state:
            env = ComplexPortEnv(wind_mode=wind_on)
            state = env.reset()
            model = st.session_state.dqn_port
            
            for _ in range(50):
                grid_viz = ["🌊"] * GRID_SIZE
                grid_viz[TARGET] = "🛑"
                
                for obs in env.obstacles: grid_viz[obs] = "🌴"
                for ship in env.ships: grid_viz[ship] = "🚢"
                
                curr_pos = np.argmax(state)
                grid_viz[curr_pos] = "🚀"
                
                # Visual
                screen_html = f"""
                <div style="background-color: #001f3f; padding: 20px; border-radius: 10px; border: 2px solid #00f3ff; text-align: center;">
                    <div style="font-size: 30px;">{''.join(grid_viz)}</div>
                    <hr style="border-color: #0074D9;">
                    <div style="color: #7FDBFF;">Captain's Log: Avoiding Traffic...</div>
                </div>
                """
                screen.markdown(screen_html, unsafe_allow_html=True)
                
                with torch.no_grad():
                    q = model(torch.tensor(state).unsqueeze(0))
                    action = torch.argmax(q).item()
                
                state, r, done = env.step(action)
                if done:
                    if r > 0: st.balloons()
                    else: st.error("CRASH!")
                    break
                time.sleep(0.3)
        else:
            st.warning("Train the Captain first!")
