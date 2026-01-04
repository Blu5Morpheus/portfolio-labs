import streamlit as st
import numpy as np
import time
import pandas as pd

st.set_page_config(page_title="RL Ship Docker", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 RL Ship Docker")
st.markdown("### Reinforcement Learning Demo")
st.markdown("Training a Q-Learning agent to dock a spaceship 🚀 on the pad 🛑 without using fuel efficiently.")

# --- Environment ---
class DockingEnv:
    def __init__(self, size=10):
        self.size = size
        self.target = size // 2
        self.agent_pos = 0 # Start left
        
    def reset(self):
        self.agent_pos = np.random.choice([0, self.size-1]) # Random start edge
        return self.agent_pos
    
    def step(self, action):
        # Actions: 0=Left, 1=Stay, 2=Right
        move = action - 1
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size-1)
        
        reward = -1 # Fuel cost per step
        done = False
        
        if self.agent_pos == self.target:
            reward = 20 # Landed!
            done = True
            
        return self.agent_pos, reward, done

# --- Training UI ---
episodes = st.slider("Training Episodes", 10, 200, 50)
if st.button("Train Agent 🧠"):
    env = DockingEnv(size=11)
    q_table = np.zeros((env.size, 3)) # State x Action
    
    # Q-Params
    alpha = 0.1 # Learning Rate
    gamma = 0.9 # Discount
    epsilon = 0.1 # Exploration
    
    rewards_hist = []
    
    # Training Loop
    progress = st.progress(0)
    
    with st.empty():
        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            steps = 0
            while not done and steps < 20:
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, 3)
                else:
                    action = np.argmax(q_table[state])
                
                next_state, reward, done = env.step(action)
                
                # Q-Update
                old_val = q_table[state, action]
                next_max = np.max(q_table[next_state])
                q_table[state, action] = old_val + alpha * (reward + gamma * next_max - old_val)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            rewards_hist.append(total_reward)
            if ep % 5 == 0:
                progress.progress((ep+1)/episodes)
                st.write(f"Episode {ep}: Reward {total_reward}")
                
    st.success("Training Complete!")
    st.line_chart(rewards_hist)
    
    # --- Demo Run ---
    st.subheader("Live Test Flight")
    state = env.reset()
    env.agent_pos = 0 # Force start left
    state = 0
    
    display_area = st.empty()
    
    # Render Loop
    for _ in range(15):
        # Vis
        grid = ["." for _ in range(env.size)]
        grid[env.target] = "🛑"
        grid[state] = "🚀"
        vis_str = " ".join(grid)
        display_area.markdown(f"## {vis_str}")
        
        if state == env.target:
            display_area.markdown(f"## {vis_str} -> **DOCKED!** 🎉")
            break
            
        action = np.argmax(q_table[state])
        state, _, _ = env.step(action)
        time.sleep(0.3)

st.markdown("""
**How it works**:
The table learns values $Q(state, action)$. 
- **Start**: Moves randomly.
- **End**: Knows exactly which direction moves closer to the center 🛑 to maximize the reward (+20).
""")
