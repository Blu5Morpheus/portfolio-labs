# Deployment Guide

You have been upgraded to the **Multi-Page App** structure!
All your simulations are now part of a single unified "Raven Physics Labs" application.

## 🚀 Easy Deployment on Render

### 1. The Structure
Your code is now organized for a single deployment:
- Entry Point: `portfolio-labs/Home.py`
- Pages: `portfolio-labs/pages/`
- Utils: `portfolio-labs/utils/`

### 2. Configure Render
Since this is now one app, you only need **ONE** Web Service (instead of 4).

1.  **Push to GitHub**: Push the `portfolio-labs` directory.
2.  **Create Web Service**:
    *   Connect your repo.
    *   **Root Directory**: `portfolio-labs` (Important! This sets the context)
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `streamlit run Home.py`

### 3. Adding New Apps (Future Proofing)
To add a new simulation in the future:
1.  Create a new file in `portfolio-labs/pages/` (e.g., `5_⚡_New_Sim.py`).
2.  Streamlit will automatically detect it and add it to the sidebar navigation!

## Linking to Portfolio
Update `index.html` to point to your single service URL (plus deep links if you want):

```html
<!-- Example: Launch the Hub -->
<a href="https://your-raven-lab.onrender.com" target="_blank" class="btn-launch">LAUNCH HUB</a>

<!-- Example: Deep Link to Lorenz -->
<a href="https://your-raven-lab.onrender.com/Lorenz_Chaos" target="_blank" class="btn-launch">LAUNCH CHAOS</a>
```
