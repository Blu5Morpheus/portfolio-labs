# Deployment Guide

Since we designed these as **Streamlit** apps, the easiest way to deploy them for free is **Render** (as originally planned) or **Streamlit Community Cloud**.

(Note: "Fiverr" is a freelance marketplace, so I assume you meant **Render** or similar cloud host! If you are packaging this to *sell* on Fiverr, these instructions will help your buyer.)

## Option 1: Render (Recommended)

1.  **Push to GitHub**: Create a repository for `portfolio-labs` and push your code.
2.  **Create Web Service**:
    *   Go to dashboard.render.com
    *   Select "New Web Service"
    *   Connect your repo.
3.  **Configure**:
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**:
        ```bash
        streamlit run apps/chaos_lorenz/app.py --server.port $PORT --server.address 0.0.0.0
        ```
    *   *Note: You will need to create a separate Web Service for each app (Lorenz, Clifford, etc.) or merge them into one multi-page app.*

## Option 2: Streamlit Community Cloud

1.  **Push to GitHub**.
2.  **Sign in** to share.streamlit.io.
3.  **Deploy**:
    *   Select Repository.
    *   Select Branch (main).
    *   Select Main File Path (e.g., `apps/chaos_lorenz/app.py`).
    *   Click **Deploy**.

## Linking to Portfolio
Once deployed, get the URL (e.g., `https://chaos-lorenz.onrender.com`) and update `index.html`:

```html
<!-- Update the onclick event -->
<a href="https://chaos-lorenz.onrender.com" target="_blank" class="btn-launch">LAUNCH DEMO</a>
```
