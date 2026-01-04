import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Engineering Blog", page_icon="📝", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 Raven Labs Engineering Log")
st.markdown("### Technical Specifications & Design Docs")

# Path to blog folder (relative to this file)
# This file is in portfolio-labs/pages/
# Blog is in portfolio-web/blog/
# So we go up two levels.

base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_path))
blog_dir = os.path.join(project_root, "blog")

if os.path.exists(blog_dir):
    files = [f for f in os.listdir(blog_dir) if f.endswith(".html")]
    
    if files:
        selected_post = st.sidebar.selectbox("Select Article", files, index=0)
        
        post_path = os.path.join(blog_dir, selected_post)
        
        with open(post_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Render
        st.markdown(f"**Viewing: {selected_post}**")
        st.markdown("---")
        
        # We use a scrolling iframe
        components.html(html_content, height=1200, scrolling=True)
        
    else:
        st.info("No blog posts found in /blog directory.")
else:
    st.error(f"Blog directory not found at {blog_dir}")
