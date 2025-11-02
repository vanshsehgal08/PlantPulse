import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is importable first
root = Path(__file__).resolve().parents[1]  # project root
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Import shared utilities
from frontend.utils.sidebar import render_sidebar
from frontend.utils.ui import load_css


def configure_page() -> None:
    st.set_page_config(
        page_title="Plant Pulse",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Load shared CSS for consistent theming
    load_css()


def render_home() -> None:
    # Hero Section
    st.markdown("""
    <div class="hero-section fade-in">
        <h1 class="hero-title">Plant Pulse</h1>
        <p class="hero-subtitle">AI-Powered Plant Disease Detection & Management</p>
        <p>Advanced machine learning technology for accurate disease identification, expert guidance, and actionable insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("### Core Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔬</span>
            <h3 class="feature-title">Smart Detection</h3>
            <p class="feature-description">Upload leaf images for instant AI-powered disease identification with confidence scores and visual explanations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">💬</span>
            <h3 class="feature-title">Expert Chat</h3>
            <p class="feature-description">Ask questions about disease causes, symptoms, treatments, and prevention strategies with AI-powered expertise.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <h3 class="feature-title">Visual Insights</h3>
            <p class="feature-description">Get Grad-CAM visualizations showing which parts of the leaf indicate disease, providing transparent AI explanations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Action Section
    st.markdown("### Get Started")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <a href="/Predict" target="_self" style="text-decoration: none; color: inherit; display: block;">
            <div class="clickable-card">
                <div class="card-icon">🔬</div>
                <h3 class="card-title">Disease Prediction</h3>
                <p class="card-description">Upload leaf images to detect diseases instantly with detailed analysis</p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href="/Chat" target="_self" style="text-decoration: none; color: inherit; display: block;">
            <div class="clickable-card">
                <div class="card-icon">💬</div>
                <h3 class="card-title">Expert Consultation</h3>
                <p class="card-description">Chat with our AI expert about plant health and disease management</p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("### Why Plant Pulse")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4 class="feature-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">AI-Powered</h4>
            <p class="feature-description" style="font-size: 0.9rem; margin: 0;">Advanced machine learning models trained on thousands of plant images for accurate disease detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4 class="feature-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">Instant Results</h4>
            <p class="feature-description" style="font-size: 0.9rem; margin: 0;">Get predictions and expert guidance in seconds, enabling quick decision-making</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4 class="feature-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">Comprehensive Knowledge</h4>
            <p class="feature-description" style="font-size: 0.9rem; margin: 0;">Deep expertise covering disease identification, treatment options, and prevention strategies</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-tertiary); padding: 2rem 1rem;">
        <p style="margin-bottom: 0.5rem;">
            <strong style="color: var(--primary-light); font-weight: 600;">Plant Pulse</strong> 
            <span style="color: var(--text-muted);">•</span> 
            <span>Powered by AI</span>
        </p>
        <p style="font-size: 0.875rem; color: var(--text-muted); margin: 0;">
            Helping farmers and gardeners protect their plants since 2024
        </p>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    configure_page()
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()


