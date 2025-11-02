import streamlit as st


def render_sidebar() -> None:
    """Render custom sidebar with branding and navigation"""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem 0; border-bottom: 2px solid var(--border); margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 2rem;">🌿</span>
                <h1 style="margin: 0; font-size: 1.5rem; font-weight: 700; color: var(--primary-light); letter-spacing: -0.025em;">
                    Plant Pulse
                </h1>
            </div>
            <p style="margin: 0; color: var(--text-secondary); font-size: 0.875rem; line-height: 1.4;">
                Disease Detection & Management
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        # Use Streamlit's page_link for same-window navigation
        # Paths are relative to project root - Streamlit handles resolution
        st.page_link("app.py", label="Home", icon="🏠", use_container_width=True)
        st.page_link("pages/1_Predict.py", label="Predict", icon="🔬", use_container_width=True)
        st.page_link("pages/2_Chat.py", label="Chat", icon="💬", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### Quick Info")
        st.markdown("""
        <div style="background: var(--bg-primary); border-radius: var(--radius-lg); padding: 1rem; border: 1px solid var(--border);">
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.25rem;">📊</span>
                    <div>
                        <div style="font-size: 0.75rem; color: var(--text-tertiary);">Supported Formats</div>
                        <div style="font-size: 0.875rem; font-weight: 600; color: var(--text-primary);">JPG, PNG</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.25rem;">🌱</span>
                    <div>
                        <div style="font-size: 0.75rem; color: var(--text-tertiary);">Detection Accuracy</div>
                        <div style="font-size: 0.875rem; font-weight: 600; color: var(--primary-light);">High Precision</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.25rem;">⚡</span>
                    <div>
                        <div style="font-size: 0.75rem; color: var(--text-tertiary);">Processing Speed</div>
                        <div style="font-size: 0.875rem; font-weight: 600; color: var(--text-primary);">Instant Results</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-top: 2px solid var(--border); margin-top: 2rem;">
            <p style="font-size: 0.75rem; color: var(--text-tertiary); margin: 0;">
                © 2024 Plant Pulse
            </p>
        </div>
        """, unsafe_allow_html=True)

