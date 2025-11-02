from pathlib import Path
import streamlit as st


def load_css() -> None:
    """Load the shared CSS file for consistent theming across all pages"""
    css_path = Path(__file__).resolve().parents[1] / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Additional JavaScript to aggressively hide default navigation
    st.markdown("""
    <script>
    // Aggressively hide default Streamlit sidebar navigation
    function hideDefaultNav() {
        // Target all possible navigation elements
        const selectors = [
            'nav[data-testid="stSidebarNav"]',
            'nav[data-testid="stSidebarNav"] > div',
            'nav[data-testid="stSidebarNav"] ul',
            'nav[data-testid="stSidebarNav"] li',
            'nav[data-testid="stSidebarNav"] a',
            'section[data-testid="stSidebar"] nav',
            'section[data-testid="stSidebar"] > div > nav',
            'section[data-testid="stSidebar"] > div > div > nav'
        ];
        
        selectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => {
                if (el && el.getAttribute('data-testid') === 'stSidebarNav') {
                    el.style.display = 'none';
                    el.style.visibility = 'hidden';
                    el.style.height = '0';
                    el.style.overflow = 'hidden';
                    el.style.opacity = '0';
                    el.style.pointerEvents = 'none';
                }
            });
        });
        
        // Hide parent sections containing the nav
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            const sections = sidebar.querySelectorAll('section');
            sections.forEach(section => {
                const nav = section.querySelector('nav[data-testid="stSidebarNav"]');
                if (nav) {
                    section.style.display = 'none';
                }
            });
            
            // Also check direct div children
            const divs = sidebar.querySelectorAll('div');
            divs.forEach(div => {
                const nav = div.querySelector('nav[data-testid="stSidebarNav"]');
                if (nav && nav.parentElement === div) {
                    div.style.display = 'none';
                }
            });
        }
        
        // Remove any links with text "app", "Predict", "Chat" that are not our custom nav
        const allLinks = document.querySelectorAll('section[data-testid="stSidebar"] a, section[data-testid="stSidebar"] button');
        allLinks.forEach(link => {
            const text = link.textContent.trim().toLowerCase();
            const parent = link.closest('nav[data-testid="stSidebarNav"]');
            const isCustomNav = link.closest('[data-testid="stPageLink-None"]');
            
            // Hide if it's in default nav and matches app/predict/chat
            if (parent && !isCustomNav && (text === 'app' || text === 'predict' || text === 'chat')) {
                parent.style.display = 'none';
                link.style.display = 'none';
                link.style.visibility = 'hidden';
                if (link.parentElement) {
                    link.parentElement.style.display = 'none';
                }
            }
        });
        
        // Also find and hide list items containing these texts
        const allItems = document.querySelectorAll('section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] div');
        allItems.forEach(item => {
            const text = item.textContent.trim().toLowerCase();
            if ((text === 'app' || text === 'predict' || text === 'chat') && item.closest('nav[data-testid="stSidebarNav"]')) {
                item.style.display = 'none';
                item.style.visibility = 'hidden';
            }
        });
        
        // Nuclear option: Hide the first direct child section/div that contains nav but not our branding
        if (sidebar) {
            const children = Array.from(sidebar.children);
            children.forEach((child, index) => {
                // Skip our custom content (usually has h1 with "Plant Pulse" or "Navigation" heading)
                const hasBranding = child.querySelector('h1, h3')?.textContent.toLowerCase().includes('plant pulse');
                const hasNavHeading = child.textContent.includes('NAVIGATION') || child.textContent.includes('Navigation');
                
                // If it's an early child without branding, check if it has the default nav
                if (index < 2 && !hasBranding && !hasNavHeading) {
                    const hasDefaultNav = child.querySelector('nav[data-testid="stSidebarNav"]');
                    if (hasDefaultNav) {
                        child.style.display = 'none';
                        child.style.visibility = 'hidden';
                        child.style.height = '0';
                        child.style.overflow = 'hidden';
                    }
                }
            });
        }
    }
    
    // Run multiple times to catch all render scenarios
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideDefaultNav);
    } else {
        hideDefaultNav();
    }
    window.addEventListener('load', hideDefaultNav);
    
    // Use MutationObserver to catch dynamically added elements
    const observer = new MutationObserver(hideDefaultNav);
    const sidebar = document.querySelector('section[data-testid="stSidebar"]');
    if (sidebar) {
        observer.observe(sidebar, { childList: true, subtree: true });
    }
    
    // Also run with delays
    setTimeout(hideDefaultNav, 50);
    setTimeout(hideDefaultNav, 100);
    setTimeout(hideDefaultNav, 300);
    setTimeout(hideDefaultNav, 500);
    setTimeout(hideDefaultNav, 1000);
    </script>
    """, unsafe_allow_html=True)

