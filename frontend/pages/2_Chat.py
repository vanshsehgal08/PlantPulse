import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


def is_plant_related_query(query: str) -> bool:
    """Check if query is related to plants/diseases."""
    query_lower = query.lower()
    
    # Plant-related keywords
    plant_keywords = [
        'plant', 'disease', 'leaf', 'symptom', 'treat', 'prevent', 'crop', 'garden',
        'fungus', 'bacterial', 'virus', 'pest', 'fungicide', 'pesticide', 'blight',
        'rot', 'spot', 'mildew', 'rust', 'scab', 'wilt', 'mosaic', 'yellow',
        'tomato', 'potato', 'apple', 'corn', 'grape', 'pepper', 'cherry', 'peach',
        'strawberry', 'blueberry', 'orange', 'soybean', 'squash', 'raspberry',
        'agriculture', 'agronomy', 'horticulture', 'farming', 'farmer', 'gardener',
        'organic', 'chemical', 'management', 'control', 'cure', 'heal', 'infection'
    ]
    
    # Check if query contains plant-related keywords
    return any(keyword in query_lower for keyword in plant_keywords)


def group_messages_into_conversations(messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Group messages into Q&A pairs."""
    conversations = []
    current_question = None
    
    for msg in messages:
        if msg["role"] == "user":
            current_question = msg["content"]
        elif msg["role"] == "assistant" and current_question:
            conversations.append((current_question, msg["content"]))
            current_question = None
    
    # Handle case where there's a question but no answer yet
    if current_question:
        conversations.append((current_question, None))
    
    return conversations


def build_system_prompt(detected_disease: Optional[str] = None) -> str:
    """Build system prompt with context about detected disease."""
    base_prompt = (
        "You are Plant Pulse, an expert agronomy assistant specializing exclusively in plant disease diagnosis, "
        "management, and agricultural best practices. Your expertise covers:\n\n"
        "**CORE RESPONSIBILITIES:**\n"
        "- Plant disease identification, causes, and contributing factors\n"
        "- Early symptom recognition and diagnostic guidance\n"
        "- Treatment options (organic, chemical, and integrated pest management)\n"
        "- Prevention strategies and cultural practices\n"
        "- Long-term disease management and crop health\n"
        "- Plant nutrition and soil health related to disease resistance\n\n"
        "**RESPONSE GUIDELINES:**\n"
        "- Provide accurate, science-based information\n"
        "- Use clear, concise language with structured bullet points\n"
        "- Include practical, actionable advice that users can implement\n"
        "- Always mention safety considerations for chemical treatments\n"
        "- Reference specific disease names, symptoms, and treatments when possible\n"
        "- If asked about non-plant topics, politely redirect to plant-related questions\n"
        "- Format responses with proper markdown (headers, bullets, emphasis)\n"
        "- Keep responses focused and relevant (2-4 paragraphs maximum)\n\n"
        "**STAY FOCUSED:** Only answer questions related to plants, crops, diseases, gardening, or agriculture. "
        "If asked about unrelated topics, politely decline and offer plant-related alternatives.\n\n"
    )
    
    if detected_disease:
        # Clean up disease name for better context
        disease_name = detected_disease.replace("___", " ").replace("_", " ").strip()
        base_prompt += (
            f"**IMPORTANT CONTEXT:** The user has recently detected '{disease_name}' in their plant analysis. "
            f"Prioritize specific information about this disease in your responses. When relevant, reference this "
            f"disease and provide targeted advice about its causes, symptoms, treatments, and prevention.\n\n"
        )
    
    base_prompt += (
        "Now, provide a helpful, accurate response to the user's question. "
        "If the question is not plant-related, politely explain that you specialize in plant diseases and "
        "ask if they have any plant health questions.\n"
    )
    
    return base_prompt


def get_gemini_client():
    """Get Google Gemini API client."""
    # Check Streamlit secrets first, then environment variable
    api_key = None
    try:
        # Try Streamlit secrets (recommended for production)
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    
    if not api_key:
        # Fallback to environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Use Gemini 2.0 Flash - faster and more efficient
        return genai.GenerativeModel('gemini-2.0-flash-exp')
    except ImportError:
        st.error("Please install google-generativeai: pip install google-generativeai")
        return None
    except Exception as e:
        st.error(f"Error initializing Gemini API: {str(e)}")
        return None


def ai_answer(model, messages: List[Dict[str, str]], detected_disease: Optional[str] = None) -> str:
    """Get AI answer using Gemini API."""
    if model is None:
        last_query = messages[-1]["content"].lower()
        
        # Check if query is plant-related
        if not is_plant_related_query(messages[-1]["content"]):
            return (
                "🌿 **I specialize in plant diseases and agricultural advice.**\n\n"
                "I can help you with:\n"
                "- Plant disease identification and symptoms\n"
                "- Treatment and prevention strategies\n"
                "- Crop management practices\n"
                "- Agricultural best practices\n\n"
                "Please ask me a question about plants, crops, or plant diseases! 🌱"
            )
        
        # Offline fallback responses for plant-related queries
        if "symptom" in last_query:
            return (
                "**Common Plant Disease Symptoms:**\n\n"
                "- **Leaf spots**: Small discolored areas that may expand\n"
                "- **Yellowing**: Chlorosis affecting leaves\n"
                "- **Wilting**: Drooping or limp leaves/stems\n"
                "- **Curling**: Distorted leaf growth\n"
                "- **Powdery coating**: White/gray fungal growth\n"
                "- **Rotting**: Soft, discolored tissue\n\n"
                "⚠️ **Note**: For accurate AI-powered responses, please configure your Gemini API key."
            )
        if "prevent" in last_query or "management" in last_query:
            return (
                "**General Disease Prevention Tips:**\n\n"
                "- Remove and destroy infected plant material\n"
                "- Improve air circulation around plants\n"
                "- Avoid overhead watering (water at base)\n"
                "- Rotate crops annually\n"
                "- Use disease-resistant varieties when available\n"
                "- Apply appropriate fungicides following label directions\n"
                "- Maintain proper plant spacing\n\n"
                "⚠️ **Note**: For detailed, disease-specific advice, please configure your Gemini API key."
            )
        return (
            "⚠️ **Gemini API not configured.**\n\n"
            "To enable full AI chat functionality:\n"
            "1. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)\n"
            "2. Set it in the 'Configure Gemini API Key' section below (or as `GEMINI_API_KEY` environment variable)\n\n"
            "Once configured, I can provide detailed, expert advice about plant diseases! 🌱"
        )
    
    try:
        # Check query relevance first
        last_query = messages[-1]["content"]
        if not is_plant_related_query(last_query):
            return (
                "🌿 **I specialize exclusively in plant diseases and agricultural topics.**\n\n"
                "I'd be happy to help you with:\n\n"
                "✅ Plant disease identification and diagnosis\n"
                "✅ Disease symptoms and early warning signs\n"
                "✅ Treatment options (organic and chemical)\n"
                "✅ Prevention strategies and best practices\n"
                "✅ Crop management and agricultural advice\n"
                "✅ Soil health and plant nutrition related to diseases\n\n"
                "**Do you have a question about plant health or disease management?** 🌱"
            )
        
        # Build conversation context with system prompt
        system_prompt = build_system_prompt(detected_disease)
        
        # Convert messages to Gemini format
        conversation_parts = []
        
        # Add system context as first message
        conversation_parts.append(system_prompt)
        
        # Add conversation history (last 10 messages to keep context manageable)
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        for msg in recent_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_parts.append(f"\n{role}: {msg['content']}")
        
        # Add current prompt
        conversation_parts.append("\nAssistant:")
        
        # Combine into prompt
        prompt = "".join(conversation_parts)
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 1500,
            }
        )
        
        result_text = response.text if response.text else "I apologize, but I couldn't generate a response. Please try again."
        
        # Ensure response is formatted nicely
        if not result_text.startswith("**") and not result_text.startswith("#"):
            # Try to format if not already formatted
            lines = result_text.split("\n")
            if len(lines) > 3:
                # Add some formatting
                formatted = "**" + lines[0] + "**\n\n" + "\n".join(lines[1:])
                return formatted
        
        return result_text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "quota" in error_msg.lower():
            return (
                "⚠️ **API Error**\n\n"
                "There was an issue with the Gemini API. Please check:\n"
                "- Your API key is valid and active\n"
                "- You have available quota\n"
                "- Your internet connection is working\n\n"
                f"Error details: {error_msg}"
            )
        return f"⚠️ **Error generating response**: {str(e)}\n\nPlease try again or check your API configuration."


def main() -> None:
    st.set_page_config(page_title="Chat · Plant Pulse", page_icon="💬", layout="wide")
    
    # Load shared CSS for consistent theming
    from frontend.utils.ui import load_css
    load_css()
    
    # Import and render sidebar
    from frontend.utils.sidebar import render_sidebar
    render_sidebar()
    
    # Custom CSS for better chat UI with carousel
    st.markdown("""
    <style>
    .chat-header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem;
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-2xl);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    .chat-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.05) 0%, transparent 100%);
        pointer-events: none;
    }
    .chat-header h1 {
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.025em;
        position: relative;
        z-index: 1;
    }
    .chat-header p {
        color: var(--text-secondary);
        font-size: 1.125rem;
        position: relative;
        z-index: 1;
        margin: 0;
    }
    .chat-carousel-container {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 1.5rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        margin-bottom: 1rem;
        position: relative;
        scrollbar-width: thin;
        scrollbar-color: var(--primary) var(--bg-primary);
        box-shadow: var(--shadow-sm);
    }
    .chat-carousel-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-carousel-container::-webkit-scrollbar-track {
        background: var(--bg-primary);
        border-radius: 4px;
    }
    .chat-carousel-container::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    .chat-carousel-container::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    .chat-carousel-container .stMarkdown {
        color: var(--text-secondary) !important;
    }
    .chat-carousel-container .stMarkdown p,
    .chat-carousel-container .stMarkdown li,
    .chat-carousel-container .stMarkdown ul,
    .chat-carousel-container .stMarkdown ol {
        color: var(--text-secondary) !important;
    }
    .chat-carousel-container .stMarkdown strong {
        color: var(--primary-light) !important;
    }
    .chat-carousel-container .stMarkdown h1,
    .chat-carousel-container .stMarkdown h2,
    .chat-carousel-container .stMarkdown h3,
    .chat-carousel-container .stMarkdown h4 {
        color: var(--primary-light) !important;
    }
    .chat-navigation {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
    }
    .nav-button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .nav-button:hover {
        background: linear-gradient(135deg, var(--primary-light), var(--primary));
        transform: translateY(-1px);
        box-shadow: var(--shadow-sm);
    }
    .nav-button:disabled {
        background: var(--bg-tertiary);
        color: var(--text-tertiary);
        cursor: not-allowed;
        opacity: 0.6;
    }
    .conversation-counter {
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.9375rem;
    }
    .question-preview {
        background: var(--bg-tertiary);
        border-left: 3px solid var(--primary);
        padding: 0.75rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .question-preview:hover {
        background: var(--bg-elevated);
        transform: translateX(4px);
        border-left-color: var(--primary-light);
    }
    .question-preview.active {
        background: var(--primary);
        color: white;
        border-left-color: var(--primary-dark);
    }
    .question-preview-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .question-preview.active .question-preview-text {
        color: white;
    }
    .sidebar-section {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }
    .disease-badge {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 0.5rem 1rem;
        border-radius: var(--radius-full);
        display: inline-block;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-sm);
    }
    .conversation-list {
        max-height: 300px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .conversation-list::-webkit-scrollbar {
        width: 6px;
    }
    .conversation-list::-webkit-scrollbar-track {
        background: var(--bg-primary);
        border-radius: 3px;
    }
    .conversation-list::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }
    .conversation-list::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="chat-header">
        <h1>Plant Pulse Chat</h1>
        <p>Ask our AI expert about plant diseases, symptoms, and best-practice management</p>
    </div>
    """, unsafe_allow_html=True)

    # Get detected disease from prediction page
    detected_disease = st.session_state.get("last_prediction_label")
    class_names = st.session_state.get("class_names", [])
    
    # Suggested prompts based on last prediction
    suggestions = []
    if detected_disease:
        disease_name = detected_disease.replace("___", " ").replace("_", " ")
        suggestions = [
            f"What causes {disease_name}?",
            f"How to treat {disease_name}?",
            f"How to prevent {disease_name}?",
            f"What are the symptoms of {disease_name}?",
            f"What are the management practices for {disease_name}?",
        ]
    else:
        # Default suggestions when no disease detected
        suggestions = [
            "How to identify plant diseases?",
            "What are common plant disease symptoms?",
            "How to prevent plant diseases?",
            "Organic treatment options for plant diseases",
            "When to use chemical treatments?",
        ]
    
    # Display detected disease context if available
    if detected_disease:
        disease_display = detected_disease.replace("___", " → ").replace("_", " ")
        st.markdown(f"""
        <div style="background: var(--bg-primary); border: 1px solid var(--border); 
                    border-left: 4px solid var(--primary); border-radius: var(--radius-lg); 
                    padding: 1rem; margin-bottom: 1.5rem; box-shadow: var(--shadow-sm);">
            <p style="margin: 0; color: var(--text-secondary); font-size: 0.9375rem;">
                <strong style="color: var(--primary-light); font-weight: 600;">Detected Disease:</strong> 
                <span class="disease-badge">{disease_display}</span>
            </p>
            <p style="margin: 0.5rem 0 0; color: var(--text-tertiary); font-size: 0.875rem;">
                Chat will provide specific advice about this disease
            </p>
        </div>
        """, unsafe_allow_html=True)

    model = get_gemini_client()

    # Initialize messages and pending query
    if "messages" not in st.session_state:
        st.session_state.messages = []  # type: ignore
    
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None  # type: ignore
    
    # Initialize current conversation index
    if "current_conversation_idx" not in st.session_state:
        st.session_state.current_conversation_idx = -1  # type: ignore

    left, right = st.columns((4, 8), gap="large")

    with left:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### Quick Actions")
        if st.button("Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []  # type: ignore
            st.session_state.pending_query = None  # type: ignore
            st.session_state.current_conversation_idx = -1  # type: ignore
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Key Configuration
        with st.expander("Configure Gemini API Key", expanded=False):
            st.markdown(
                "**Get your free API key:**\n"
                "1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)\n"
                "2. Create an API key\n"
                "3. Set it below (stored in session only)\n\n"
                "**Or set as environment variable:** `GEMINI_API_KEY=your_key_here`"
            )
            api_key_input = st.text_input(
                "Enter Gemini API Key:",
                type="password",
                help="This is stored in session only. For permanent setup, use environment variable.",
                key="api_key_input"
            )
            if api_key_input:
                os.environ["GEMINI_API_KEY"] = api_key_input
                st.success("✅ API key set! Refresh the page to use Gemini AI.")
                st.rerun()
        
        # Conversation list sidebar
        conversations = group_messages_into_conversations(st.session_state.messages)  # type: ignore
        if conversations:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Recent Questions")
            st.markdown('<div class="conversation-list">', unsafe_allow_html=True)
            
            # Display conversation list (most recent first)
            for idx, (question, answer) in enumerate(reversed(conversations)):
                active_class = "active" if (len(conversations) - 1 - idx) == st.session_state.current_conversation_idx else ""
                preview_text = question[:60] + "..." if len(question) > 60 else question
                
                if st.button(
                    f"{len(conversations) - idx}. {preview_text}",
                    key=f"conv_btn_{idx}",
                    use_container_width=True,
                    help=question
                ):
                    st.session_state.current_conversation_idx = len(conversations) - 1 - idx  # type: ignore
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### Suggested Questions")
        if suggestions:
            for idx, s in enumerate(suggestions):
                if st.button(s, use_container_width=True, key=f"suggestion_{idx}", 
                           help="Click to ask this question"):
                    # Store the query and trigger processing
                    st.session_state.pending_query = s  # type: ignore
                    st.rerun()
        else:
            st.markdown("""
            <div style="padding: 1rem; background: var(--bg-primary); border-radius: var(--radius-lg); border: 1px solid var(--border);">
                <h4 style="color: var(--primary-light); margin-bottom: 0.75rem; font-weight: 600; font-size: 1rem;">Get Started</h4>
                <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">Try asking about:</p>
                <ul style="color: var(--text-tertiary); font-size: 0.875rem; margin: 0.5rem 0 0; padding-left: 1.25rem; line-height: 1.6;">
                    <li>"Tell me about Apple scab"</li>
                    <li>"How to prevent plant diseases?"</li>
                    <li>"What causes yellow leaves?"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### Chat with Plant Expert")
        
        # Process pending query from suggestion button
        if st.session_state.pending_query:
            query = st.session_state.pending_query
            st.session_state.pending_query = None  # type: ignore
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})  # type: ignore
            
            # Generate and show assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    reply = ai_answer(model, st.session_state.messages, detected_disease)  # type: ignore
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": reply})  # type: ignore
            
            # Set to show latest conversation
            conversations = group_messages_into_conversations(st.session_state.messages)  # type: ignore
            st.session_state.current_conversation_idx = len(conversations) - 1  # type: ignore
            st.rerun()
        
        # Group messages into conversations
        conversations = group_messages_into_conversations(st.session_state.messages)  # type: ignore
        
        # Chat carousel display
        if conversations:
            # Ensure current index is valid
            if st.session_state.current_conversation_idx < 0 or st.session_state.current_conversation_idx >= len(conversations):
                st.session_state.current_conversation_idx = len(conversations) - 1  # type: ignore
            
            current_idx = st.session_state.current_conversation_idx  # type: ignore
            current_question, current_answer = conversations[current_idx]
            
            # Navigation controls
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if st.button("⬅️ Previous", use_container_width=True, disabled=(current_idx == 0)):
                    st.session_state.current_conversation_idx = current_idx - 1  # type: ignore
                    st.rerun()
            
            with col2:
                st.markdown(f"""
                <div class="conversation-counter" style="text-align: center; padding: 8px;">
                    Conversation {current_idx + 1} of {len(conversations)}
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if st.button("Next ➡️", use_container_width=True, disabled=(current_idx == len(conversations) - 1)):
                    st.session_state.current_conversation_idx = current_idx + 1  # type: ignore
                    st.rerun()
            
            # Display current conversation in compact format
            # Escape HTML in question
            escaped_question = current_question.replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&#39;")
            
            # Handle answer generation if needed
            answer_to_display = current_answer
            if not current_answer:
                with st.spinner("Generating response..."):
                    # Reconstruct messages up to this point to generate answer
                    msg_pairs = []
                    for i in range(current_idx):
                        msg_pairs.extend([
                            {"role": "user", "content": conversations[i][0]},
                            {"role": "assistant", "content": conversations[i][1] if conversations[i][1] else ""}
                        ])
                    msg_pairs.append({"role": "user", "content": current_question})
                    
                    reply = ai_answer(model, msg_pairs, detected_disease)  # type: ignore
                    answer_to_display = reply
                    
                    # Update the messages with the generated answer
                    if len(st.session_state.messages) % 2 == 1:  # Only user message, no answer yet
                        st.session_state.messages.append({"role": "assistant", "content": reply})  # type: ignore
                    st.rerun()
            
            # Display question header
            st.markdown(f"""
            <div class="chat-carousel-container">
                <div style="margin-bottom: 1.5rem;">
                    <div style="background: linear-gradient(135deg, var(--primary), var(--primary-dark)); padding: 1.25rem; border-radius: var(--radius-lg); margin-bottom: 1rem; box-shadow: var(--shadow-sm);">
                        <div style="color: white; font-weight: 600; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">
                            Your Question
                        </div>
                        <div style="color: white; font-size: 1rem; line-height: 1.6;">
                            {escaped_question}
                        </div>
                    </div>
            """, unsafe_allow_html=True)
            
            # Display answer header
            st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.25rem; border-radius: var(--radius-lg); border-left: 4px solid var(--primary); margin-bottom: 1rem; box-shadow: var(--shadow-sm);">
                <div style="color: var(--primary-light); font-weight: 600; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">
                    AI Expert Answer
                </div>
            """, unsafe_allow_html=True)
            
            # Display answer content using Streamlit's markdown (properly renders markdown formatting)
            st.markdown(answer_to_display)
            
            # Close all containers
            st.markdown('</div></div></div>', unsafe_allow_html=True)
        else:
            # No conversations yet - show welcome message
            st.markdown("""
            <div style="text-align: center; padding: 3.5rem 2.5rem; background: var(--bg-primary); 
                        border-radius: var(--radius-2xl); border: 2px dashed var(--border); margin-bottom: 1.5rem; box-shadow: var(--shadow-sm);">
                <h3 style="color: var(--primary-light); margin-bottom: 1rem; font-size: 1.5rem; font-weight: 600;">Welcome to Plant Pulse Chat</h3>
                <p style="color: var(--text-secondary); margin: 0; font-size: 1rem; line-height: 1.6;">Ask me anything about plant diseases, symptoms, treatments, or prevention strategies!</p>
                <p style="color: var(--text-tertiary); margin: 0.75rem 0 0; font-size: 0.9rem;">Click a suggested question or type your own below</p>
            </div>
            """, unsafe_allow_html=True)

        # Chat input (always visible)
        user_input = st.chat_input("Ask about plant diseases, symptoms, or treatments...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})  # type: ignore

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = ai_answer(model, st.session_state.messages, detected_disease)  # type: ignore
                    st.markdown(reply)
            
            st.session_state.messages.append({"role": "assistant", "content": reply})  # type: ignore
            
            # Set to show latest conversation
            conversations = group_messages_into_conversations(st.session_state.messages)  # type: ignore
            st.session_state.current_conversation_idx = len(conversations) - 1  # type: ignore
            st.rerun()


if __name__ == "__main__":
    main()
