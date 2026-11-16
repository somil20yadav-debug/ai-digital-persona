"""Streamlit UI for YOU - AI Writing Style Clone"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# 🔧 FIXED IMPORT HERE
from src.persona_trainer import PersonaTrainer
from src.generator import PersonaGenerator
from src.style_analyzer import StyleAnalyzer
from src.database import add_sample, get_all_samples, get_conversations, save_conversation, delete_all_data

import config

# Page config
st.set_page_config(
    page_title="YOU - AI Persona",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'trainer' not in st.session_state:
    st.session_state.trainer = PersonaTrainer()
    st.session_state.generator = PersonaGenerator(st.session_state.trainer)
    st.session_state.analyzer = StyleAnalyzer()
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #6366f1;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎭 YOU - AI Writing Style Clone</div>', unsafe_allow_html=True)
st.markdown("### Train an AI to write just like you")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150/6366f1/ffffff?text=YOU", width=150)
    st.title("Navigation")
    page = st.radio("Go to", ["📝 Train", "💬 Chat", "📊 Analytics", "⚙️ Settings"])
    
    st.divider()
    
    # Status
    st.subheader("System Status")
    ollama_status = st.session_state.generator.check_ollama_status()
    sample_count = st.session_state.trainer.get_sample_count()
    
    st.metric("Ollama", "🟢 Connected" if ollama_status else "🔴 Disconnected")
    st.metric("Training Samples", sample_count)
    
    if not ollama_status:
        st.warning("⚠️ Ollama not running! Start it with: `ollama serve`")

# PAGE 1: TRAIN
if page == "📝 Train":
    st.header("Train Your AI Persona")
    
    tab1, tab2, tab3 = st.tabs(["✍️ Add Samples", "📁 Upload File", "📚 View Samples"])
    
    # Tab 1: Manual input
    with tab1:
        st.subheader("Paste Your Writing")
        sample_text = st.text_area(
            "Enter text (emails, blog posts, social media, etc.)",
            height=200,
            placeholder="Paste your writing here... Minimum 100 characters."
        )
        
        source_name = st.text_input("Source (optional)", placeholder="e.g., Email, Blog, Tweet")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ Add Sample", type="primary"):
                if len(sample_text) < config.MIN_SAMPLE_LENGTH:
                    st.error(f"Text too short! Need at least {config.MIN_SAMPLE_LENGTH} characters.")
                else:
                    with st.spinner("Analyzing and storing..."):
                        add_sample(sample_text, source_name or "manual")
                        st.session_state.trainer.add_samples([sample_text], [source_name or "manual"])
                        st.success("✅ Sample added successfully!")
                        st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.trainer.clear_samples()
                st.success("All samples cleared!")
                st.rerun()
    
    # Tab 2: File upload
    with tab2:
        st.subheader("Upload Text File")
        uploaded_file = st.file_uploader("Choose a .txt file", type=['txt'])
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            st.text_area("File Preview", content[:500], height=150)
            
            if st.button("📤 Process File", type="primary"):
                # Split into chunks if large
                chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
                for chunk in chunks:
                    if len(chunk) >= config.MIN_SAMPLE_LENGTH:
                        add_sample(chunk, uploaded_file.name)
                
                st.session_state.trainer.add_samples(chunks, [uploaded_file.name] * len(chunks))
                st.success(f"✅ Added {len(chunks)} samples from file!")
                st.rerun()
    
    # Tab 3: View samples
    with tab3:
        samples = get_all_samples()
        st.subheader(f"Stored Samples ({len(samples)})")
        
        if samples:
            for i, sample in enumerate(samples[:10]):  # Show latest 10
                with st.expander(f"Sample {i+1} - {sample.source} ({sample.word_count} words)"):
                    st.write(sample.content[:500] + "..." if len(sample.content) > 500 else sample.content)
                    st.caption(f"Added: {sample.added_at.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No samples yet. Add some to get started!")

# PAGE 2: CHAT
elif page == "💬 Chat":
    st.header("Chat with Your AI Clone")
    
    if st.session_state.trainer.get_sample_count() < config.MIN_SAMPLES:
        st.warning(f"⚠️ Need at least {config.MIN_SAMPLES} samples to chat. Go to Train page first!")
    else:
        # Chat interface
        st.subheader("Conversation")
        
        # Temperature slider
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        # Chat input
        prompt = st.chat_input("Ask your AI clone anything...")
        
        if prompt:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.generator.generate(prompt, temperature)
                    st.write(response)
                    
                    # Analyze similarity
                    if st.session_state.trainer.style_metrics:
                        gen_metrics = st.session_state.analyzer.analyze_text(response)
                        similarity = st.session_state.analyzer.compare_styles(
                            st.session_state.trainer.style_metrics,
                            gen_metrics
                        )
                        st.caption(f"Style Match: {similarity}%")
                        save_conversation(prompt, response, similarity)
                    else:
                        save_conversation(prompt, response, 0.0)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# PAGE 3: ANALYTICS
elif page == "📊 Analytics":
    st.header("Writing Style Analytics")
    
    if not st.session_state.trainer.style_metrics:
        st.warning("No style analysis available. Add samples first!")
    else:
        metrics = st.session_state.trainer.style_metrics
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Sentence Length", f"{metrics['avg_sentence_length']} words")
        with col2:
            st.metric("Vocabulary Richness", f"{metrics['vocab_richness']:.2%}")
        with col3:
            st.metric("Readability", f"{metrics['readability_score']}")
        with col4:
            st.metric("Formality", f"{metrics['formality_score']}/100")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[
                    metrics['avg_sentence_length'],
                    metrics['vocab_richness'] * 100,
                    metrics['readability_score'],
                    metrics['formality_score']
                ],
                theta=['Sentence Length', 'Vocabulary', 'Readability', 'Formality'],
                fill='toself',
                name='Your Style'
            ))
            fig.update_layout(title="Writing Style Profile", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Common phrases
            st.subheader("Common Phrases")
            phrases = metrics.get('common_phrases', [])
            if phrases:
                for phrase, count in phrases[:8]:
                    st.write(f"**{phrase}** - {count} times")
            else:
                st.info("Not enough data yet")
        
        # Conversation history
        st.subheader("Recent Conversations")
        convs = get_conversations(10)
        if convs:
            for conv in convs:
                with st.expander(f"Q: {conv.prompt[:60]}... (Score: {conv.similarity_score}%)"):
                    st.write(f"**Prompt:** {conv.prompt}")
                    st.write(f"**Response:** {conv.response}")
                    st.caption(f"Time: {conv.timestamp}")
        else:
            st.info("No conversations yet")

# PAGE 4: SETTINGS
elif page == "⚙️ Settings":
    st.header("Settings")
    
    st.subheader("Model Configuration")
    st.info(f"Current Model: {config.OLLAMA_MODEL}")
    st.info(f"Ollama URL: {config.OLLAMA_BASE_URL}")
    
    st.subheader("Database")
    st.write(f"📍 Location: `{config.DATABASE_URL}`")
    st.write(f"📍 ChromaDB: `{config.CHROMA_DIR}`")
    
    st.subheader("Danger Zone")
    if st.button("🗑️ Delete All Data", type="secondary"):
        if st.checkbox("I understand this cannot be undone"):
            st.session_state.trainer.clear_samples()
            st.success("All data deleted!")
    
    st.divider()
    st.subheader("About")
    st.write("**YOU v1.0** - AI Writing Style Clone")
    st.write("Built with Streamlit, Ollama, LangChain")
    st.write("[GitHub](https://github.com/yourusername/you)")

# Footer
st.divider()
st.caption("Made with ❤️ using Ollama & Streamlit")
