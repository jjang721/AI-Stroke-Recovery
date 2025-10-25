import streamlit as st
import sys

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

# ========================
# ENVIRONMENT SETUP
# ========================
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Using this context, find an answer!:

{context}

---

Answer my question: {question}
"""

# ========================
# MODEL & DB INITIALIZATION
# ========================
embeddings = OpenAIEmbeddings()
CHROMA_PATH = "chroma"  # Path to existing DB
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)
model = ChatOpenAI()
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ========================
# STREAMLIT PAGE CONFIG
# ========================
st.set_page_config(
    page_title="🧠 Stroke Recovery Assistant", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================
# ENHANCED CUSTOM STYLES
# ========================
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Header styling */
    h1 {
        text-align: center;
        color: #ffffff;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* User message bubble */
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin-bottom: 16px;
        margin-left: auto;
        width: fit-content;
        max-width: 75%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* AI message bubble */
    .ai-bubble {
        background: #ffffff;
        color: #2d3748;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin-bottom: 16px;
        margin-right: auto;
        width: fit-content;
        max-width: 75%;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        animation: slideInLeft 0.3s ease-out;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Message labels */
    .user-bubble b, .ai-bubble b {
        display: block;
        margin-bottom: 6px;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    /* Input container */
    .input-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 12px 16px;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 12px;
        font-weight: 600;
        color: #2d3748;
        border: 1px solid #e2e8f0;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 0 0 12px 12px;
        padding: 1rem;
    }
    
    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        padding: 16px;
        margin-bottom: 12px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .source-card b {
        color: #667eea;
        font-size: 0.9rem;
        display: block;
        margin-bottom: 8px;
    }
    
    .source-card p {
        color: #4a5568;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Warning message styling */
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 12px 16px;
        border-radius: 12px;
        border-left: 4px solid #ffc107;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #5568d3;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# HEADER
# ========================
st.markdown("<h1>🧠 Stroke Recovery Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions about stroke rehabilitation backed by scientific research</p>", unsafe_allow_html=True)

# ========================
# SESSION STATE FOR CHAT
# ========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# ========================
# CHAT DISPLAY CONTAINER
# ========================
chat_html = "<div class='chat-container'>"
if len(st.session_state.messages) == 0:
    chat_html += """
    <div style='text-align:center;padding:3rem;color:#a0aec0;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>💬</div>
        <p style='font-size:1.1rem;font-weight:500;'>Start a conversation</p>
        <p style='font-size:0.9rem;'>Ask any question about stroke recovery and rehabilitation</p>
    </div>
    """
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"<div class='user-bubble'><b>You</b>{msg['content']}</div>"
        else:
            if "⚠️" in msg['content']:
                chat_html += f"<div class='ai-bubble warning-message'>{msg['content']}</div>"
            else:
                chat_html += f"<div class='ai-bubble'><b>AI Assistant</b>{msg['content']}</div>"

chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)
# ========================
# USER INPUT CONTAINER
# ========================
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("💬 Type your question here...", label_visibility="collapsed")
    submit = st.form_submit_button("Send")

st.markdown("</div>", unsafe_allow_html=True)


# ========================
# QUERY PROCESSING
# ========================
if submit and query and not st.session_state.processing:
    st.session_state.processing = True
    
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("🔍 Searching scientific literature..."):
        results = db.similarity_search_with_relevance_scores(query, k=3)

        if len(results) == 0 or results[0][1] < 0.5:
            answer = "⚠️ No good matches found. Try rewording your question or asking something more specific."
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt = prompt_template.format(context=context_text, question=query)
            response_text = model.predict(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response_text})


            st.session_state.last_sources = results  # Save sources
    
    st.session_state.processing = False
    st.rerun()

if "last_sources" in st.session_state and len(st.session_state.last_sources) > 0:
    with st.expander("📄 View sources from scientific papers"):
        for i, (doc, score) in enumerate(st.session_state.last_sources):
            st.markdown(f"""
            <div class="source-card">
                <b>Source {i+1} • Relevance Score: {score:.2f}</b>
                <p>{doc.page_content[:400]}{'...' if len(doc.page_content) > 400 else ''}</p>
            </div>
            """, unsafe_allow_html=True)