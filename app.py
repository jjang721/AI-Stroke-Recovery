import streamlit as st
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
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
model = ChatOpenAI()
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ========================
# STREAMLIT PAGE CONFIG
# ========================
st.set_page_config(page_title="🧠 Stroke Recovery Assistant", page_icon="🧠", layout="wide")

# ========================
# CUSTOM CSS STYLES
# ========================
st.markdown("""
<style>
    .stApp {
        background-color: #fff9c4 !important; /* light yellow */
        color: #000 !important;               /* black text */
    }
    body, h1, h2, h3, h4, h5, h6, p, div, span {
        color: #000 !important;               /* force black text everywhere */
    }
    h1 {
        text-align: center;
        color: #000 !important;
        padding-bottom: 10px;
    }
    .user-bubble {
        background-color: #d1e7dd;
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000 !important;
    }
    .ai-bubble {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000 !important;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 8px;
        color: #000 !important;
        background-color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# HEADER
# ========================
st.title("🧠 Stroke Recovery Assistant")
st.markdown("<p style='text-align:center;color:#555;'>Ask a question about stroke rehab, and I’ll pull answers from scientific papers!</p>", unsafe_allow_html=True)

# ========================
# SESSION STATE FOR CHAT
# ========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ========================
# CHAT DISPLAY (keeps conversation going)
# ========================
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-bubble'><b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Auto-scroll to bottom
st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# ========================
# USER INPUT AT THE BOTTOM
# ========================
query = st.text_input("💬 Type your question:", key=f"input_{len(st.session_state.messages)}")

if query:
    # Save user question
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🔍 Searching for answers..."):
        results = db.similarity_search_with_relevance_scores(query, k=3)

        if len(results) == 0 or results[0][1] < 0.5:
            answer = "⚠️ No good matches found. Try rewording your question."
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt = prompt_template.format(context=context_text, question=query)
            response_text = model.predict(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.last_sources = results

    st.rerun()  # Refresh page to keep chat stacked

# ========================
# SOURCES (only for latest answer)
# ========================
if st.session_state.last_sources:
    with st.expander("📄 View sources from scientific papers"):
        for i, (doc, score) in enumerate(st.session_state.last_sources):
            st.markdown(f"""
            <div style="background-color:#ffffff;padding:10px;margin-bottom:10px;border-radius:8px;
                        box-shadow:0 2px 4px rgba(0,0,0,0.1);border-left:4px solid #2c3e50;">
                <b>Source {i+1} (Relevance: {score:.2f})</b>
                <p>{doc.page_content[:400]}{'...' if len(doc.page_content) > 400 else ''}</p>
            </div>
            """, unsafe_allow_html=True)
