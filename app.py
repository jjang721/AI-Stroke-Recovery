import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# ✅ Get OpenAI API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Paths and template
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Using this context, find an answer!:

{context}

---

Answer my question: {question}
"""

# ✅ Use the API key in LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.load_local(CHROMA_PATH, embeddings, allow_dangerous_deserialization=True)
model = ChatOpenAI(api_key=api_key)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Streamlit UI
st.set_page_config(page_title="🧠 Stroke Recovery Assistant")
st.title("🧠 Stroke Recovery Assistant")
st.write("Ask a question about stroke rehab, and I’ll pull answers from your knowledge base!")

query = st.text_input("❓ Enter your question:")

if query:
    with st.spinner("🔍 Searching for answers..."):
        results = db.similarity_search_with_relevance_scores(query, k=3)

        if len(results) == 0 or results[0][1] < 0.7:
            st.error("No good matches found. Try rewording your question.")
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt = prompt_template.format(context=context_text, question=query)
            response_text = model.predict(prompt)

            st.markdown("### 💬 Answer")
            st.write(response_text)

            with st.expander("📄 Context"):
                st.code(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)

            sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
            st.markdown("### 📚 Sources")
            for src in set(sources):
                st.markdown(f"- `{src}`")