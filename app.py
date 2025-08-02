import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

# Load environment
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Using this context, find an answer!:

{context}

---

Answer my question: {question}
"""

embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
model = ChatOpenAI()
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

st.set_page_config(page_title="🧠 Stroke Recovery Assistant")
st.title("Stroke Recovery Assistant")
st.write("Ask a question about stroke rehab, and I’ll pull answers from sceintific papers! ")


query = st.text_input("Enter your question:")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #fff9c4;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if query:
    with st.spinner("Searching for answers..."):
        results = db.similarity_search_with_relevance_scores(query, k=3)

        if len(results) == 0 or results[0][1] < 0.5:
            st.error("No good matches found. Try rewording your question.")
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
            prompt = prompt_template.format(context=context_text, question=query)
            response_text = model.predict(prompt)

            st.markdown("### Your chatbot answer:")
            st.write(response_text)

            with st.expander("Context from scientific papers"):
                st.code(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)
