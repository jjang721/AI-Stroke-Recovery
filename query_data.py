import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate

dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

PROMPT_TEMPLATE = """
Using this context, find an answer!:

{context}

---

Answer my question: {question}

"""

CHROMA_PATH = "chroma"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args= parser.parse_args()
    query_text = args.query_text

    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    print("Using .env at:", dotenv_path)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        print("No results found.")
        return 
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {', '.join(sources)}"
    print(formatted_response)


if __name__ == "__main__":
    main()
