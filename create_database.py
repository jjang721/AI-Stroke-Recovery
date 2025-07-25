from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import shutil
import openai
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
print("Using .env at:", dotenv_path)
load_dotenv(dotenv_path, override=True)

DATA_PATH = "data/science"
CHROMA_PATH = "chroma"

embeddings = OpenAIEmbeddings()



def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_md(documents)[:5]
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

def split_md(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    parts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(parts)} chunks.")
    document = parts[10]

    print(document.page_content)
    print(document.metadata)
    return parts 

def save_to_chroma(parts: list[Document]):

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
    parts,
    embeddings,  
    persist_directory=CHROMA_PATH,
    )

    db.persist()
    print(f"Saved {len(parts)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()



