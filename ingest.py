from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "chroma_db"


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, api_key, base_url="https://openrouter.ai/api/v1"):
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        openai_api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "medical-rag-app",
        },
    )


    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )

    return vectordb

