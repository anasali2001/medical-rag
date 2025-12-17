from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "chroma_db"

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")

def load_pdfs():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    docs = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        print(f"🔹 Loading {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    if not docs:
        raise ValueError(f"No PDF files found in {DATA_DIR}")
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        # This is an embedding model exposed via OpenRouter
        # See: https://openrouter.ai/models/openai/text-embedding-3-large
        model="openai/text-embedding-3-large",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    print(f"💾 Vector DB saved to: {VECTOR_DIR}")
    return vectordb

def main():
    print("📥 Loading PDFs...")
    docs = load_pdfs()

    print("✂️ Splitting into chunks...")
    chunks = chunk_docs(docs)

    print("🧠 Building embeddings via OpenRouter + Chroma...")
    build_vectorstore(chunks)

    print("🎉 Ingestion complete!")

if __name__ == "__main__":
    main()
