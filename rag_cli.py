from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "chroma_db"

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")


from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def get_retriever(api_key, base_url="https://openrouter.ai/api/v1"):
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        api_key=api_key,
        base_url=base_url,
    )

    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )

    return vectordb.as_retriever(search_kwargs={"k": 6})

def format_docs(docs):
    """Format retrieved docs (with page + source) into a single string."""
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        lines.append(f"[{i}] (source: {src}, page {page})\n{d.page_content}")
    return "\n\n".join(lines)

def make_chain(api_key, base_url="https://openrouter.ai/api/v1"):
    retriever = get_retriever(api_key, base_url)

    llm = ChatOpenAI(
        model="moonshotai/kimi-k2",
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_tokens=512,
        max_retries=2,
    )






def main():
    chain = make_chain()
    print("✅ RAG CLI ready. Ask questions about your PDFs.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input(">>> ")

        if not question.strip():
            continue

        if question.strip().lower() in {"exit", "quit", "q"}:
            print("👋 Bye!")
            break

        try:
            print("\nThinking...\n")
            answer = chain.invoke(question)
            print("----- Answer -----")
            print(answer)
            print("------------------\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
