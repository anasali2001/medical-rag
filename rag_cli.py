from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "chroma_db"

# --- Environment (OpenRouter) ---
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")


# --- Reuse same embeddings & Chroma as ingest.py ---
def get_retriever():
    """Load persisted Chroma DB and expose as retriever."""
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Vector DB folder not found: {VECTOR_DIR}. Run ingest.py first."
        )

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )

    # k = how many chunks per query
    return vectordb.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Format retrieved docs (with page + source) into a single string."""
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        lines.append(f"[{i}] (source: {src}, page {page})\n{d.page_content}")
    return "\n\n".join(lines)


def make_chain():
    retriever = get_retriever()

    # 🔒 Prompt to reduce hallucinations
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant for question-answering over a set of PDF documents.\n"
                "Use ONLY the provided context to answer the question.\n"
                "If the answer is not clearly present in the context, say you don't know instead of guessing.\n"
                "Answer concisely and clearly.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    # 🧠 Kimi K2 via OpenRouter
    llm = ChatOpenAI(
        model="moonshotai/kimi-k2",   # OpenRouter model name
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=512,               # limit output length to keep cost low
        max_retries=2,                # optional: retry on transient issues
    )

    # RAG chain: question -> retrieve -> prompt -> Kimi -> string
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


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
