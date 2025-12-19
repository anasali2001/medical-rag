from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "chroma_db"


def get_retriever(api_key, base_url="https://openrouter.ai/api/v1"):
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        openai_api_key=api_key,
        base_url=base_url,
    )

    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )

    return vectordb.as_retriever(search_kwargs={"k": 6})


def format_docs(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{i}] {d.page_content}")
    return "\n\n".join(lines)


def make_chain(api_key, base_url="https://openrouter.ai/api/v1"):
    retriever = get_retriever(api_key, base_url)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a medical data assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not present, say you don't know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    llm = ChatOpenAI(
        model="moonshotai/kimi-k2",
        openai_api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_tokens=512,
    )

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
