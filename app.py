import streamlit as st

from rag_cli import make_chain  # reuse EXACT same RAG chain

# Try to build the chain once at startup
try:
    CHAIN = make_chain()
    INIT_ERROR = None
except Exception as e:
    CHAIN = None
    INIT_ERROR = str(e)


def main():
    st.set_page_config(page_title="PDF RAG Demo", page_icon="📄")
    st.title("📄 PDF Retrieval-Augmented Generation (RAG)")

    st.write(
        "Ask questions about the PDF documents stored in the `data/` folder.\n\n"
        "The system retrieves the most relevant chunks using semantic search and "
        "uses them as context for Kimi K2 via OpenRouter."
    )

    # Show if there was any problem creating the chain
    if INIT_ERROR is not None:
        st.error(f"Error initializing RAG chain at startup:\n\n{INIT_ERROR}")
        return

    question = st.text_area(
        "Your question",
        height=80,
        placeholder="e.g. What skills are tested in the compiler construction assignment?",
    )

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please type a question first.")
            return

        with st.spinner("Thinking..."):
            try:
                answer = CHAIN.invoke(question)
            except Exception as e:
                st.error(f"Error during RAG call:\n\n{e}")
                return

        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
