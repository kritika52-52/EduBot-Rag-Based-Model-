"""
Step 7: Chat interface.

The final deliverable your team demos in the viva. Loads a vector store,
lets the student type a question, retrieves the top chunks, sends them to
the LLM, and displays the answer along with the source document(s).

Run:
    streamlit run app.py
"""

import streamlit as st
from retrieval.retriever import Retriever
from generation.llm import generate_answer
from dotenv import load_dotenv
load_dotenv()


VECTOR_STORE_PATH = "vector_store/minilm_fixed"  # change to whichever store performed best
TOP_K = 4

st.set_page_config(page_title="EduBot", page_icon="🎓")
st.title("🎓 EduBot — Academic Query Assistant")
st.caption("Ask me about attendance, syllabus, exam rules, and other college policies.")


@st.cache_resource
def load_retriever():
    return Retriever(VECTOR_STORE_PATH)


retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Type your question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching college documents..."):
            hits = retriever.retrieve(question, top_k=TOP_K)

            if not hits:
                answer = "I couldn't find anything relevant in the college documents."
            else:
                answer = generate_answer(question, hits)

            st.markdown(answer)

            if hits:
                with st.expander("Sources used"):
                    for hit in hits:
                        st.markdown(f"- **{hit['source']}** (similarity: {hit['similarity']:.2f})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
