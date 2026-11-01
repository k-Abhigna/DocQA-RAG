"""Streamlit chat UI.  Run:  streamlit run app.py"""

import tempfile
from pathlib import Path

import streamlit as st

from rag import RAGPipeline
from rag.llm import pick_provider

st.set_page_config(page_title="DocQA-RAG", page_icon="📄")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = get_pipeline()

with st.sidebar:
    st.header("Documents")
    for doc in pipeline.list_documents():
        st.write(f"- {doc}")
    uploaded = st.file_uploader("Add a document", type=["pdf", "md", "txt"])
    if uploaded and st.button("Index document"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(uploaded.name).name
            path.write_bytes(uploaded.getvalue())
            n = pipeline.add_document(path)
        st.success(f"Indexed {uploaded.name} ({n} chunks total)")
        st.rerun()
    st.divider()
    k = st.slider("Passages to retrieve", 1, 8, 4)
    st.caption(f"Answer provider: **{pick_provider()}**")

st.title("📄 DocQA-RAG")
st.caption("Ask questions about your documents. Answers cite the passages they come from.")

if "history" not in st.session_state:
    st.session_state.history = []
    # Shareable demo links: /?q=How+many+sick+days  (repeat q for several questions)
    for q in st.query_params.get_all("q"):
        st.session_state.history.append(pipeline.ask(q, k=4))

for turn in st.session_state.history:
    st.chat_message("user").write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander(f"Sources ({len(turn['sources'])}) · {turn['provider']} · {turn['latency_ms']} ms"):
            for s in turn["sources"]:
                where = s["source"] + (f" p.{s['page']}" if s["page"] else "")
                st.markdown(f"**[{s['n']}] {where}** — {s['section'] or ''} (score {s['score']})")
                st.text(s["text"])

if question := st.chat_input("e.g. How many sick days do I get?"):
    st.session_state.history.append(pipeline.ask(question, k=k))
    st.rerun()
