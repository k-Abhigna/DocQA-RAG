import os

import pytest
from fastapi.testclient import TestClient

os.environ["LLM_PROVIDER"] = "extractive"  # tests never call a paid or local LLM

from rag import RAGPipeline  # noqa: E402
from rag.ingest import chunk_text  # noqa: E402
from rag.pipeline import DEFAULT_DOCS  # noqa: E402


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    index_path = tmp_path_factory.mktemp("idx") / "index.joblib"
    return RAGPipeline(docs_dir=DEFAULT_DOCS, index_path=index_path)


def test_chunking_keeps_heading_and_code_fences():
    text = "# Title\n\n## Setup\n\n```\nline one\n\nline two\n```\n\nAfter code."
    chunks = chunk_text(text, max_chars=1000)
    assert chunks[0][1] == "Setup"
    assert "line one\n\nline two" in chunks[0][0]  # blank line inside fence did not split


def test_chunking_respects_max_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    assert all(len(c) <= 700 + 300 for c, _ in chunk_text(text, max_chars=700))


def test_retrieval_finds_right_document(pipeline):
    hits = pipeline.retrieve("minimum password length")
    assert hits and hits[0][0].source == "it_security.md"
    assert all(0 <= score <= 1 for _, score in hits)
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_ask_returns_cited_answer(pipeline):
    result = pipeline.ask("How many sick days do I get per year?")
    assert "12" in result["answer"] and "[1]" in result["answer"]
    assert result["sources"][0]["source"] == "leave_policy.md"


def test_unanswerable_question(pipeline):
    result = pipeline.ask("What is the company's stock ticker symbol?")
    assert "couldn't find" in result["answer"]


def test_api_endpoints():
    from api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    body = client.post("/ask", json={"question": "What is the daily meal allowance?"}).json()
    assert "60" in body["answer"]
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    bad = client.post("/documents", files={"file": ("evil.exe", b"data")})
    assert bad.status_code == 400
