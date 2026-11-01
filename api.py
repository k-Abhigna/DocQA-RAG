"""REST API for DocQA-RAG.  Run:  uvicorn api:app --reload"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag import RAGPipeline
from rag.ingest import SUPPORTED_EXTENSIONS
from rag.llm import pick_provider

app = FastAPI(title="DocQA-RAG", version="1.0.0",
              description="Ask questions about your documents and get cited answers.")
pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(4, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(pipeline.list_documents()),
            "chunks": len(pipeline.index.chunks), "provider": pick_provider()}


@app.get("/documents")
def documents() -> dict:
    return {"documents": pipeline.list_documents()}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return pipeline.ask(req.question, k=req.k)


@app.post("/documents")
async def upload(file: UploadFile) -> dict:
    name = Path(file.filename or "").name  # strip any client-supplied directories
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type; allowed: {sorted(SUPPORTED_EXTENSIONS)}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(await file.read())
        chunks = pipeline.add_document(path)
    return {"added": name, "total_chunks": chunks}
