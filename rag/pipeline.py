"""End-to-end RAG pipeline: ingest -> index -> retrieve -> generate."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from rag.index import TfidfIndex
from rag.ingest import SUPPORTED_EXTENSIONS, load_directory
from rag.llm import generate_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = PROJECT_ROOT / "data" / "docs"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "index.joblib"


class RAGPipeline:
    def __init__(self, docs_dir: Path = DEFAULT_DOCS, index_path: Path = DEFAULT_INDEX,
                 min_score: float = 0.05, min_coverage: float = 0.2) -> None:
        self.docs_dir = Path(docs_dir)
        self.index_path = Path(index_path)
        self.min_score = min_score
        self.min_coverage = min_coverage
        if self.index_path.exists():
            self.index = TfidfIndex.load(self.index_path)
        else:
            self.index = TfidfIndex()
            self.rebuild()

    def rebuild(self) -> int:
        """Re-read every document and rebuild the index. Returns the chunk count."""
        chunks = load_directory(self.docs_dir)
        if chunks:
            self.index = TfidfIndex()
            self.index.build(chunks)
            self.index.save(self.index_path)
        return len(chunks)

    def add_document(self, src: Path) -> int:
        if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type {src.suffix}; use {sorted(SUPPORTED_EXTENSIONS)}")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.docs_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy(src, dest)
        return self.rebuild()

    def list_documents(self) -> list[str]:
        return sorted({c.source for c in self.index.chunks})

    def retrieve(self, question: str, k: int = 4):
        """Top-k passages, or [] when even the best one doesn't cover the question."""
        hits = self.index.search(question, k=k, min_score=self.min_score)
        if not hits or max(self.index.coverage(question, c) for c, _ in hits) < self.min_coverage:
            return []
        return hits

    def ask(self, question: str, k: int = 4) -> dict:
        start = time.perf_counter()
        hits = self.retrieve(question, k=k)
        answer, provider = generate_answer(question, hits)
        return {
            "question": question,
            "answer": answer,
            "provider": provider,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "sources": [
                {"n": n, "source": c.source, "page": c.page, "section": c.section,
                 "score": round(score, 3), "text": c.text}
                for n, (c, score) in enumerate(hits, start=1)
            ],
        }
