"""Offline retrieval index: TF-IDF word + character n-grams with cosine scoring.

Scores are cosine similarities in [0, 1] where HIGHER IS BETTER, so the
min_score filter keeps working if this is swapped for a dense vector store
(normalise the score direction at this boundary, not in callers).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rag.ingest import Chunk
from rag.text import tokens


class TfidfIndex:
    def __init__(self) -> None:
        # Word n-grams capture meaning; char n-grams tolerate typos and plurals.
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.chunks: list[Chunk] = []
        self.matrix = None

    def _embed_text(self, chunk: Chunk) -> str:
        # Headings carry topic words the body often omits.
        return f"{chunk.section or ''}\n{chunk.text}"

    def _transform(self, texts: list[str]):
        return normalize(hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr())

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index - add documents first.")
        self.chunks = chunks
        texts = [self._embed_text(c) for c in chunks]
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        self.matrix = self._transform(texts)

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        if self.matrix is None:
            return []
        scores = (self.matrix @ self._transform([query]).T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]

    def coverage(self, query: str, chunk: Chunk) -> float:
        """IDF-weighted share of the query's content words found in the chunk (0-1).

        Answers "does this passage actually talk about what was asked?". Common
        words like "company" weigh little; words absent from every document
        (e.g. "ticker") weigh the most, so off-topic questions score low.
        """
        q = tokens(query)
        if not q:
            return 0.0
        n = len(self.chunks)
        doc_freq = {t: sum(t in tokens(f"{c.section or ''} {c.text}") for c in self.chunks) for t in q}
        idf = {t: np.log((1 + n) / (1 + df)) + 1 for t, df in doc_freq.items()}
        found = q & tokens(f"{chunk.section or ''} {chunk.text}")
        return float(sum(idf[t] for t in found) / sum(idf.values()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"word": self.word_vec, "char": self.char_vec, "matrix": self.matrix,
             "chunks": [asdict(c) for c in self.chunks]},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfIndex":
        data = joblib.load(path)
        index = cls()
        index.word_vec, index.char_vec, index.matrix = data["word"], data["char"], data["matrix"]
        index.chunks = [Chunk(**c) for c in data["chunks"]]
        return index
