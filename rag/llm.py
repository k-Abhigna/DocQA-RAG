"""Answer generation with three interchangeable providers.

Selection (override with LLM_PROVIDER=anthropic|ollama|extractive):
  1. anthropic  - if ANTHROPIC_API_KEY is set
  2. ollama     - if a local Ollama server answers at OLLAMA_HOST
  3. extractive - always available, no model: picks the best-matching sentences
"""

from __future__ import annotations

import os
import re
import time

import requests

from rag.ingest import Chunk
from rag.text import tokens

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.
Cite the passages you used with bracketed numbers such as [1] or [2][3].
If the passages do not contain the answer, say "I couldn't find that in the documents."
Keep answers short and factual."""

NOT_FOUND = "I couldn't find that in the documents."


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    parts = []
    for n, (chunk, _) in enumerate(hits, start=1):
        where = chunk.source + (f", page {chunk.page}" if chunk.page else "")
        heading = f" - {chunk.section}" if chunk.section else ""
        parts.append(f"[{n}] ({where}{heading})\n{chunk.text}")
    return "\n\n".join(parts)


def _user_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    return f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"


_ollama_cache: tuple[float, bool] = (0.0, False)


def _ollama_available(ttl: float = 60.0) -> bool:
    """Probe the local Ollama server, caching the result: a refused connection
    can take ~2 s on Windows, which would otherwise be added to every request."""
    global _ollama_cache
    checked_at, ok = _ollama_cache
    if time.monotonic() - checked_at < ttl:
        return ok
    try:
        ok = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1).ok
    except requests.RequestException:
        ok = False
    _ollama_cache = (time.monotonic(), ok)
    return ok


def pick_provider() -> str:
    forced = os.getenv("LLM_PROVIDER")
    if forced:
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _ollama_available():
        return "ollama"
    return "extractive"


def answer_anthropic(question: str, hits: list[tuple[Chunk, float]]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(question, hits)}],
        output_config={"effort": "low"},  # short grounded answers don't need deep reasoning
        # Server-side fallback: if a safety classifier declines, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this question."
    return "".join(b.text for b in response.content if b.type == "text").strip()


def answer_ollama(question: str, hits: list[tuple[Chunk, float]]) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(question, hits)},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def answer_extractive(question: str, hits: list[tuple[Chunk, float]], max_sentences: int = 2) -> str:
    """No-LLM fallback: return the sentences that overlap the question most, with citations."""
    q = tokens(question)
    scored = []
    for n, (chunk, chunk_score) in enumerate(hits, start=1):
        heading = tokens(chunk.section or "")
        for sent in re.split(r"(?<=[.!?])\s+", chunk.text.replace("\n", " ")):
            overlap = len(q & (tokens(sent) | heading))
            if overlap:
                # Tie-break on the retriever's score so higher-ranked chunks win.
                scored.append((overlap + chunk_score, n, sent.strip()))
    if not scored:
        return NOT_FOUND
    scored.sort(key=lambda t: -t[0])
    return " ".join(f"{sent} [{n}]" for _, n, sent in scored[:max_sentences])


def generate_answer(question: str, hits: list[tuple[Chunk, float]]) -> tuple[str, str]:
    """Return (answer, provider_used). Falls back to extractive if the LLM call fails."""
    if not hits:
        return NOT_FOUND, "none"
    provider = pick_provider()
    try:
        if provider == "anthropic":
            return answer_anthropic(question, hits), f"anthropic:{ANTHROPIC_MODEL}"
        if provider == "ollama":
            return answer_ollama(question, hits), f"ollama:{OLLAMA_MODEL}"
    except Exception as exc:  # network/auth/rate-limit: degrade gracefully instead of failing the request
        fallback = answer_extractive(question, hits)
        return f"{fallback}\n\n(LLM unavailable: {type(exc).__name__}; showing extractive answer.)", "extractive"
    return answer_extractive(question, hits), "extractive"
