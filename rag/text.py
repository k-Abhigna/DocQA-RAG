"""Shared tokenisation for answerability checks and extractive answers."""

import re

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "be", "of", "to", "in", "for", "and", "or", "on", "at",
    "with", "it", "its", "by", "as", "from", "this", "that", "there", "if", "not", "no",
    "what", "which", "who", "when", "where", "why", "how", "many", "much",
    "do", "does", "did", "can", "could", "should", "must", "will", "would", "need", "get",
    "i", "me", "my", "we", "our", "you", "your", "s",
}


def tokens(text: str) -> set[str]:
    """Lowercase content words with crude plural stemming ("days" -> "day")."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in _WORD.findall(text.lower()) if w not in STOPWORDS}
