import math
import re
from collections import Counter

from core.models import Chunk, RetrievalResult
from .base import SparseRetriever

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever(SparseRetriever):
    """Okapi BM25 over in-memory chunks. No extra dependency."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.docs: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avgdl = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.docs = [tokenize(chunk.text) for chunk in chunks]
        self.doc_freq = Counter()
        for tokens in self.docs:
            for term in set(tokens):
                self.doc_freq[term] += 1
        lengths = [len(tokens) for tokens in self.docs]
        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens or not self.avgdl:
            return 0.0
        counts = Counter(doc_tokens)
        score = 0.0
        dl = len(doc_tokens)
        for term in query_tokens:
            freq = counts.get(term, 0)
            if not freq:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self._idf(term) * (freq * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query_tokens = tokenize(query)
        scored = []
        for index, doc_tokens in enumerate(self.docs):
            score = self._score(query_tokens, doc_tokens)
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)
        results: list[RetrievalResult] = []
        for rank, (score, index) in enumerate(scored[:k], 1):
            results.append(RetrievalResult(self.chunks[index], float(score), rank, channel="sparse"))
        return results
