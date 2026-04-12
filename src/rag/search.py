"""
RAG search pipeline: hybrid BM25 + vector search with cross-encoder re-ranking.

Provides a single `search(query, top_k)` interface that:
1. Runs vector similarity search
2. Runs full-text (BM25) search
3. Fuses results via reciprocal rank fusion
4. Re-ranks top candidates with a cross-encoder

Usage:
    from src.rag.search import TariffSearcher
    searcher = TariffSearcher()
    results = searcher.search("bluetooth speaker", top_k=5)
"""

import json
import time
from pathlib import Path

import lancedb
from sentence_transformers import SentenceTransformer, CrossEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import LANCEDB_DIR, EMBEDDING_MODEL, RERANKER_MODEL


class TariffSearcher:
    """Hybrid search over the tariff knowledge base."""

    def __init__(self, db_path: str = None, table_name: str = "tariff_chunks"):
        self.db_path = db_path or str(LANCEDB_DIR)
        self.table_name = table_name

        # Lazy-loaded models
        self._embedder = None
        self._reranker = None
        self._db = None
        self._table = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    @property
    def table(self):
        if self._table is None:
            self._db = lancedb.connect(self.db_path)
            self._table = self._db.open_table(self.table_name)
        return self._table

    def search(self, query: str, top_k: int = 5, rerank: bool = True,
               chunk_types: list[str] = None) -> list[dict]:
        """
        Hybrid search: vector + FTS with reciprocal rank fusion, then re-rank.

        Args:
            query: Natural language query
            top_k: Number of final results
            rerank: Whether to apply cross-encoder re-ranking
            chunk_types: Filter by chunk type (e.g. ["cross_ruling", "hts_heading"])

        Returns:
            List of dicts with keys: text, source, chunk_type, metadata, score
        """
        fetch_k = top_k * 4  # Over-fetch for fusion

        # 1. Vector search
        q_emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        vector_results = (
            self.table
            .search(q_emb)
            .limit(fetch_k)
            .to_list()
        )

        # 2. Full-text search  
        try:
            fts_results = (
                self.table
                .search(query, query_type="fts")
                .limit(fetch_k)
                .to_list()
            )
        except Exception:
            fts_results = []

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(vector_results, fts_results, k=60)

        # 4. Filter by chunk_type if specified
        if chunk_types:
            fused = [r for r in fused if r.get("chunk_type") in chunk_types]

        # 5. Take top candidates for re-ranking
        candidates = fused[:fetch_k]

        if not candidates:
            return []

        # 6. Re-rank with cross-encoder
        if rerank and len(candidates) > 1:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 7. Format output
        results = []
        for c in candidates[:top_k]:
            results.append({
                "text": c["text"],
                "source": c.get("source", ""),
                "chunk_type": c.get("chunk_type", ""),
                "metadata": json.loads(c.get("metadata_json", "{}")),
                "score": c.get("rerank_score", c.get("rrf_score", 0.0)),
            })

        return results

    def vector_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Pure vector similarity search (no FTS, no re-ranking)."""
        q_emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = self.table.search(q_emb).limit(top_k).to_list()
        return [
            {
                "text": r["text"],
                "source": r.get("source", ""),
                "chunk_type": r.get("chunk_type", ""),
                "metadata": json.loads(r.get("metadata_json", "{}")),
                "score": 1.0 - r.get("_distance", 0.0),
            }
            for r in results
        ]

    @staticmethod
    def _reciprocal_rank_fusion(vector_results: list, fts_results: list, k: int = 60) -> list:
        """Combine vector and FTS results using reciprocal rank fusion."""
        scores = {}  # key: text hash -> {score, record}

        for rank, r in enumerate(vector_results):
            key = r["text"][:200]  # Use text prefix as dedup key
            if key not in scores:
                scores[key] = {"score": 0.0, "record": r}
            scores[key]["score"] += 1.0 / (k + rank + 1)

        for rank, r in enumerate(fts_results):
            key = r["text"][:200]
            if key not in scores:
                scores[key] = {"score": 0.0, "record": r}
            scores[key]["score"] += 1.0 / (k + rank + 1)

        # Sort by RRF score
        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        results = []
        for item in ranked:
            rec = item["record"]
            rec["rrf_score"] = item["score"]
            results.append(rec)

        return results
