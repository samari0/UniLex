"""Semantic search with relevance filtering and exact-term priority."""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE = "sbert_embeddings.npy"


class SemanticSearch:
    def __init__(
        self,
        df: pd.DataFrame,
        search_texts: list[str],
        cache_dir: str = ".",
    ):
        self.df = df
        self.model = SentenceTransformer(MODEL_NAME)
        cache_path = os.path.join(cache_dir, EMBEDDINGS_CACHE)

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if self.embeddings.shape[0] != len(search_texts):
                self.embeddings = self._encode_and_cache(
                    search_texts, cache_path
                )
        else:
            self.embeddings = self._encode_and_cache(
                search_texts, cache_path
            )

    def _encode_and_cache(
        self, search_texts: list[str], cache_path: str
    ) -> np.ndarray:
        embeddings = self.model.encode(
            search_texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
        )
        np.save(cache_path, embeddings)
        return embeddings

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not isinstance(query, str) or top_k <= 0:
            return []

        query = " ".join(query.split())
        if not query or len(self.embeddings) == 0:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_vec, self.embeddings).flatten()

        terms = [
            " ".join(str(term).casefold().split())
            for term in self.df["term"].fillna("")
        ]
        exact = [term == query.casefold() for term in terms]

        accepted = [
            i for i, score in enumerate(scores)
            if np.isfinite(score)
            and (exact[i] or score >= 0.40)
        ]
        accepted.sort(
            key=lambda i: (not exact[i], -float(scores[i]), i)
        )

        return [
            {"index": int(i), "score": float(scores[i])}
            for i in accepted[:top_k]
        ]
