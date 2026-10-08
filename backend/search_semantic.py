"""SBERT search with relevance filtering and content-aware caching."""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    def __init__(
        self,
        df: pd.DataFrame,
        search_texts: list[str],
        cache_dir: str = ".",
        min_similarity: float = 0.40,
    ):
        if len(df) != len(search_texts):
            raise ValueError(
                "Dataset and search texts must have the same length."
            )
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1.")

        self.df = df
        self.min_similarity = min_similarity
        self._terms = [
            " ".join(str(term).casefold().split())
            for term in df["term"].fillna("")
        ]

        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        dimension = self.model.get_sentence_embedding_dimension()
        self.embeddings = np.empty((0, dimension), dtype=np.float32)

        if not search_texts:
            return

        os.makedirs(cache_dir, exist_ok=True)

        # Rebuild embeddings when indexed content or its order changes.
        fingerprint = hashlib.sha256(
            json.dumps(
                [MODEL_NAME, search_texts], ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()

        cache_path = os.path.join(
            cache_dir, f"sbert_{fingerprint}.npy"
        )

        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if (
                    cached.shape == (len(search_texts), dimension)
                    and np.issubdtype(cached.dtype, np.number)
                    and np.isfinite(cached).all()
                ):
                    self.embeddings = cached
                    return
            except (OSError, ValueError):
                pass

        self.embeddings = self._encode_and_cache(
            search_texts, cache_path
        )

    def _encode_and_cache(
        self, search_texts: list[str], cache_path: str
    ) -> np.ndarray:
        embeddings = self.model.encode(
            search_texts,
            show_progress_bar=True,
            batch_size=32,
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

        # Preserve natural wording for the semantic model.
        query_vec = self.model.encode(
            [query], convert_to_numpy=True
        )
        scores = cosine_similarity(
            query_vec, self.embeddings
        ).flatten()

        exact = [
            term == query.casefold() for term in self._terms
        ]

        # Exact dictionary terms remain available regardless of threshold.
        accepted = [
            i for i, score in enumerate(scores)
            if np.isfinite(score)
            and (exact[i] or score >= self.min_similarity)
        ]

        # Exact terms first, then descending semantic similarity.
        accepted.sort(
            key=lambda i: (not exact[i], -float(scores[i]), i)
        )

        return [
            {"index": int(i), "score": float(scores[i])}
            for i in accepted[:top_k]
        ]
