"""SBERT retrieval with query/document reranking and exact-term priority.

The two models are used sequentially to keep the web service's memory bounded.
Returned scores remain SBERT cosine similarities, not reranker probabilities.
"""

import ctypes
import gc
import os
import sys
import threading
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE = "sbert_embeddings.npy"
RERANKER_NAME = "cross-encoder/ms-marco-TinyBERT-L2-v2"


class SemanticSearch:
    def __init__(
        self,
        df: pd.DataFrame,
        search_texts: list[str],
        cache_dir: str = ".",
    ):
        if len(df) != len(search_texts):
            raise ValueError("Dataset and search texts must have the same length.")
        self.df = df
        self._lock = threading.Lock()
        self._terms = [
            " ".join(str(term).casefold().split())
            for term in df["term"].fillna("")
        ]
        self._documents = [
            ". ".join(str(value).strip() for value in row if str(value).strip())
            for row in df[
                ["term", "formal_definition", "student_friendly_explanation"]
            ].fillna("").itertuples(index=False, name=None)
        ]
        torch.set_num_threads(1)
        # build.sh initializes this class, so both models are cached at build
        # time. Release the reranker before loading SBERT, including on startup.
        reranker = self._load_reranker(local_files_only=False)
        del reranker
        self._release_memory()
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
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

    @staticmethod
    def _release_memory():
        gc.collect()
        # Return freed CPU allocations to Linux rather than retaining both
        # models' allocations in the process allocator between requests.
        if sys.platform.startswith("linux"):
            try:
                ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError):
                pass

    @staticmethod
    def _load_reranker(local_files_only=True):
        return CrossEncoder(
            RERANKER_NAME,
            device="cpu",
            max_length=256,
            local_files_only=local_files_only,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not isinstance(query, str) or top_k <= 0:
            return []

        query = " ".join(query.split())
        if not query or len(self.embeddings) == 0:
            return []

        # The lock also prevents concurrent requests from retaining both
        # models while another request is switching the active model.
        with self._lock:
            if self.model is None:
                self.model = SentenceTransformer(
                    MODEL_NAME, device="cpu", local_files_only=True
                )
            query_vec = self.model.encode([query], convert_to_numpy=True)
            scores = cosine_similarity(query_vec, self.embeddings).flatten()
            exact = [term == query.casefold() for term in self._terms]
            finite = [i for i, score in enumerate(scores) if np.isfinite(score)]

            # Preserve the tested exact-term behavior and unrelated-query gate.
            if any(exact):
                accepted = [i for i in finite if exact[i] or scores[i] >= 0.40]
                accepted.sort(key=lambda i: (not exact[i], -float(scores[i]), i))
            elif not finite or max(scores[i] for i in finite) < 0.40:
                return []
            else:
                # A wider candidate pool lets the reranker recover relevant
                # definitions that SBERT did not place in its first five.
                candidates = sorted(
                    (i for i in finite if scores[i] >= 0.30),
                    key=lambda i: (-float(scores[i]), i),
                )[:256]
                self.model = None
                self._release_memory()
                reranker = self._load_reranker()
                try:
                    relevance = reranker.predict(
                        [(query, self._documents[i]) for i in candidates],
                        batch_size=1,
                        activation_fct=torch.nn.Identity(),
                        show_progress_bar=False,
                    )
                    ranked = [
                        (i, float(value))
                        for i, value in zip(candidates, np.asarray(relevance).reshape(-1))
                        if np.isfinite(value) and value >= 0.0
                    ]
                    ranked.sort(key=lambda item: (-item[1], -float(scores[item[0]]), item[0]))
                    accepted = [i for i, _ in ranked]
                finally:
                    del reranker
                    self._release_memory()
            return [
                {"index": int(i), "score": float(scores[i])}
                for i in accepted[:top_k]
            ]
