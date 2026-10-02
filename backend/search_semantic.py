"""
Day 4: Semantic search using Sentence-BERT (all-MiniLM-L6-v2) + Cosine Similarity.

NOTE: This downloads the 'all-MiniLM-L6-v2' model from Hugging Face on first run
and caches it locally afterward. That download needs outbound internet access to
huggingface.co. It will work fine on Render (or any normal machine/deployment),
but could not be test-run inside the sandbox this backend was built in, since
that sandbox's network policy blocks huggingface.co specifically.
"""
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE = "sbert_embeddings.npy"


class SemanticSearch:
    def __init__(self, df: pd.DataFrame, search_texts: list[str], cache_dir: str = "."):
        self.df = df
        self.model = SentenceTransformer(MODEL_NAME)
        cache_path = os.path.join(cache_dir, EMBEDDINGS_CACHE)

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if self.embeddings.shape[0] != len(search_texts):
                self.embeddings = self._encode_and_cache(search_texts, cache_path)
        else:
            self.embeddings = self._encode_and_cache(search_texts, cache_path)

    def _encode_and_cache(self, search_texts: list[str], cache_path: str) -> np.ndarray:
        embeddings = self.model.encode(
            search_texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True
        )
        np.save(cache_path, embeddings)
        return embeddings

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_vec, self.embeddings).flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({"index": int(idx), "score": float(scores[idx])})
        return results
