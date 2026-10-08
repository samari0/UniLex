"""
Day 3: Basic search using TF-IDF + Cosine Similarity.
"""
"""
TF-IDF + Cosine Similarity search for UniLex.

This module keeps the original TfidfSearch interface used by the backend,
while adding relevance filtering so unrelated queries do not receive
weak/random matches.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


class TfidfSearch:
    def __init__(
        self,
        df: pd.DataFrame,
        search_texts: list[str],
        min_similarity: float = 0.12,
    ):
        self.df = df
        self.min_similarity = min_similarity

        # Keep the existing UniLex preprocessing pipeline.
        self._cleaned = [clean_text(t) for t in search_texts]

        # Word unigrams + bigrams preserve the original search behaviour.
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
        )

        self.matrix = self.vectorizer.fit_transform(self._cleaned)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the UniLex knowledge base.

        Returns only results whose cosine similarity reaches the minimum
        relevance threshold. Results remain ordered by similarity score.
        """
        query_clean = clean_text(query)

        # Empty/whitespace-only input should never be sent to the model.
        if not query_clean.strip():
            return []

        query_vec = self.vectorizer.transform([query_clean])

        # If none of the query terms exist in the fitted vocabulary,
        # cosine similarity will be zero for every document.
        if query_vec.nnz == 0:
            return []

        scores = cosine_similarity(query_vec, self.matrix).flatten()

        # Highest score first.
        ranked_indices = scores.argsort()[::-1]

        results = []

        for idx in ranked_indices:
            score = float(scores[idx])

            # Because results are sorted descending, once a score is below
            # the threshold, all remaining results can be ignored.
            if score < self.min_similarity:
                break

            results.append(
                {
                    "index": int(idx),
                    "score": score,
                }
            )

            if len(results) >= top_k:
                break

        return results
