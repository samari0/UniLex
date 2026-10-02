"""
Day 3: Basic search using TF-IDF + Cosine Similarity.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


class TfidfSearch:
    def __init__(self, df: pd.DataFrame, search_texts: list[str]):
        self.df = df
        self._cleaned = [clean_text(t) for t in search_texts]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        self.matrix = self.vectorizer.fit_transform(self._cleaned)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_clean = clean_text(query)
        query_vec = self.vectorizer.transform([query_clean])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({"index": int(idx), "score": float(scores[idx])})
        return results
