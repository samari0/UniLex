"""
TF-IDF search with exact-term priority and relevance filtering.
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
        min_query_coverage: float = 0.5,
    ):
        if len(df) != len(search_texts):
            raise ValueError(
                "Dataset and search texts must have the same length."
            )

        self.df = df
        self.min_similarity = min_similarity
        self.min_query_coverage = min_query_coverage

        self._cleaned = [clean_text(t) for t in search_texts]

        terms = df["term"].fillna("")
        definitions = df["formal_definition"].fillna("")

        self._terms = [clean_text(str(term)) for term in terms]

        # Require evidence in the term or formal definition.
        # Matches found only in illustrative examples are insufficient.
        self._core_tokens = [
            set(clean_text(str(term) + " " + str(definition)).split())
            for term, definition in zip(terms, definitions)
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
        )
        self.matrix = self.vectorizer.fit_transform(self._cleaned)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not isinstance(query, str) or top_k <= 0:
            return []

        query_clean = clean_text(query)
        if not query_clean.strip():
            return []

        query_vec = self.vectorizer.transform([query_clean])
        if query_vec.nnz == 0:
            return []

        scores = cosine_similarity(query_vec, self.matrix).flatten()
        query_tokens = set(query_clean.split())
        accepted = []

        for idx in scores.argsort()[::-1]:
            score = float(scores[idx])

            exact_match = query_clean == self._terms[idx]
            if not exact_match and (score < self.min_similarity or score <= 0):
                continue

            if not exact_match:
                matched_count = len(
                    query_tokens & self._core_tokens[idx]
                )
                coverage = matched_count / len(query_tokens)

                if coverage < self.min_query_coverage:
                    continue

                # Multiword queries need more than one matching word.
                if len(query_tokens) > 1 and matched_count < 2:
                    continue

            accepted.append(
                (bool(exact_match), int(idx), score)
            )

        # Prioritize normalized exact terms, then cosine similarity.
        # Keep the returned score as the original cosine similarity.
        accepted.sort(
            key=lambda item: (-item[0], -item[2], item[1])
        )

        return [
            {"index": idx, "score": score}
            for _, idx, score in accepted[:top_k]
        ]
