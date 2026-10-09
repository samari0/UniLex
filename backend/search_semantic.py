"""SBERT retrieval with query/document reranking and exact-term priority.

The two models are used sequentially to keep the web service's memory bounded.
Definitions are ranked before explanations to identify the requested concept.
Returned scores remain SBERT cosine similarities, not reranker probabilities.
"""

from collections import Counter
import ctypes
import gc
import hashlib
import json
import os
import re
import sys
import threading
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

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
        self._definitions = [
            ". ".join(str(value).strip() for value in row if str(value).strip())
            for row in df[["term", "formal_definition"]]
            .fillna("").itertuples(index=False, name=None)
        ]
        self._phrase_definitions = df["formal_definition"].fillna("").astype(str).tolist()
        # Short phrase anchors must include an informative word, not just
        # generic wording such as "data structure uses".
        stemmer = PorterStemmer()
        self._definition_frequencies = Counter()
        for definition in self._phrase_definitions:
            self._definition_frequencies.update({
                stemmer.stem(token) for token in
                re.findall(r"[^\W_]+", definition.casefold())
            })
        torch.set_num_threads(1)
        # build.sh initializes this class, so both models are cached at build
        # time. Release the reranker before loading SBERT, including on startup.
        reranker = self._load_reranker(local_files_only=False)
        del reranker
        self._release_memory()
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        cache_path = os.path.join(cache_dir, EMBEDDINGS_CACHE)

        fingerprint = hashlib.sha256(json.dumps(
            [MODEL_NAME, search_texts], ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        metadata_path = cache_path + ".sha256"
        cached = None
        try:
            with open(metadata_path, encoding="utf-8") as f:
                if f.read().strip() == fingerprint:
                    cached = np.load(cache_path, allow_pickle=False)
            if cached is not None and (cached.shape != (len(search_texts),
                    self.model.get_sentence_embedding_dimension()) or
                    not np.isfinite(cached).all()):
                cached = None
        except (OSError, ValueError):
            cached = None
        self.embeddings = cached if cached is not None else self._encode_and_cache(search_texts, cache_path)
        if cached is None:
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(fingerprint)

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

    def _prefer_phrase_matches(self, query: str, ranked: list[tuple]) -> list[tuple]:
        """Disambiguate close semantic matches using ordered definition phrases.

        Normalize punctuation and inflections; short three-word phrases must
        consist entirely of content words. Apply to SBERT candidates before the
        reranker cutoff, so a direct definition phrase is not discarded by an imperfect reranker.
        """
        stemmer = PorterStemmer()
        tokens = re.findall(r"[^\W_]+", query.casefold())
        stemmed = [stemmer.stem(token) for token in tokens]
        phrases = {
            tuple(stemmed[start:start + size])
            for size in (3, 4)
            for start in range(len(tokens) - size + 1)
            if len(set(tokens[start:start + size])) >= 3
            and (size == 4 or (
                all(token not in ENGLISH_STOP_WORDS
                    for token in tokens[start:start + size])
                and any(0 < self._definition_frequencies[token]
                        <= max(1, len(self.df) * 0.02)
                        for token in stemmed[start:start + size])
            ))
        }
        if not phrases or len(ranked) < 2:
            return ranked
        evidence = []
        for index, relevance in ranked:
            words = [stemmer.stem(token) for token in
                     re.findall(r"[^\W_]+", self._phrase_definitions[index].casefold())]
            definition_phrases = {
                tuple(words[start:start + size])
                for size in (3, 4)
                for start in range(len(words) - size + 1)
            }
            evidence.append((index, relevance, len(phrases & definition_phrases)))
        strongest = max(count for _, _, count in evidence)
        if strongest == 0:
            return ranked
        return [(index, relevance) for index, relevance, count in evidence
                if count == strongest]

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
                    # Frame the task as dictionary concept identification.
                    # Long explanations can mention a symptom incidentally;
                    # prefer a definition that directly describes the query.
                    definition_query = "Which term describes this situation? " + query
                    relevance = np.asarray(reranker.predict(
                        [(definition_query, self._definitions[i]) for i in candidates],
                        batch_size=1,
                        activation_fct=torch.nn.Identity(),
                        show_progress_bar=False,
                    )).reshape(-1)
                    valid = relevance[np.isfinite(relevance)]
                    if not len(valid) or float(valid.max()) < 0.0:
                        # Some definitions are terse. Preserve explanation-based
                        # retrieval when no definition clears the relevance gate.
                        relevance = np.asarray(reranker.predict(
                            [(query, self._documents[i]) for i in candidates],
                            batch_size=1,
                            activation_fct=torch.nn.Identity(),
                            show_progress_bar=False,
                        )).reshape(-1)
                        valid = relevance[np.isfinite(relevance)]
                    # top_k is a maximum, not a quota. Keep only close contenders
                    # within 0.5 raw reranker logit of the strongest match.
                    contenders = [
                        (i, float(value))
                        for i, value in zip(candidates, relevance)
                        if np.isfinite(value)
                    ]
                    phrase_matches = self._prefer_phrase_matches(query, contenders)
                    if len(phrase_matches) < len(contenders):
                        # A normalized definition phrase supplies lexical evidence
                        # independently of the small reranker's raw logit.
                        phrase_cutoff = max(value for _, value in phrase_matches) - 0.5
                        ranked = [(i, value) for i, value in phrase_matches
                                  if value >= phrase_cutoff]
                    else:
                        cutoff = max(0.0, float(valid.max()) - 0.5) if len(valid) else float("inf")
                        ranked = [(i, value) for i, value in contenders if value >= cutoff]
                    ranked.sort(key=lambda item: (-item[1], -float(scores[item[0]]), item[0]))
                    accepted = [i for i, _ in ranked]
                finally:
                    del reranker
                    self._release_memory()
            return [
                {"index": int(i), "score": float(scores[i])}
                for i in accepted[:top_k]
            ]
