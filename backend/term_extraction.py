"""
Day 5: Term extraction from pasted lecture text.
Finds which UniLex dictionary terms actually appear in a block of text.

Implementation note: this does its OWN lemma n-gram matching rather than using
spaCy's built-in PhraseMatcher(attr="LEMMA"). PhraseMatcher's LEMMA matching
compares spaCy's raw per-token lemma attribute, which is case-sensitive and
depends on the POS tag spaCy assigns. A short two-word pattern like "Machine
Learning", tagged in isolation with no surrounding sentence, gets its second
word mistagged as a proper noun, so its lemma stays "Learning" (capitalized)
instead of "learning" — this is correct spaCy behavior for how that attribute
works, but then the pattern no longer matches the same phrase inside a real
sentence, where "learning" is lowercase. Normalizing every lemma to lowercase
ourselves and matching lowercase n-grams directly sidesteps that entirely.
"""
import pandas as pd

from preprocessing import get_nlp


class TermExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.nlp = get_nlp()

        terms = [t.strip() for t in df["term"] if t.strip()]
        ids = [i for t, i in zip(df["term"], df["id"]) if t.strip()]

        # lemma_key (lowercase, space-joined) -> row id
        self.term_to_index = {}
        self.max_term_len = 1
        docs = self.nlp.pipe(terms, disable=["parser", "ner"])
        for term_doc, idx in zip(docs, ids):
            lemmas = [tok.lemma_.lower() for tok in term_doc if not tok.is_punct and not tok.is_space]
            if not lemmas:
                continue
            key = " ".join(lemmas)
            self.term_to_index[key] = idx
            self.max_term_len = max(self.max_term_len, len(lemmas))

    def extract(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        tokens = [tok for tok in doc if not tok.is_punct and not tok.is_space]
        lemmas = [tok.lemma_.lower() for tok in tokens]

        found_indices = {}
        n = len(lemmas)
        # try longest n-grams first, so "machine learning" wins over "learning" alone
        for size in range(min(self.max_term_len, n), 0, -1):
            for start in range(n - size + 1):
                key = " ".join(lemmas[start:start + size])
                idx = self.term_to_index.get(key)
                if idx is not None and idx not in found_indices:
                    found_indices[idx] = key

        results = sorted(found_indices.items(), key=lambda kv: -len(kv[1]))
        return [{"index": int(idx)} for idx, _ in results]
