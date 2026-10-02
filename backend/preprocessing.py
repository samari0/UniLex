import re
import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_nlp = None
_stemmer = PorterStemmer()


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation, drop stopwords, then stem every remaining word
    (Porter stemmer). Order matters: stopwords are removed BEFORE stemming,
    using their normal English spelling (sklearn's stopword list), because
    stemming first would turn words like "does" -> "doe" and "why" -> "whi",
    which no longer match a standard stopword list.

    Stemming matters too: without it, a query like "overfit" would NOT match
    a document containing only "overfitting" (different surface forms), since
    TF-IDF treats them as two unrelated tokens. Stemming unifies them to the
    same root ("overfit") on both sides, so keyword search actually works for
    the word forms students naturally type.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    stemmed = [_stemmer.stem(w) for w in words]
    return " ".join(stemmed)


def tokenize_lemmatize(text: str) -> list[str]:
    """Tokenize + lemmatize + remove stopwords, using spaCy. Used for term extraction."""
    nlp = get_nlp()
    doc = nlp(text)
    tokens = [
        tok.lemma_.lower()
        for tok in doc
        if not tok.is_stop and not tok.is_punct and not tok.is_space and len(tok.text) > 1
    ]
    return tokens
