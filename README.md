# UniLex

Student-friendly CS/AI dictionary: keyword search, semantic search and lecture-term extraction.

Website: https://unilex-samar.onrender.com/

## Setup

Production uses Python 3.11. Install `backend/requirements.txt` (CPU PyTorch is sufficient), then run `python -m spacy download en_core_web_sm`. From `backend`, run `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1`. FastAPI serves `frontend` at `/`; `frontend/config.js` uses the same origin. `bash backend/build.sh` installs and preloads models/embeddings for Render. The existing Render service configuration is authoritative; the historical root Blueprint is not used for this single-service deployment.

## Data and methods

Canonical production dataset: `backend/unilex_dataset.csv`, 3,067 entries after the October 2026 review. Root datasets are historical inputs. Sources are stored per row; many legacy source labels still require provenance/license review. This is not a claim that all entries are scientifically reviewed or openly licensed.

Keyword search uses normalized/stemmed TF-IDF word/bigram features, exact-term priority, minimum cosine 0.12, core-query coverage 0.5, and at least two core word matches for nonexact multiword queries. Smart search uses all-MiniLM-L6-v2 embeddings and ms-marco-TinyBERT-L2-v2 reranking. Cosine gates: query maximum >=0.40, candidates >=0.30, up to 256 candidates. Reranker keeps close contenders within 0.5 raw logit of the best, normally with a zero floor; informative definition phrase matches can override that floor. Exact terms take priority. Isolated acronyms explicitly found in dictionary definitions can bypass weak sentence similarity. Scores are cosine similarities, not calibrated confidence.

The two models swap sequentially under a lock to limit memory. Search can take tens of seconds on the free service. Embedding caches are fingerprinted by model name and full search text to prevent stale content after edits. Client timeout is 90 seconds; aborting the browser request does not cancel CPU work already running on the server.

Lecture extraction uses lemma and surface matching, acronym aliases, longest non-overlapping spans, punctuation boundaries and a technical-context guard for ambiguous everyday words. It can still miss unfamiliar forms and misinterpret mixed-topic text.

## API

- `GET /health`: row count and semantic availability.
- `GET /term/{term_name}`: exact lookup; unknown terms return 404.
- `GET /search?q=...&top_k=5`: keyword results.
- `GET /search/semantic?q=...&top_k=5`: semantic results; unavailable model returns 503.
- `POST /extract` with JSON `{"text":"Machine learning uses neural networks."}`.

Search limit: 1,000 characters; top_k: 1–20. Extraction limit: 20,000 characters. Empty inputs return empty lists; invalid lengths return 422.

## Testing

Install `tests/requirements.txt`, then `python -m playwright install chromium`. Run `python -m pytest -q tests/test_acceptance.py` and `python tests/ui_acceptance.py` from the repository root. API tests use real models/data, initialized once without repeating startup. UI tests serve the frontend on port 8765 and use controlled API fixtures. Screenshots cover 320/390/768/1440-pixel widths. They are UI evidence, not live search accuracy evidence.

`docs/search-results.json` records fixed development queries in both modes. Known house-price and sentiment paraphrases remain explicitly expected-failure, not a pass. Development tests are not held-out academic evaluation or real-student impact measurements. See `docs/M3-checklist.md` for scope and remaining evidence.
