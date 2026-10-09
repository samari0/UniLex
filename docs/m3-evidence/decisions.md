# Verified implementation decisions

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 2026-10-09 | Preserve the existing same-origin Render service | Deployment restructuring is unnecessary for the requested fixes | config.js retained; API serves frontend |
| 2026-10-09 | Fingerprint embedding cache by searchable content and model | Row-count-only validation can retain stale embeddings after edits | search_semantic.py |
| 2026-10-09 | Prefer exact surface plus lemma matches for extraction | POS/lemma variation can hide technical terms such as Overfitting | term_extraction.py; extraction tests |
| 2026-10-09 | Keep punctuation as span boundaries | “machine. learning” must not become “machine learning” | extraction boundary test |
| 2026-10-09 | Keep known retrieval failures visible | Threshold tuning for one example can introduce unrelated results | search-results.json; expected-failure annotation |
| 2026-10-09 | Save a remote review checkpoint before deployment | The previous temporary environment disappeared before saving completed | codex/m3-review branch |

These are observed implementation decisions. They do not assign student authorship, record a team meeting, or stand in for personal reflections. Add actual team decisions and dated evidence separately.
