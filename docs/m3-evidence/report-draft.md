# M3 report technical draft — not submission-ready

## Problem and intended users

UniLex targets university students studying English CS/AI terminology. The working problem is that unfamiliar technical wording can make formal definitions difficult to use. This draft does not quantify that need: insert verified anonymized survey findings, dates, sample size and recruitment method before claiming validation.

## Solution design and implementation

The implementation is a web glossary with keyword search, semantic concept retrieval and lecture-text scanning. FastAPI serves both the API and the HTML/CSS/JavaScript interface from one Render service. A CSV is the canonical knowledge base. Each card shows a stored definition, student-friendly explanation, example, category, difficulty, related terms and source when available. The system retrieves stored material; it does not generate a new explanation for each question.

Keyword search uses normalization, stopword removal before stemming, TF-IDF and cosine similarity with relevance filtering. Smart search uses pretrained MiniLM embeddings followed by a small cross-encoder. A sequential model-loading strategy limits memory consumption but increases latency. Extraction combines surface/lemma matching, acronym aliases, longest spans and ambiguity heuristics. Input limits, safe text rendering, request-state handling and clear blank/no-result/error messages improve usability.

## Data and quality

The revised runtime corpus contains 3,067 entries: the earlier 3,066 plus Data Leakage. Sources are stored per entry, with legacy generic source labels still requiring provenance/license review. Structural validation checks required fields and duplicate term names. Targeted corrections addressed Stack, Canary Insertion and Class Index Mismatch. This audit is not certification of all definitions; the reviewed subset must be distinguished from the full available corpus.

## Technical testing

Refer to the committed test report for the final measured software results. API tests cover contracts, limits, empty inputs, unknown terms and model unavailability. Extraction tests include plurals, abbreviations, ordinary language and punctuation. Chromium UI fixture tests cover state messages, escaping, mode changes, stale responses and four widths. Live deployment verification is separately recorded. Fixed development search cases are not a held-out NLP study and do not measure learning impact.

## Outcomes and impact — awaiting student evidence

Insert real unassisted task completion counts, rating distributions, task times, anonymous feedback and limitations from the student pilot. Do not substitute the software test pass rate for user satisfaction, improved understanding or community impact. Record the actual number of participants, missing observations and concrete changes prompted by feedback.

## Leadership and reflection — awaiting team evidence

Each member should supply dated work, links, coordination decisions, challenges and personal learning. Use the contribution template as a collection aid, not as completed evidence. Include mentor/domain review and final approval only when received. The technical decisions log documents implementation choices without inventing student authorship.

## Remaining delivery work

Complete problem-validation evidence, student testing, content review/provenance, feedback collection, team contributions, reflections, final mentor review, report formatting, slides and demo. Academic NLP experiments remain a separate task and must use the approved proposal and real held-out evaluation. This draft is not a declaration that M3 is complete.
