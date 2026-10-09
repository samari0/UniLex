# M3 evidence checklist — 9 October 2026

Source: Part 1 of UniLex_Full_Work_Plan_From_Start_to_Finish(1).docx, plus the broader M3 development and 20-day plans. Academic NLP work is separate. The plan's definition-to-category experiment conflicts with later proposal versions; do not implement that academic design without reconciling the approved proposal.

## Website tuning

| Item | Implementation / evidence | Verification |
|---|---|---|
| Keyword search | Exact-match priority repaired; thresholds documented in README; fixed queries in both modes | Local regression report |
| Smart search | API retained; existing relevance gates retained; content fingerprint prevents stale embeddings | Local regression report; live verification pending |
| API | Empty input, 404 unknown term, 503 semantic unavailable, 422 limits, complete cards | Automated API checks |
| Lecture text | Plurals, aliases, punctuation boundaries, ambiguous-word filtering, specialized surface matching | Automated extraction checks |
| Dataset | 3,067 rows; populated fields; no normalized exact duplicate names; related lists deduplicated | Global structural audit, targeted scientific review only |
| Frontend | Blank/loading/no-result/error, safe card rendering, source links, related navigation, mode switching, stale-request protection | Chromium fixture checks |
| Configuration | Existing same-origin config and Render service retained | Deployment and live checks pending |
| UI/UX | Responsive controls, wrapping, focus indicators, reduced-motion behavior | Four viewport sizes |
| Group testing | 25 fixed queries; automated API tests; screenshots | See actual test outputs; do not infer all-query accuracy |

## Data corrections

Stack example now describes undo history (LIFO), not a printer queue (FIFO). Canary Insertion describes testing reproduction of an artificial unique sequence, not removing markers from images. Class Index Mismatch describes reversed numeric-label mappings. Added the missing general Data Leakage entry. Whitespace and repeated related terms were cleaned.

Reviewed references:
- https://research.google/pubs/the-secret-sharer-evaluating-and-testing-unintended-memorization-in-neural-networks/
- https://keras.io/api/data_loading/image/
- https://scikit-learn.org/stable/common_pitfalls.html#data-leakage

All rows were structurally checked. Thousands of definitions have NOT been individually scientifically certified. Concept variants remain, and legacy source/license details need review. Known house-price and sentiment questions remain in the test set even if rejected by relevance filtering.

## Broader M3 completion gates

| Requirement | Evidence available / next action | Status |
|---|---|---|
| Implemented solution and NLP | Repository, site, architecture/API documented in README | Technical work in progress until live checks |
| Problem validation | A proposal mentions a student survey; actual responses/results must be located and audited before reporting counts | Not yet verified |
| Target users | University CS/AI students; English glossary MVP | Defined |
| Measurable impact | Use fixed technical tests separately from student clarity/usefulness/task completion ratings | Real-student results required |
| Content quality target | 3,067 available entries is not equivalent to 150–300 expert-reviewed entries | Reviewed subset/provenance still needed |
| Feedback capability | No persistent feedback collection service is currently configured | Feedback channel/form still needed |
| Community/mentor engagement | Existing mentor correspondence may document proposal suitability, not final sign-off | Final review evidence required |
| Leadership/teamwork | Actual dated contributions, meetings and decisions required | Do not invent member contributions |
| Reflection | Actual member learning/challenges required | Team input needed |
| Final report/slides/demo | Technical evidence can populate drafts after live verification | Not final/approved |
| Academic NLP notebook/report | Separate approved experimental task, held-out data and measured results | Outside website tuning completion |

This checklist deliberately keeps real-user testing, team evidence, scientific review and mentor approval open. Automated software tests cannot replace them.

## Local validation on 9 October 2026

26 API/data/extraction tests passed. The semantic batch explicitly xfailed because two of 25 fixed cases returned no result (house-price and sentiment paraphrases); 23 cases met their expected top-result/empty-result criterion. FIFO alone is ambiguous and accepts the actual FIFO page-replacement concept as well as Queue. Dropout accepts the synonymous dropout-regularization entry. These labels reflect dictionary concepts, not a guarantee of all-query accuracy. All four UI viewport checks passed.
