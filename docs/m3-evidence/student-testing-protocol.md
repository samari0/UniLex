# UniLex student pilot — ready-to-use protocol

Status: prepared protocol, not evidence that testing has happened. Use participant codes such as P01; do not collect names, IDs, grades or contact details in the exported results. Participation is voluntary. Explain the task and allow withdrawal; follow any institutional requirements. Do not upload private lecture content or personal data.

## Proposed objectives (confirm before recruiting)

1. At least 80% of assigned lookup tasks completed without facilitator assistance.
2. Median clarity and usefulness ratings at least 4/5.
3. Record and prioritize every critical misunderstanding or incorrect definition observed; document changes and retests.

These are proposed targets, not achieved outcomes. Recruit 20–30 students if feasible; report the actual sample and recruitment limitations. A smaller sample is acceptable to report honestly, not to extrapolate broadly.

## Session script

“UniLex is a CS/AI glossary prototype. We are testing the site, not your knowledge. Please think aloud as you try the tasks. You can stop at any time. We will record task success, approximate time, ratings and anonymous comments.”

Keep the tested deployment commit, date, device and browser with each session. Do not give the target term during a paraphrase task. Randomize task order if comparing methods; do not show participants a successful answer in one mode before timing the identical task in the other mode.

| Task | Student instruction | Facilitator success criterion |
|---|---|---|
| 1 | Find Overfitting using Keyword search and explain it in your own words | Opens correct entry; distinguishes training from new data |
| 2 | Use Smart search: “Why does my model perform well on training data but poorly on unseen data?” | Finds Overfitting without coaching |
| 3 | Find the data structure where the first item added is the first removed | Finds Queue; understands FIFO |
| 4 | Paste “Machine learning uses neural networks. Overfitting can be reduced with regularization.” | Recognizes matching terms and can open/read cards |
| 5 | Try unrelated input, then clear the box and search | Understands no-result and blank-input messages |

Record success yes/no, time in seconds, assistance yes/no, clarity/ease/usefulness each 1–5, and one optional comment. Success is assessed against the criterion, not only whether a result card appears. Do not mark failed tasks successful after facilitator help.

## Analysis

Task completion = successful unassisted tasks / assigned tasks. Report numerator and denominator, sample size, missing responses, median time for completed tasks, and median rating with distribution. Group comments into concrete themes; retain contradictory feedback. Compare pre/post changes only if the same measurement protocol supports it. Never report improved understanding from satisfaction ratings alone.

## Follow-up

Link each actionable finding to an issue, chosen change, commit and retest. Ask the mentor/domain reviewer to review terminology accuracy separately from usability. The facilitator and team must provide the real observations; no rows have been fabricated in the companion CSV.
