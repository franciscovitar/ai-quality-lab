# Human review handoff — prepared, with **zero real reviewers so far**

This is an **engineering workflow for future independent annotation**, not a claim that human evaluation has happened. It turns the [twelve frozen real model outputs](GENERATED_CLAIM_REVIEW.md) into a label-hidden packet with **fifteen provisional author-extracted claim spans**, supplies two separate **blank** reviewer templates, validates submitted files and computes inter-rater agreement only if real completed submissions are provided. The runner never fabricates reviewer decisions or automatically resolves disagreements.

## Boundaries and blinding

- Every question, answer and supporting excerpt is **synthetic and public**; no customer, PAS, university or other private source is used. The packet has the original frozen generator output and exact source citations, but **omits** the author's existing `supported`/`contradicted`/`not_established` judgments, rationales, expected abstention labels and all NLI decisions/logits.
- Claim **segmentation is still provided by the project author**. Offsets into raw generated text and a SHA-256 fingerprint prevent changing or omitting those existing claim spans; the reviewers evaluate labels and response completeness on *that* segmentation. This does **not** independently establish that every atomic claim was extracted. Repetition and truncated statements remain known limitations.
- Reviewers should not read `generated_claim_review_v1.json`, model NLI reports or previous outcome summaries before submitting. Since the repo and labels are already public, **true blinded/independent participation cannot be technically guaranteed** by this packet. Independent full-text segmentation and a sealed unseen evaluation set remain separate future requirements.
- Reviewer identifiers are pseudonyms `reviewer_a` and `reviewer_b`. The comparison accepts two distinct, structurally valid files but permanently reports `independent_human_identities_verified: 0`; a different filename/ID **does not certify two humans**. Reviewer contact information, credentials and private notes should stay out of this public repository.

## Evaluation instructions for real reviewers

For *each* of the 15 claim occurrences, inspect **only** the exact source snippets displayed with its question and generated answer, then record:

| Claim label | Decision rule |
|---|---|
| `supported` | Every assertion in the supplied claim span follows from the inspected excerpts. If mixed supported and unsupported content appears within a supplied span, label the entire span not established and flag segmentation for discussion. |
| `contradicted` | At least one material assertion in that span directly conflicts with an explicit source fact, such as the generated *3 days* versus source *thirty days*. |
| `not_established` | The provided excerpts do not establish or disprove the assertion. Do **not** assume it is false merely because the evidence is absent. |

Every claim entry needs at least one **source ID from the same prompt**, a clear written justification and the supplied stable claim ID. Two identical sentences appearing in one model response are **two occurrences**, not two independent tests of a fact.

Separately, rate each of the 12 **whole answers**, without using the claim verdict as a proxy:

- `addresses_question`: supplies the requested information (this label alone does not imply source correctness).
- `partial`: attempts the task but omits key requested details or is truncated.
- `does_not_answer`: does not give the requested information, including a true but irrelevant source quotation.

Write a rationale for each response label. Review the invalid citation markers in the *actual answer* independently; the current template does not yet collect a separate citation-integrity vote because the existing structural validator already reports that check. Do not infer human agreement on claim segmentation from agreement on these author-supplied spans.

## Run, share and compare

From the repository root, using Python 3.13 and the checked-in `uv.lock`:

```bash
uv sync --locked --extra dev
uv run --locked python -m aql.human_review_runner prepare
```

This writes three ignored local files:

1. `test-results/human-review/packet.json` — exact quoted evidence, question, generated answer, stable claim IDs and character offsets, but **no reference verdicts**.
2. `test-results/human-review/template-reviewer_a.json` — all 15 claim labels and 12 answer ratings deliberately `null`, rationales empty.
3. `test-results/human-review/template-reviewer_b.json` — an independently fillable, equally empty copy.

Both templates are **intentionally invalid submissions** until completed. Give each reviewer the same packet and only their template, privately, without revealing the other's work or the author/model judgments. The reviewed source files must remain unmodified; the template contains a `packet_sha256` binding. Collect completed JSON files separately and compare them:

```bash
uv run --locked python -m aql.human_review_runner compare \
  --left /path/to/completed-reviewer-a.json \
  --right /path/to/completed-reviewer-b.json
```

The comparison produces `test-results/human-review/reviewer-agreement.json`. Incomplete, missing, duplicated or out-of-order claims/responses; unseen citation IDs; weak/missing rationales; mismatched packet hashes; and identical reviewer IDs are all rejected. The comparison reports all disagreements, never silently fills a label or creates adjudicated truth. **Do not commit completed reviewer files without separate consent and privacy review.**

## Agreement and interpretation

The report separates three-class claim labels, binary claim support (`supported` vs other), and three-class answer responsiveness. For each, it records observed raw agreement, chance agreement computed from each annotator's class frequencies, and unweighted Cohen's \(\kappa=(p_o-p_e)/(1-p_e)\). If both reviewers assign one identical label throughout, \(p_e=1\), so kappa is **undefined (`null`)** rather than invented as 1.0. With just fifteen non-independent, project-segmented claim occurrences, confidence intervals and external validity are not established. See [scikit-learn's definition and conventions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) and the [TRIPOD-LLM reporting guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC12104976/) on disclosing annotator counts and agreement.

Automated unit tests use **explicitly synthetic fake reviewer submissions** to test both the success path and negative cases. They do not count as a single human judgment. The CI job **only prepares and archives blank packet/templates**, never reports reviewer agreement as observed. Human identity verification, adjudication of disagreements, independent claim extraction and a sealed held-out dataset are still open gates.
