# Matching runbook

How to run and evaluate job–CV matching and ranking. Design, scoring rules and evaluation method:
[matching.md](matching.md).

Run every command from the repository root inside the project virtual environment.

## Offline evaluation and demonstration (no key)

```bash
python -m scripts.match_jobs --offline --top 3 --explain
python -m scripts.evaluate_job_matching
```

Offline mode uses a manually labeled synthetic candidate profile and lexical hash embeddings. It
does not parse arbitrary CVs and does not measure semantic retrieval quality. `--explain` prints
each job's deterministic summary: score, supported, unmet and unknown requirements, and the cited
CV quotations.

The evaluator checks 25 labeled requirement statuses over 6 synthetic jobs and compares vector-only
with hybrid retrieval. Recorded result: 25/25 statuses correct; Hit@3 5/6 (vector) vs 6/6
(hybrid). These are development fixtures, not a measure of accuracy on real postings. Details:
[`benchmarks/matching_baseline.md`](../benchmarks/matching_baseline.md).

## Live matching with PostgreSQL and OpenAI (paid)

Requires the `.env` and database setup from [cv-retrieval-runbook.md](cv-retrieval-runbook.md).

```bash
docker compose up -d --wait postgres
python -m scripts.cv init-db
python -m scripts.cv ingest data/fixtures/sample_cv.md --cv-id synthetic-demo
python -m scripts.match_jobs --live --cv-id synthetic-demo --top 3 --explain
```

Live mode extracts cited candidate facts from the stored chunks with the LLM, then scores and ranks
deterministically. `--output` is rejected in live mode so that evidence from a real CV is not
written to a file by accident; never publish terminal output from a real CV.

Fact extraction validates that every cited chunk exists and every quotation is verbatim. It may
retry once, and it re-anchors a skill quotation only when the original citation was valid.
Year-only employment dates are excluded rather than given invented month boundaries, so the
experience requirement becomes unknown. Extraction can still reject invalid model citations; it
fails closed rather than bypassing the evidence checks.

## Ragas explanation-quality evaluation (paid, separate environment)

Ragas 0.4.3 requires OpenAI SDK < 3, while the main environment pins OpenAI 3.x. Install
`requirements-matching-eval.txt` into a **separate Python 3.12 virtual environment**, never into
the main one.

```bash
python3.12 -m venv ~/.venvs/job-match-ragas
~/.venvs/job-match-ragas/bin/python -m pip install -r requirements-matching-eval.txt
~/.venvs/job-match-ragas/bin/python -m scripts.evaluate_matching_ragas \
  --cv-id synthetic-demo --limit 3 --profile-source labeled-fixture \
  --output evaluation/results/matching_ragas_final.json
```

(On Windows the interpreter is `~/.venvs/job-match-ragas/Scripts/python.exe`.)

The evaluator ranks the whole fixture set deterministically, generates LLM explanations for the
top N only, and scores them with Ragas Faithfulness and AnswerRelevancy. Contexts are the retrieved
evidence plus the approved cited chunks.

- `--profile-source labeled-fixture` uses the versioned, manually labeled synthetic profile after
  verifying that the stored CV and chunks match the fixture exactly. It isolates explanation
  quality from extraction variance.
- `--profile-source extracted` runs live candidate extraction too, and may fail closed on invalid
  citations.

Both metrics are LLM-judge estimates and vary between runs (about ±0.04). The recorded
labeled-fixture baseline (mean faithfulness 0.7028, answer relevancy 0.6980) is in
[`evaluation/results/matching_ragas_baseline.json`](../evaluation/results/matching_ragas_baseline.json).
Live outputs such as `matching_ragas_final.json` are git-ignored.

## Interpretation

The compatibility score is a deterministic weighted heuristic, not an employment probability or a
hiring decision. A skill the CV does not mention is unknown, not unmet, and a year-only range such
as `2023 to 2025` does not establish a month-precise duration. Citation checks do not prove that
model prose is semantically correct, which is why faithfulness around 0.70 is reported as a
limitation.
