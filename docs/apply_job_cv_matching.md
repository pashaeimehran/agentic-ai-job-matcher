# Run and evaluate Job–CV matching and ranking

This guide describes the **implemented** matching component. It supersedes
the historical ZIP installer instructions from the initial feature delivery.
It assumes you are in the repository root with the feature files present.

## 1. Offline validation and demonstration

The offline mode uses a manually labeled synthetic candidate profile and
hash-based lexical embeddings. It never claims to parse arbitrary CVs or to
measure semantic embedding quality.

```powershell
$python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
& $python -m pytest -q
& $python -m scripts.match_jobs --offline --top 3 --explain
& $python -m scripts.evaluate_job_matching
```

The 6-job development fixture includes 25 labeled requirement outcomes.
The recorded offline evaluation achieved 25/25 correct supported/unmet/unknown
statuses; vector-only hash Hit@3 was 5/6, and hybrid hash Hit@3 was 6/6.
These are synthetic development measurements, not general matching accuracy.
See [the matching benchmark](../benchmarks/matching_baseline.md).

## 2. Live matching with PostgreSQL and OpenAI

The main project virtual environment was validated on Windows with Python
3.14.6. Docker Desktop should be running. Preserve the existing `.env`
(never commit it), PostgreSQL volume, and virtual environment.

```powershell
$python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
docker compose up -d --wait
& $python -m scripts.cv init-db
& $python -m scripts.cv ingest data/fixtures/sample_cv.md --cv-id synthetic-demo
& $python -m scripts.match_jobs --live --cv-id synthetic-demo --top 3 --explain
```

CV ingestion and live matching make **paid** OpenAI embedding and LLM calls.
The validated live run ranked all six synthetic jobs on October 21, 2026.
`--explain` prints the job-specific summary and grounded claims; never publish
terminal output from a real personal CV. `--output` is deliberately disallowed
in live mode to avoid inadvertently exporting private evidence.

The candidate fact extractor validates chunk membership and verbatim quotes,
repairs unsuitable skill quotes only if the original citation is valid,
and may retry extraction once. Year-only employment intervals are excluded
rather than inventing month boundaries, leaving experience unknown.
Live extraction can still intermittently reject invalid model-generated
citations; this is a known limitation, not a reason to bypass evidence checks.

The PostgreSQL integration test was passed separately on the validated
Windows setup; see [the CV retrieval guide](apply_cv_retrieval.md) for the
explicit test database command.

## 3. Optional Ragas explanation-quality evaluation

**Do not install `requirements-matching-eval.txt` into the main project `venv`.**
The validated evaluation setup uses a separate Python **3.12.10** virtual
environment, because the main environment pins OpenAI 3.x and the selected
Ragas 0.4.3 stack requires OpenAI below 3.

On the validated Windows machine, the isolated environment is located at
`$HOME/.venvs/job-match-ragas-py312`. It can be recreated anywhere; the path
is not part of the repository. Use a resolver such as `uv` or pip in that
isolated environment to install `requirements-matching-eval.txt`.

The preferred stable benchmark uses the **verified labeled synthetic profile**.
This validates that the ingested CV and chunk boundaries match the versioned
fixture, avoiding repeated variable LLM candidate extraction. It does *not*
measure full live profile-extraction reliability.

```powershell
$evalPython = Join-Path $HOME '.venvs/job-match-ragas-py312/Scripts/python.exe'
& $evalPython -m scripts.evaluate_matching_ragas `
    --cv-id synthetic-demo `
    --limit 3 `
    --profile-source labeled-fixture `
    --output evaluation/results/matching_ragas_final.json
```

The evaluator deterministically ranks the full fixture set, generates LLM
explanations only for the top N, then scores the rendered explanations using
Ragas Faithfulness and AnswerRelevancy. Contexts contain retrieved evidence
plus approved cited source chunks. Both metrics are judge estimates, not a
proof of correctness. Runs make **paid OpenAI calls** and can vary.

To exercise the more variable end-to-end candidate extraction path, use
`--profile-source extracted` instead. This mode may fail closed on invalid
source citations; it should be tested separately from explanation quality.

The verified October 21, 2026, labeled-fixture three-job baseline is:

| Job | Faithfulness | Answer relevancy |
|---|---:|---:|
| Machine Learning Engineer | 0.7273 | 0.6813 |
| Python Backend Engineer | 0.7143 | 0.7497 |
| Robotics Engineer | 0.6667 | 0.6629 |
| **Mean** | **0.7028** | **0.6980** |

The recorded numbers-only baseline is stored in
[`evaluation/results/matching_ragas_baseline.json`](../evaluation/results/matching_ragas_baseline.json).
The live generated `matching_ragas_final.json` stays local and is ignored by
Git. Use synthetic records only for public demonstrations.

## 4. Limitations and repository workflow

A compatibility score is a deterministic weighted heuristic, **not** an
employment probability or a hiring decision. A skill absent from a CV is
unknown, not explicitly unmet; `2023 to 2025` does not supply a verified
month-precise experience duration. Source-citation checks do not guarantee
that model prose is semantically correct, so the observed Ragas faithfulness
of approximately 0.70 is a limitation worth documenting.

Before committing, inspect `git status -sb`, `git diff --check`, and the
staged diff. Do not stage `.env`, `venv/`, private CVs, local patch ZIPs, or
unreviewed live evaluation outputs. Use a feature branch such as
`feat/job-cv-matching-ranking` and a conventional commit message such as
`feat: implement evidence-grounded job matching and ranking`.
