# Job–CV matching evaluation baseline

Evaluation mode: **offline, manually labeled synthetic candidate and jobs**.
No paid API, real resumes, or PostgreSQL were used for this measurement.

Command: `python -m scripts.evaluate_job_matching`

| Metric | Result |
|---|---:|
| Labeled job/CV pairs | 6 |
| Labeled requirement outcomes | 25 |
| Supported/unmet/unknown agreement | 25/25 (100%) |
| Deterministic score replay | Pass |
| Vector-only lexical hash Hit@3 | 5/6 (83.3%) |
| Hybrid lexical hash Hit@3 | 6/6 (100.0%) |
| Vector-only Precision@3 | 27.8% |
| Hybrid Precision@3 | 33.3% |
| Vector-only Recall@3 | 83.3% |
| Hybrid Recall@3 | 100.0% |

The reported 100% outcome agreement shows the rules match *this hand-labeled
development fixture*, not accuracy on real applications. The hash-embedding
baseline does not measure real semantic search. Precision@3 is necessarily
low when only one relevant labeled chunk exists per job. The reference quote
is required to occur **in full** in a same-section chunk. No results are
excluded from denominators for insufficient source coverage.

Offline deterministic job ranking:

| Rank | Synthetic job | Compatibility score |
|---:|---|---:|
| 1 | Machine Learning Engineer | 95.68 |
| 2 | Python Backend Engineer | 95.11 |
| 3 | Robotics Engineer | 70.76 |
| 4 | Cloud Platform Engineer | 53.45 |
| 5 | Senior Platform Engineer | 41.26 |
| 6 | C++ Security Engineer | 28.53 |

The scores are arbitrary weighted matching heuristics, *not* hiring odds or
independent recommendation quality. A useful next validation is a separate,
unseen labeled CV/job set (including negative skills, ambiguous proficiency,
and contextual work requirements), followed by a controlled live semantic
retrieval comparison. The separately measured paid Ragas explanation baseline
appears below.

## Paid Ragas explanation-quality baseline (October 21, 2026)

This **separate** experiment ran in a Python 3.12.10 environment with Ragas
0.4.3 against three top-ranked synthetic jobs. The stored synthetic CV was
validated against the exact fixture and the candidate profile came from
versioned, manually labeled facts (`--profile-source labeled-fixture`).
Ranking used deterministic Python. Only the selected top three jobs received
paid LLM explanations. Ragas contexts contained retrieved chunks plus the
source chunks approved in comparisons and cited claims.

| Job ID | Faithfulness | Answer relevancy | Context chunks |
|---|---:|---:|---:|
| `match-02-ml-engineer` | 0.7273 | 0.6813 | 5 |
| `match-01-python-backend` | 0.7143 | 0.7497 | 5 |
| `match-03-robotics` | 0.6667 | 0.6629 | 5 |
| **Mean** | **0.7028** | **0.6980** | — |

Reviewed numeric report:
[`evaluation/results/matching_ragas_baseline.json`](../evaluation/results/matching_ragas_baseline.json).

Reproduction (paid OpenAI calls; requires ingested `synthetic-demo` CV):

```powershell
$evalPython = Join-Path $HOME '.venvs/job-match-ragas-py312/Scripts/python.exe'
& $evalPython -m scripts.evaluate_matching_ragas `
  --cv-id synthetic-demo `
  --limit 3 `
  --profile-source labeled-fixture `
  --output evaluation/results/matching_ragas_final.json
```

This experiment evaluates explanation quality with a fixed source-validated
profile, **not** end-to-end extraction reliability, real-world job matching,
or hiring accuracy. Judge outputs can vary between runs. The mean faithfulness
score (~0.70) indicates some claims were judged insufficiently supported.
Earlier one-job diagnostics used different explanation formatting or context
policies and must not be described as a controlled before/after comparison.
