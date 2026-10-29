# Evaluation

One entry point, `scripts/evaluate_system.py`, runs the dedicated evaluators and writes a single
report. The dedicated evaluators remain usable on their own.

```bash
python -m scripts.evaluate_system                     # deterministic, offline, no key (CI gate)
python -m scripts.evaluate_system --live              # + paid live tool selection and cache experiment
python -m scripts.compare_versions --mode offline     # version-comparison harness check
python -m scripts.compare_versions --mode live --repeats 5   # paid prompt v1 vs v2 comparison
```

Dedicated evaluators: `evaluate_cv_retrieval` (retrieval Hit@k / MRR), `evaluate_job_matching`
(requirement-status accuracy, vector vs hybrid retrieval), `evaluate_tool_selection` (replay or
live), `compare_agents` (manual loop vs LangGraph), `evaluate_job_extraction` (live), and
`evaluate_matching_ragas` (Ragas, in its own environment; see below).

## Kinds of evidence

Reports label every metric with one of four classes. They are not interchangeable:

| Class | Meaning | Examples |
|---|---|---|
| Deterministic offline | Same input, same number; no network. Tests code paths and labels, not model quality. | retrieval (hash embeddings), matching status accuracy, replay, orchestration equivalence, fixture end to end |
| Single live run | One sample of a non-deterministic model. Indicative only. | live tool selection, cache cold/warm experiment, extraction regression |
| Repeated live samples | Several samples per scenario and version, with a significance test | prompt v1 vs v2 (5 × 6 × 2 runs) |
| LLM judge | A model scoring another model's output. An estimate, not ground truth. | Ragas faithfulness, answer relevancy |

All data is synthetic: one synthetic CV, 16 fixture jobs, 6 labeled matching cases. These are
development sets, not evidence of generalization to real CVs.

## Results

### Deterministic offline (re-run 2026-10-29)

| Metric | Result |
|---|---|
| Retrieval Hit@3 / MRR (lexical hash embeddings, 900/150 chunks, 20 questions) | 95% / 0.867 |
| Requirement-status accuracy (25 labeled checks, 6 jobs) | 100% |
| Deterministic score reproducibility | true |
| Matching retrieval @3: vector vs hybrid (Hit, precision, recall) | 83.3/27.8/83.3% vs 100/33.3/100% |
| Tool-selection replay, manual loop and LangGraph (18 cases) | first-tool accuracy 100%, completion 100% |
| Manual loop vs LangGraph (29 scenarios) | 28 equivalent, 1 documented guard difference, identical scores |
| Termination behaviour covered | completed, model-turn limit, tool-call limit, provider error, invalid-request guard (graph) |
| Fixture-mode end to end (8 requests × 5, LangGraph + real tools, scripted provider) | task success 100%, p50 6 ms, p95 ≈ 13 ms |

Fixture-mode latency measures orchestration and tool execution only. No model is involved.

### Live (paid, `gpt-4o-mini`, synthetic data)

| Metric | Result | Class |
|---|---|---|
| Live semantic retrieval Hit@3 / MRR (`text-embedding-3-small`, 900/150) | 100% / 0.975 | single run |
| Job extraction (40 fixture variants) | 40/40 exact | single run |
| Live tool selection, manual loop (18 cases) | first-tool 100%, sequence 3/3, arguments valid 100% | single run |
| Live tool selection, LangGraph, cold cache | first-tool 100%, 0.000402 USD and 2,174 tokens per query, p50 2.4 s, p95 6.9 s | single run |
| Same, warm cache | 91.2% hit rate, 0.000027 USD per query, p50 6.5 ms | single run |

### Version comparison: prompt v1 vs v2

The controlled change: v2 adds explicit definitions of supported / unmet / unknown and forbids
describing an unknown requirement (no CV evidence either way) as lacking, missing, a gap or
unmet. Only the system prompt differs. Model, tools, deterministic scoring, limits and the 6
labeled scenarios are identical, and the cache is off. Runs are interleaved, 5 repeats per
scenario and version. A deterministic lexical detector (`src/agent/grounding.py`) flags deficit
wording about unknown requirements.

| Metric | v1 | v2 |
|---|---:|---:|
| Task success rate | 86.7% | 96.7% |
| Runs describing an unknown requirement as a deficit | 4 / 30 | 1 / 30 |
| Deficit-described unknown mentions | 16 | 5 |
| Runs with ungrounded scores | 0 | 0 |
| Mean tool calls | 1.60 | 2.37 |
| Mean cost per query | $0.000632 | $0.000733 |
| Wall latency p50 / p95 | 3.7 / 9.9 s | 3.9 / 11.7 s |

Deterministic scores were identical across all 60 runs. Fisher exact two-sided p = 0.35 for the
deficit-run difference: the direction is consistent but **not statistically significant** at this
sample size. Full report and every flagged sentence: `evaluation/results/version_comparison.md`
and `version_comparison_live.json`.

### Ragas (LLM judge)

Three top-ranked jobs, labeled synthetic profile, explanations from `gpt-4o-mini`, judged by Ragas
0.4.3 in an isolated Python 3.12 environment. The main environment's OpenAI SDK conflicts with
Ragas, so it is never installed there (`requirements-matching-eval.txt`,
`docs/apply_job_cv_matching.md`).

| Run | Faithfulness | Answer relevancy |
|---|---:|---:|
| Baseline, 2026-10-21 (`matching_ragas_baseline.json`) | 0.703 | 0.698 |
| 2026-10-28 (`matching_ragas_final.json`) | 0.744 | 0.672 |
| 2026-10-29 re-run | 0.676 | 0.689 |

The spread between runs (±0.04) is as large as any change between them, so the three runs
together show judge and model variance, not a trend.

## Limitations of these numbers

- Small synthetic sets. 100% on 25 labeled checks or 18 cases shows correctness on those cases,
  not a population accuracy.
- Replay and fixture metrics replay scripted model turns. They catch orchestration and tool
  regressions, not model regressions.
- Live results are mostly single runs on one model. Only the version comparison repeats samples,
  and its effect is not significant at n = 30 per arm.
- The deficit detector is lexical and can miss paraphrases. Its qualifier list was revised after
  a manual audit, and both versions were re-judged with the same revision.
- Estimated cost depends on the pricing table in `src/observability/pricing.py`.
