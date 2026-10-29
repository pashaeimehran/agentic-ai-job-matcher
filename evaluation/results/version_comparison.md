# Version comparison: prompt v1 vs v2 (live)

Generated 2026-10-28T23:15:36+00:00. Agent: LangGraph. Cache: disabled. 6 labeled scenarios x 5 repeat(s) per version, model `gpt-4o-mini`.

**What changed:** Prompt-only change. v2 adds explicit definitions of supported / unmet / unknown and forbids describing an unknown requirement as lacking, missing, a gap, a weakness or unmet. Tools, deterministic matching and scoring, model, limits and scenarios are unchanged.

| Metric | v1 (baseline) | v2 (candidate) |
|---|---:|---:|
| Task success rate | 0.8667 | 0.9667 |
| Completed runs | 1.0 | 1.0 |
| Runs with an unknown requirement described as a deficit | 4 | 1 |
| Deficit-described unknown requirements (total) | 16 | 5 |
| Unknown requirements mentioned in the answer | 0.9615 | 0.9615 |
| Runs with ungrounded scores | 0 | 0 |
| calculate_match covered the requested jobs | 1.0 | 1.0 |
| Mean tool calls | 1.6 | 2.367 |
| Mean model turns | 2.167 | 2.167 |
| Wall latency p50 (ms) | 3734.4 | 3931.5 |
| Wall latency p95 (ms) | 9949.9 | 11691.9 |
| Mean tokens per query | 3329.2 | 3800.9 |
| Mean estimated cost per query (USD) | 0.000632 | 0.000733 |
| Total estimated cost (USD) | 0.018967 | 0.022002 |

Deterministic score consistency across all runs and both versions: **True** (11 jobs scored).
Fisher exact two-sided p-value for runs with a deficit-described unknown: 0.3533.

## Per scenario (runs with a deficit-described unknown / task success rate)

| Scenario | v1 | v2 |
|---|---:|---:|
| gaps-llm-engineer | 0/5 · 1.0 | 0/5 · 1.0 |
| lacking-genai-developer | 0/5 · 1.0 | 0/5 · 1.0 |
| weaknesses-cpp-security | 0/5 · 1.0 | 0/5 · 1.0 |
| unmet-and-unknown-platform | 0/5 · 1.0 | 0/5 · 1.0 |
| compare-gaps-two-jobs | 2/5 · 0.6 | 1/5 · 0.8 |
| search-and-explain-germany | 2/5 · 0.6 | 0/5 · 1.0 |

Runs were re-judged on 2026-10-28T23:17:43+00:00 with the current detector revision (same stored answers, same rules for both versions). Every flagged sentence is listed in the JSON report under runs.<version>[].flagged_sentences for audit.

## Limitations

5 live sample(s) per scenario and version on synthetic data with one model; the lexical deficit detector is conservative and can miss paraphrases. Treat differences as indicative, not statistically conclusive.

Task success = run completed with an answer, calculate_match covered the requested jobs, no ungrounded score, and no unknown requirement described as a deficit. The deficit check is `src/agent/grounding.py` (lexical, deterministic).
