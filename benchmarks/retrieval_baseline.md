# CV retrieval validation

Measured on 19 October 2026 with a synthetic CV and 20 labeled questions.
Hit@3 measures whether expected evidence appears among the first three results.
MRR measures the rank of the first relevant result.

## Offline lexical baseline

Model: `lexical-hash-v1`; dimensions: 512.

| Chunk size | Overlap | Hit@3 | MRR |
|---|---|---|---|
| 64 | 12 | 50.0% | 0.392 |
| 120 | 25 | 95.0% | 0.783 |
| 300 | 60 | 95.0% | 0.867 |
| 900 | 150 | 95.0% | 0.867 |

## Live semantic retrieval

Model: `text-embedding-3-small`; dimensions: 1536.

| Chunk size | Overlap | Hit@3 | MRR |
|---|---|---|---|
| 64 | 12 | 75.0% | 0.550 |
| 120 | 25 | 100.0% | 0.950 |
| 300 | 60 | 100.0% | 0.975 |
| 900 | 150 | 100.0% | 0.975 |

## Additional checks

- The PostgreSQL integration test (search, isolation, replacement, rollback,
  deletion) passed; CI runs it on every push.
- Live Q&A returned a source-backed German B1 quotation.
- A salary question produced an insufficient-evidence response.
- The default live configuration passed the 80% Hit@3 acceptance gate.

These results describe a synthetic development benchmark, not accuracy on
unseen CVs. Q&A relevance and abstention need broader evaluation.

Reports: `evaluation/results/retrieval_offline.json` and
`evaluation/results/retrieval_semantic.json`.
