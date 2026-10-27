# CareerMatch AI / Job_Match_Agent

A Germany-focused job matching portfolio project. Implemented capabilities are
structured job extraction, CV ingestion with evidence retrieval, and
source-attributed Job–CV matching and ranking. Agent orchestration and the web
API remain planned.

## Implemented capabilities

- Job text cleaning, Pydantic validation, provider-independent structured outputs.
- Optional BA Jobsuche client and a controlled 40-case extraction fixture set.
- PDF, Markdown and text CV ingestion with deterministic section-aware chunks.
- Batched OpenAI embeddings, PostgreSQL/pgvector storage, exact cosine retrieval.
- CV-scoped search, atomic replacement, embedding configuration checks and deletion.
- Extractive CV Q&A with checked source IDs and verbatim quotations.
- Synthetic CV, 20 retrieval questions, chunk-size comparison and offline tests.
- Candidate facts with source-checked citations; explicit skill, CEFR and employment comparisons.
- Reproducible Python compatibility scores and deterministic job ranking.
- Hybrid keyword/cosine retrieval, metadata section filtering, lightweight reranking.
- Optional grounded explanation provider with evidence/status validation.
- Six synthetic matching cases, 25 labeled outcomes, precision/recall evaluation.

PostgreSQL integration and live OpenAI retrieval have been validated locally.
The default configuration achieved 20/20 hit@3 on the synthetic benchmark.
Both offline lexical and live semantic retrieval reports are included.

## Setup and commands

Use Python 3.11 or newer and a virtual environment. From the project root:

```powershell
python -m pip install -r requirements-retrieval.txt
python -m pytest
python -m scripts.evaluate_cv_retrieval
```

Preserve your existing `.env`; add the new database and embedding settings from
`.env.example`. Install and start Docker Desktop before database commands.

```powershell
docker compose up -d --wait
python -m scripts.cv init-db
python -m scripts.cv ingest data/fixtures/sample_cv.md --cv-id demo
python -m scripts.cv search "What is the German language level?" --cv-id demo
python -m scripts.cv ask "What is the German language level?" --cv-id demo
```

Ingest, search and ask use paid OpenAI calls. CV chunks are sent to OpenAI for
embeddings; ask sends retrieved chunks for answer selection. Start with the
synthetic fixture. Keep personal documents in ignored `data/private/`.
To try the matching workflow without database or API calls:

```powershell
python -m scripts.match_jobs --offline --top 3 --explain
python -m scripts.evaluate_job_matching
```

The zero-API demo uses *manual synthetic candidate fact labels* and hash
embeddings, not a general offline candidate parser. The live version extracts
source-checked facts from an ingested CV and makes paid LLM/embedding calls:

```powershell
python -m scripts.match_jobs --live --cv-id demo --top 3 --explain
```

See [the matching application guide](docs/apply_job_cv_matching.md),
[matching architecture](docs/job_cv_matching_architecture.md) and
[matching evaluation baseline](benchmarks/matching_baseline.md).
For optional paid Ragas metrics use a **separate Python 3.12 virtual
environment**; do not install `requirements-matching-eval.txt` in the
main project environment because of incompatible OpenAI SDK pins.
The earlier [CV retrieval guide](docs/apply_cv_retrieval.md) remains available.

## Validation status

Local Windows validation passed all 36 offline tests and the separate
PostgreSQL integration test. Dependency installation completed successfully.
Live Q&A returned a source-backed German B1 answer and abstained when asked
about salary, which the synthetic CV does not specify. These two checks are
smoke tests, not a comprehensive answer-quality evaluation.

Job–CV matching was installed and validated locally on Windows on October 21,
2026. All available offline tests passed (the default suite skipped the
opt-in database integration test); the PostgreSQL integration test passed when
run separately. Live matching completed a six-job ranking against the
synthetic CV. The stricter live candidate extractor may still reject invalid
LLM-generated citations intermittently; unverifiable year-only employment
periods are excluded rather than assigned invented months.

The six-job synthetic benchmark scored 25/25 labeled requirement outcomes.
Hybrid hash retrieval achieved 6/6 Hit@3 versus 5/6 for vector-only hash
retrieval. These are development-fixture results, not unseen-job accuracy.

Paid Ragas evaluation succeeded on the top three synthetic jobs using the
**verified labeled candidate fixture** and approved evidence contexts.
Mean faithfulness was **0.7028** and answer relevancy **0.6980**. This
measures explanation quality with judge estimates, not live candidate-fact
extraction accuracy. Faithfulness remains a documented limitation. See
[matching evaluation results](benchmarks/matching_baseline.md).

The supplied job extraction report records 40/40 exact fixture passes. It was
renamed without changing its measurements; no paid extraction run was repeated.
The set covers 40 variants across 10 output scenarios, not real-world accuracy.

```powershell
python -m scripts.evaluate_job_extraction --limit 0
```

That command makes paid calls and writes
`evaluation/results/job_extraction_evaluation.json`.

See [retrieval results](benchmarks/retrieval_baseline.md) and
[architecture](docs/cv_retrieval_architecture.md).

## Optional BA Jobsuche source

```powershell
python -m scripts.fetch_ba_jobs --query "Softwareentwickler" --location "Berlin" --limit 10
```

This uses an unofficial public backend described by the community
`bundesAPI/jobsuche-api` project, not a guaranteed developer API. HTTP 403 and
other failures are handled gracefully. Use `data/fixtures/jobs.json` for
repeatable development when the live source is unavailable.

## Repository layout

```text
src/data_sources/       Optional live job source and fixtures
src/extraction/         Structured job extraction
src/llm/                Structured generation provider interface
src/rag/                CV parsing, embeddings, stores, retrieval, Q&A
src/matching/           Candidate evidence, job requirements, comparisons, ranking
scripts/                Extraction, retrieval, matching and Ragas evaluation CLIs
tests/unit/             Offline regression tests
tests/integration/      Opt-in PostgreSQL checks
data/fixtures/          Synthetic public examples and evaluation labels
docs/                   Setup and architecture
benchmarks/             Measured results and limitations
```

## Next capabilities

Tool calling, manual agent loop, LangGraph, REST/MCP, production tracing and
cloud deployment are future phases. Match scores are heuristic aids and must
not be used for autonomous hiring or rejection decisions.

## License

[MIT](LICENSE). Copyright © 2026 Mehran Pashaei.
