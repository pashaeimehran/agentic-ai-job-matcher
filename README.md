# Agentic AI Job Matcher

A Germany-focused AI job-matching project with structured job extraction, semantic CV search, and answers grounded in retrieved CV evidence.

**In development · Python · LLM applications · PostgreSQL/pgvector**

The current version provides a local CLI, the extraction/retrieval foundation, deterministic Job–CV matching, and a framework-free tool-calling agent loop. LangGraph orchestration and a web API are planned.

## Problem & Motivation

Job descriptions and CVs express requirements and experience in different formats. Comparing them systematically requires extracting consistent information and finding evidence for a candidate's qualifications.

This project builds that foundation: it converts job text into validated fields and retrieves relevant CV passages. Answers cite checked source quotations, allowing a user to inspect the evidence behind a response. An end-to-end ranking workflow is the next stage, not a completed capability.

## Approach

### Structured job extraction

Job text is cleaned, sent through a provider-independent LLM interface, and validated against Pydantic schemas. An optional BA Jobsuche client supplies job listings; committed fixtures support repeatable development without that service.

### CV ingestion and retrieval

1. Parse a PDF, Markdown, or text CV.
2. Split the normalized content into deterministic chunks, using section boundaries where available.
3. Generate OpenAI embeddings and store chunks in PostgreSQL with pgvector.
4. Embed a question and retrieve the closest chunks using CV-scoped cosine search.
5. Ask the LLM to select verbatim evidence. Check that returned source IDs and quotations belong to the retrieved chunks; reject invalid citations and represent missing evidence explicitly.

The implementation separates LLM providers, embeddings, storage, and retrieval services. Database replacement is transactional, and retrieval checks embedding-model compatibility. The default chunk size is 900 characters with 150-character overlap; these are character counts, not token counts.

See the [retrieval architecture](docs/cv_retrieval_architecture.md) for implementation details and boundaries.

### Tool calling and manual agent loop

Five validated tools (`search_jobs`, `get_job_details`, `search_cv`, `calculate_match`, `save_job`) wrap the existing job fixtures, CV retrieval, and deterministic matching. A bounded manual loop lets the LLM choose tools while Python validates arguments, executes, classifies failures, and retries only retryable errors. Match scores come only from deterministic Python. See the [tool-calling architecture](docs/tool_calling_agent_architecture.md).

## Technologies

| Technology | Role |
|---|---|
| Python 3.11+ | CLI, extraction, ingestion, and evaluation |
| OpenAI API | Structured LLM outputs and semantic embeddings |
| Pydantic | Input and output schema validation |
| PostgreSQL 16 and pgvector | Persistent CV chunks and exact vector search |
| Docker Compose | Local database service |
| pypdf | Text extraction from digitally generated PDFs |
| pytest | Offline regression and opt-in database integration tests |

The example configuration uses `gpt-4o-mini` and `text-embedding-3-small`. Agent frameworks, a public API, and cloud deployment are not part of this version.

## How to Run

### 1. Install the project

```bash
git clone https://github.com/pashaeimehran/agentic-ai-job-matcher.git
cd agentic-ai-job-matcher
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements-retrieval.txt
```

Run the remaining commands from the repository root.

### 2. Try the offline checks

These commands do not require an API key or running database:

```bash
python -m pytest
python -m scripts.evaluate_cv_retrieval --output evaluation/results/retrieval_offline_local.json
python -m scripts.evaluate_tool_selection --mode replay --output evaluation/results/tool_selection_replay_local.json
```

The default test run skips the opt-in database test unless its database configuration is supplied. The offline evaluator uses lexical hash embeddings and in-memory storage; it exercises retrieval logic rather than OpenAI semantic retrieval. It writes a local report for comparison with the committed benchmark.

### 3. Configure live retrieval

For a fresh clone, copy `.env.example` to `.env`. If `.env` already exists, preserve it and add any missing settings.

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` to set `OPENAI_API_KEY`, `POSTGRES_PASSWORD`, and `DATABASE_URL`. Use the same database password in the password variable and connection URL. For initial setup, letters, numbers, and underscores avoid URL-escaping issues. Keep `.env` out of commits.

Install and start Docker with Compose support, then initialize the database:

```bash
docker compose up -d --wait
python -m scripts.cv init-db
```

The database binds to local port 5432. Docker preserves its data in a named volume; changing `.env` does not reset an existing database password.

### 4. Run the synthetic CV example

The following commands make paid OpenAI calls. Ingestion sends CV chunks for embeddings; Q&A also sends retrieved chunks for answer selection.

```bash
python -m scripts.cv ingest data/fixtures/sample_cv.md --cv-id demo
python -m scripts.cv search "What is the German language level?" --cv-id demo
python -m scripts.cv ask "What is the German language level?" --cv-id demo
python -m scripts.cv ask "What salary does the candidate expect?" --cv-id demo
```

The fixture contains German B1 evidence. The salary question has no supporting answer and is intended to exercise abstention. Search displays chunk IDs and text; Q&A displays selected quotations or an insufficient-evidence response. Reusing a CV ID replaces its stored content.

### 5. Run live evaluation

```bash
python -m scripts.evaluate_cv_retrieval --live --output evaluation/results/retrieval_live_local.json
python -m scripts.evaluate_job_extraction --limit 0
```

Both commands use paid APIs. The retrieval evaluator creates temporary CV records, compares four chunk configurations, cleans up those records, and checks the default configuration against an 80% Hit@3 gate. The extraction evaluator runs all fixture cases and writes `evaluation/results/job_extraction_evaluation.json`, replacing that report if it exists.

Database integration checks and troubleshooting are documented in the [existing setup guide](docs/apply_cv_retrieval.md#3-configure-local-postgresql). Its first section describes a historical patch installer; a fresh clone should use the setup steps above.

## Results & Limitations

### Recorded component evaluation

Results below are preserved in the repository's reports. They are not an evaluation of end-to-end job matching.

| Component | Recorded result | Scope |
|---|---|---|
| Job extraction | 40/40 exact fixture passes | 40 controlled variants across 10 output scenarios |
| Offline lexical retrieval | Hit@3: 95%; MRR: 0.867 | Synthetic CV, 20 questions, 900/150 character chunking |
| Live semantic retrieval | Hit@3: 100%; MRR: 0.975 | Same synthetic benchmark and chunking |
| Offline tests | 36 passed in recorded local validation | Regression checks reported in project documentation |
| PostgreSQL integration | Passed in recorded local validation | Database behavior, including retrieval and replacement |
| Live Q&A smoke checks | Source-backed German B1 answer; salary abstention | Two checks, not a comprehensive answer-quality evaluation |

Hit@3 measures whether expected evidence occurs in the top three retrieved chunks. MRR measures how highly the first relevant chunk is ranked. The synthetic benchmark is a development set, not an independent test of generalization to unseen CVs.

Evidence: [benchmark summary](benchmarks/retrieval_baseline.md), [semantic retrieval report](evaluation/results/retrieval_semantic.json), [offline retrieval report](evaluation/results/retrieval_offline.json), and [extraction report](evaluation/results/job_extraction_evaluation.json).

### Current boundaries

- LangGraph orchestration and the web API remain planned. The manual agent searches a small synthetic fixture job catalog, not live job boards.
- Q&A validates quotations and source IDs, but these checks do not establish that a quotation answers the question correctly. Broader relevance and abstention evaluation remains necessary.
- Scanned PDFs need OCR before ingestion; multi-column PDF reading order may need review.
- The CLI is a local single-user application, without multi-user authentication or authorization.
- Start with the synthetic CV. Keep personal files in ignored `data/private/`; live processing sends their text to OpenAI.
- The optional BA Jobsuche client uses an unofficial backend whose availability is not guaranteed. Fixtures remain the repeatable development input.

## Repository Guide

| Directory | Purpose |
|---|---|
| `src/extraction/` | Job cleaning and structured extraction |
| `src/llm/` | Provider-independent structured generation |
| `src/rag/` | CV ingestion, embeddings, storage, retrieval, and Q&A |
| `src/data_sources/` | Optional live source and fixture access |
| `src/matching/` | Deterministic, evidence-aware Job–CV matching |
| `src/tools/` | Validated tool schemas, registry, and tool implementations |
| `src/agent/` | Framework-free, bounded tool-calling agent loop |
| `scripts/` | CLI and evaluation entry points |
| `tests/` | Offline unit tests and opt-in integration test |
| `data/fixtures/` | Synthetic examples and evaluation labels |
| `benchmarks/` and `evaluation/results/` | Recorded measurements and reports |
| `docs/` | Architecture and detailed setup notes |

## License

[MIT License](LICENSE). Copyright © 2026 Mehran Pashaei.
