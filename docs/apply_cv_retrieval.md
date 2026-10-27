# CV ingestion and semantic retrieval: setup and validation

This guide documents the implemented CV retrieval component. The original
installation package was merged into `main` at `0c1741c`; do not apply an old
installer or create the former `feat/cv-semantic-retrieval` branch again.

Run all commands from the **repository root**, using the project's virtual
environment. No absolute machine-specific project path is required.

## Prerequisites

- A supported Python environment with `requirements.txt` and
  `requirements-retrieval.txt` installed.
- Docker Desktop with the Linux container engine for PostgreSQL/pgvector.
- A local `.env` containing `OPENAI_API_KEY`, `OPENAI_MODEL`,
  `POSTGRES_PASSWORD`, `DATABASE_URL`, `EMBEDDING_MODEL`, and
  `EMBEDDING_DIMENSIONS`. Keep `.env` out of Git.
- Synthetic CV for demonstrations: `data/fixtures/sample_cv.md`.

On the validated Windows development machine, the main project used Python
3.14.6. Commands below use the explicitly selected `venv` executable; if the
virtual environment is recreated, install the repository's pinned requirements
before running the scripts.

## Offline validation

```powershell
$python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
& $python -m pytest -q
& $python -m scripts.evaluate_cv_retrieval
```

The retrieval evaluation uses a 20-question labeled synthetic CV and compares
several chunk-size configurations. The historical validated live default
(900-character chunks, 150-character overlap) achieved Hit@3 **20/20** and
MRR **0.975** with OpenAI embeddings; these numbers are documented in
[`benchmarks/retrieval_baseline.md`](../benchmarks/retrieval_baseline.md).
Offline lexical/hash results are not measurements of semantic quality.
The full repository suite includes newer matching tests, so do not expect the
original CV-only count of 36 offline tests.

## PostgreSQL and live retrieval

Start Docker Desktop first. These commands preserve the existing volume.

```powershell
$python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
docker compose up -d --wait
& $python -m scripts.cv init-db
& $python -m scripts.cv ingest data/fixtures/sample_cv.md --cv-id synthetic-demo
& $python -m scripts.cv search 'What is the German language level?' --cv-id synthetic-demo
& $python -m scripts.cv ask 'What is the German language level?' --cv-id synthetic-demo
```

Ingest, search, and ask use paid OpenAI calls. The synthetic CV is ingested
into persistent PostgreSQL/pgvector storage and yields 13 chunks with default
settings. Reusing a CV ID **replaces** the stored content for that ID.

To run the real database integration test, set an explicit test connection
string without printing the password:

```powershell
$python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
try {
    $env:TEST_DATABASE_URL = & $python -c "from dotenv import dotenv_values; print(dotenv_values('.env').get('DATABASE_URL', ''))"
    if (-not $env:TEST_DATABASE_URL) { throw 'DATABASE_URL missing from .env' }
    & $python -m pytest -q tests/integration/test_postgres_retrieval.py
    if ($LASTEXITCODE -ne 0) { throw 'Database test failed' }
}
finally {
    Remove-Item Env:TEST_DATABASE_URL -ErrorAction SilentlyContinue
}
```

The integration test passed separately on Windows on October 21, 2026.
It checks search, isolation, replacement, rollback, and deletion using test CV
IDs. An integration-test skip in the default suite is not an integration pass.

For a live 20-question semantic retrieval benchmark (paid calls):

```powershell
& $python -m scripts.evaluate_cv_retrieval --live
```

Do not delete the Docker volume when troubleshooting. A changed `.env`
password does not automatically change credentials inside an existing volume.

## Privacy and limitations

Keep personal CVs under ignored `data/private/`, and use a distinct CV ID.
CV content is transmitted to OpenAI for embeddings and, when using Q&A or
matching, for model inference. Deleting database rows does not remove the
original file, shell history, backups, or upstream provider records.

Digitally generated PDFs, Markdown, and text files are supported. Scanned PDFs
need OCR, which this reader does not perform. Retrieval Hit@3 is not answer
faithfulness, ranking accuracy, or hiring likelihood. The later matching
component is documented in [its application guide](apply_job_cv_matching.md).
