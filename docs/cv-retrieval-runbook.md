# CV retrieval runbook

How to run, ingest into and evaluate the CV retrieval component. Design and evaluation method:
[cv-retrieval.md](cv-retrieval.md).

Run every command from the repository root inside the project virtual environment
(`python -m pip install -r requirements-dev.txt -c constraints.txt`). On Windows PowerShell, use
backtick line continuations instead of `\`.

## Offline evaluation (no key, no database)

```bash
python -m pytest
python -m scripts.evaluate_cv_retrieval
```

The evaluator ingests the synthetic CV (`data/fixtures/sample_cv.md`) with deterministic lexical
hash embeddings and scores 20 labeled questions across four chunk-size configurations. The gate
checks the default configuration (900-character chunks, 150-character overlap) against Hit@3 ≥ 80%.
Offline numbers measure chunking and evidence placement, not semantic quality. Recorded results
for both modes: [`benchmarks/retrieval_baseline.md`](../benchmarks/retrieval_baseline.md).

## PostgreSQL and live retrieval (paid)

Requirements: Docker, and a `.env` copied from `.env.example` with `OPENAI_API_KEY`,
`POSTGRES_PASSWORD` and a matching `DATABASE_URL` (use `127.0.0.1`, not `localhost`, on Windows).

```bash
docker compose up -d --wait postgres
python -m scripts.cv init-db
python -m scripts.cv ingest data/fixtures/sample_cv.md --cv-id synthetic-demo
python -m scripts.cv search "What is the German language level?" --cv-id synthetic-demo
python -m scripts.cv ask "What is the German language level?" --cv-id synthetic-demo
python -m scripts.evaluate_cv_retrieval --live          # 20-question semantic benchmark
```

`ingest`, `search`, `ask` and `--live` call OpenAI. The synthetic CV yields 13 chunks with the
default settings. Ingesting again under the same CV ID replaces that CV's document and chunks in
one transaction; `python -m scripts.cv delete --cv-id <id>` removes them.

`ask` answers only with verbatim quotations from retrieved chunks and abstains when the evidence is
insufficient (for example, a salary question against the synthetic CV).

## PostgreSQL integration tests

The default test run skips the database tests. Point them at a disposable pgvector database:

```bash
docker run -d --rm --name jma-test -e POSTGRES_DB=test -e POSTGRES_USER=test \
  -e POSTGRES_PASSWORD=test -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16
TEST_DATABASE_URL=postgresql://test:test@127.0.0.1:5433/test python -m pytest
docker stop jma-test
```

They cover search, CV isolation, replacement, rollback and deletion with test CV IDs. CI runs them
on every push against a pgvector service container. A skipped integration test is not a pass.

## Operational notes

- `docker compose down` keeps the database volume; `down -v` deletes it. The volume keeps the
  password it was created with, so changing `POSTGRES_PASSWORD` later requires resetting the volume.
- Search refuses a CV whose stored embedding model or dimensions differ from the configured ones
  until the CV is ingested again.

## Privacy

Keep real CVs in the git-ignored `data/private/` directory and use a distinct CV ID. In live mode,
CV text is sent to OpenAI for embeddings and, for Q&A and matching, for model inference. Deleting
database rows does not remove the original file, shell history, backups or provider-side records.
See [data-protection.md](data-protection.md).

Digitally generated PDFs, Markdown and plain text are supported. Scanned PDFs need OCR first.
Retrieval Hit@3 does not measure answer faithfulness, ranking accuracy or hiring likelihood.
Matching is covered in [matching-runbook.md](matching-runbook.md).
