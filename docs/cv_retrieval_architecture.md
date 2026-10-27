# CV ingestion and semantic retrieval

## Data flow

`PDF/Markdown/text -> normalized CVDocument -> CVChunk list -> embeddings -> PostgreSQL`

`Question -> embedding -> CV-filtered cosine search -> source chunks -> extractive answer`

The reader handles digitally generated PDFs. It rejects encrypted PDFs and any
page without extractable text, so scanned or mixed PDFs need OCR first. PDF
columns may produce a poor reading order: inspect search output or export
Markdown. Markdown headings form section boundaries; plain text and PDFs use
character windows when headings are absent. Defaults are 900 characters and
150-character overlap, bounded at 2,000 characters per input. These are character
limits, not token counts. Source offsets refer to normalized text.

`CVDocument` validates the document ID, source name, text and content hash.
`CVChunk` records source offsets, section, order and a deterministic identifier.
This layer stores source evidence, not a permanent LLM-generated career
profile. The matching layer extracts source-attributed candidate facts from
stored chunks when required; the **labeled-fixture Ragas mode** instead uses
versioned synthetic facts after verifying the exact stored CV and chunks.
Neither mode changes the original document/chunk schema or requires a new
PostgreSQL migration. See [matching architecture](job_cv_matching_architecture.md).

## Dependencies and boundaries

`CVService` depends on `EmbeddingProvider` and `VectorStore` protocols.
`OpenAIEmbeddings` uses ordered, validated batches of 32 and finite nonzero
vectors. `HashEmbeddings` is a deterministic lexical test substitute. It does
not understand synonyms and must not be advertised as semantic retrieval.

`MemoryVectorStore` supports offline tests. `PostgresVectorStore` provides the
persistent implementation. Each CV records embedding model and dimensions.
Search refuses mismatched configurations until the CV is re-ingested.
Embeddings are generated before a write transaction. The transaction replaces
metadata and chunks together; a failed insert rolls back. Row locks serialize
replacements and prevent search from reading mismatched metadata and chunks.
Deleting a document cascades to its chunks. Exact search is appropriate for a
small CV corpus; no approximate vector index is needed yet.

The database contains source chunks and vectors, but no separate copy of the
full original file. CV IDs scope every search. This is a local single-user CLI,
not an authentication or tenant authorization system. A future API must add
access control before exposing it to multiple users.

## Grounding and limitations

Q&A selects verbatim excerpts instead of writing unconstrained prose. Every
returned ID must belong to retrieved evidence and every quote must occur in
that chunk. Invalid citations fail closed. Missing evidence yields an explicit
abstention. These checks prevent invented quotations, but cannot prove that a
real quote answers the question or is factually true. Relevance and abstention
quality still require live evaluation. Source text is treated as untrusted data
in the generation instructions; prompt instructions alone are not a security
boundary.

OpenAI requests transmit CV text. The CLI does not log raw content in error
messages; search and ask intentionally display selected evidence. Real CVs
belong in ignored `data/private/`. Deletion removes database records, not the
original input, terminal history, database backups or provider-side retention.
The Docker volume persists across `docker compose down` (without `-v`).

## What the evaluation measures

Twenty manually labeled questions specify a section and an exact source quote.
For each configuration the evaluator computes expected chunk IDs from those
labels. A hit requires one of those IDs in the top k. A chunk containing only
part of the quote is not counted. Evidence coverage records whether any chunk
preserves the full quote. Missing coverage is counted as failure, never dropped
from the denominator. MRR measures the rank of the first relevant chunk.

The 64-character configuration is deliberately small and illustrates evidence
fragmentation. Larger configurations compare normal chunking choices. The
acceptance gate checks the default 900/150 configuration against 80% hit@3,
not whichever configuration scored best. This is a synthetic development set,
not an independent hold-out benchmark; add unseen CVs before generalizing.

## Reference documentation

- [pgvector operators and exact search](https://github.com/pgvector/pgvector)
- [OpenAI embeddings API](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)
- [pypdf text extraction and OCR limitations](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
