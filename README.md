# rag-engine

A self-hosted **RAG engine** with hybrid retrieval, parent/child chunking and source-grounded answers. No UI, no accounts, no vendor lock-in: you bring your own LLM and embedding keys.

- **Backend only** — a clean JSON API plus a CLI.
- **Bring your own keys** — any OpenAI-compatible chat API (OpenAI, OpenRouter, Groq, Ollama, LM Studio, vLLM, Gemini compat…) and either local ONNX embeddings or any OpenAI-compatible `/embeddings` endpoint.
- **Zero-config default** — SQLite + local embeddings: `pip install`, ingest, ask. No external service required.
- **Postgres + pgvector** when you need scale.
- **Grounded answers** — every answer comes with numbered sources, page numbers when available, and scores.
- **Built for real documents** — PDF, DOCX, DOC (via LibreOffice), PPTX, TXT, Markdown.

Requires Python 3.11+ (3.12 recommended; the Docker image uses 3.12).

```text
PDF / DOCX / PPTX / TXT / MD
             │
      parse + structure-aware chunking
             │
      parent section (~2000 chars) ──► sent to the LLM as context
             │
      child chunks (~500 chars) ──► embedded + indexed
             │
   query ──► dense search + BM25/FTS ──► RRF fusion ──► LLM rerank ──► answer + sources
```

## Quickstart

```bash
git clone https://github.com/Simouw/rag-engine.git
cd rag-engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set LLM_API_KEY and LLM_MODEL (embeddings work offline by default)
```

Start the API:

```bash
uvicorn rag_engine.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000/docs> for the interactive API, then ingest the sample document:

```bash
python -m rag_engine ingest ./examples/docs --collection demo
```

Ask a question:

```bash
curl -s http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does photosynthesis store energy?","top_k":5}' | jq
```

Or upload through the API:

```bash
curl -s http://127.0.0.1:8000/api/documents \
  -F 'file=@./examples/docs/photosynthesis.md' \
  -F 'collection=demo' \
  -F 'doc_type=notes' \
  -F 'tags=biology,plants' | jq
```

## Bring your own keys

### Generation (`LLM_*`)

| Provider | `LLM_BASE_URL` | `LLM_MODEL` example |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-3.5-sonnet` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Gemini (OpenAI compat) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.2` |
| LM Studio (local) | `http://localhost:1234/v1` | your loaded model |
| vLLM / any gateway | your `/v1` URL | your model |

Set `LLM_API_KEY` for remote providers. Local servers usually accept an empty key.

### Embeddings (`EMBEDDING_*`)

Default: **local fastembed model**, no API key, CPU only, downloaded once into `EMBEDDING_CACHE_DIR`.

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384
```

Or use any OpenAI-compatible embeddings endpoint:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

> **Dimension is part of the database schema.** Changing the embedding model or dimension requires a fresh database (or a migration) and a full reindex. The engine validates the returned vector size and fails loudly on mismatch.

## API

All routes are under `/api`. When `RAG_API_KEY` is set, send `Authorization: Bearer <key>` or `X-API-Key: <key>`. `/api/health` stays public.

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/ready` | Database, embedding and LLM configuration |
| `GET` | `/api/stats` | Document/chunk counts and collections |
| `POST` | `/api/documents` | Upload a file and index it |
| `GET` | `/api/documents` | List documents (`collection`, `doc_type`, `status`, `limit`, `offset`) |
| `GET` | `/api/documents/{id}` | Document metadata |
| `GET` | `/api/documents/{id}/file` | Download the original file |
| `GET` | `/api/documents/{id}/chunks` | List chunks (`include_parents`, `include_disabled`) |
| `PATCH` | `/api/documents/{id}/chunks/{chunk_id}` | Enable/disable or edit a child chunk |
| `POST` | `/api/documents/{id}/reindex` | Re-parse, re-chunk and re-embed |
| `DELETE` | `/api/documents/{id}` | Delete a document and its file |
| `POST` | `/api/search` | Hybrid retrieval, no generation |
| `POST` | `/api/ask` | Retrieve, generate and return sources |

### Search

```bash
curl -s http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "oxygen production under light",
    "top_k": 5,
    "filters": {"collection": "demo", "tags": ["plants"]}
  }' | jq
```

### Ask

```bash
curl -s http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What does the light reaction produce?",
    "top_k": 5,
    "filters": {"collection": "demo"},
    "history": [
      {"role": "user", "content": "Explain photosynthesis."},
      {"role": "assistant", "content": "It converts light into chemical energy."}
    ],
    "include_context": true
  }' | jq
```

Response shape:

```json
{
  "answer": "The light reaction produces ATP and NADPH [1].",
  "sources": [
    {
      "index": 1,
      "chunk_id": "...",
      "document_id": "...",
      "title": "photosynthesis",
      "content": "…",
      "score": 0.031,
      "page_start": 2,
      "page_end": 2,
      "metadata": {"section_path": ["Light reaction"], "chunk_role": "child"}
    }
  ],
  "model": "gpt-4o-mini",
  "usage": {"prompt_tokens": 1200, "completion_tokens": 140, "total_tokens": 1340},
  "retrieved": 3,
  "rewritten_query": null
}
```

`history` is stateless: the engine uses it to rewrite follow-up questions into standalone search queries, then discards it. Nothing is persisted.

## CLI

```bash
python -m rag_engine initdb                          # create tables
python -m rag_engine ingest ./docs --collection my-docs
python -m rag_engine ingest ./docs --dry-run         # list files without writing
python -m rag_engine ingest ./docs --no-embed        # chunk only (lexical search only)
python -m rag_engine stats
python -m rag_engine reset --collection my-docs --yes
```

Directories are scanned recursively. The collection defaults to the folder name. Sub-directory names become tags, which lets you filter retrieval:

```bash
python -m rag_engine ingest ./corpus/biology --collection science --tags spring-2026
```

Re-running the CLI on the same files updates documents in place (stable `source_key`), replacing their chunks instead of duplicating them.

## Configuration

Everything lives in `.env` (see `.env.example`). The most useful settings:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/rag.db` | SQLite or `postgresql+asyncpg://…` |
| `AUTO_CREATE_TABLES` | `true` | Create tables on startup |
| `RAG_API_KEY` | empty | Optional bearer/header API key |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI defaults | Generation |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.2` / `1200` | Generation controls |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | multilingual MiniLM / `384` | Embeddings |
| `EMBEDDING_BATCH_SIZE` | `64` | Embedding request batch size |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Child chunks |
| `PARENT_CHUNK_SIZE` | `2000` | Parent sections |
| `TOP_K` | `5` | Default number of sources |
| `HYBRID_SEARCH` | `true` | Dense + lexical + RRF |
| `CANDIDATES_PER_MODE` | `20` | Candidates per retrieval mode |
| `RERANK_ENABLED` | `true` | LLM reranking of the top candidates |
| `RERANK_TOP_N` | `10` | Reranked candidates |
| `PARENT_CONTEXT_ENABLED` | `true` | Send the full parent section to the LLM |
| `FTS_LANGUAGE` | `simple` | PostgreSQL text-search config |
| `MAX_UPLOAD_MB` | `100` | Upload limit |

## PostgreSQL + pgvector

SQLite is fine up to a few thousand documents and is perfect for local use. For larger corpora:

```bash
docker compose up -d db

export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag
python -m rag_engine initdb
uvicorn rag_engine.main:app --port 8000
```

The startup routine tries to run `CREATE EXTENSION IF NOT EXISTS vector`, then creates:

- a GIN index on `to_tsvector('<FTS_LANGUAGE>', content)` for lexical search;
- an HNSW index on the embedding column for dense search.

Managed databases may require an administrator to enable the `vector` extension first. Index creation failures are logged and do not prevent the API from starting; search still works, just more slowly.

## Docker

```bash
cp .env.example .env        # fill in LLM settings
docker compose up --build
```

The compose file starts the API and a `pgvector/pgvector:pg16` database. Data and uploaded files are persisted in named volumes.

Build the image alone:

```bash
docker build -t rag-engine .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/data:/app/data" rag-engine
```

## How retrieval works

1. The query is embedded.
2. **Dense search** returns the closest child chunks (pgvector cosine on Postgres, Python cosine on SQLite).
3. **Sparse search** runs in parallel: PostgreSQL full-text search (`ts_rank_cd`) or BM25 Okapi on SQLite.
4. Both ranked lists are merged with **Reciprocal Rank Fusion** (`k = 60`).
5. The top candidates can be **reranked by the LLM**; failures fall back to the fused order.
6. Each hit is replaced by its **parent section** for the prompt, while the source list still points at the precise chunk and page.
7. The model answers with inline `[n]` citations and a strict "say you don't know" rule.

Chunking is structure-aware: headings are detected (Markdown, numbered, roman, lettered, keywords, uppercase titles) and every section is split into a parent (~2000 chars) and children (~500 chars with overlap). Child embeddings include the title and section path so retrieval understands the surrounding context.

## Project layout

```text
rag_engine/
├── api/             # routes, API-key dependency, error payloads
├── documents/       # parsers, recursive chunker, structure-aware chunker
├── llm/             # chat client, embedding clients, prompts
├── retrieval/       # vector store, BM25/FTS, RRF + rerank pipeline
├── services/        # ingestion and ask orchestration
├── cli.py           # initdb / ingest / reset / stats
├── config.py        # pydantic-settings configuration
├── db.py            # async engine, session factory, Postgres indexes
├── models.py        # documents and parent/child chunks
├── schemas.py       # API contracts
└── main.py          # FastAPI app
tests/               # pytest, offline (no network, deterministic embeddings)
```

## Tests and lint

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

Tests use an in-memory-style temporary SQLite database and monkeypatched embeddings/LLM: no API key and no network required.

## Security notes

- Keys never leave the server and are never returned by an endpoint.
- Set `RAG_API_KEY` before exposing the service; use TLS (reverse proxy) in front of it.
- Uploaded documents are **untrusted input**. Prompt injection can still influence an LLM even with grounding rules; do not feed documents you would not trust a model to read.
- There is no user/account system. If you need multi-tenant isolation, deploy one instance per tenant or put your own auth and per-tenant databases in front.

## Limitations and roadmap

Not included today, on purpose:

- no frontend, no accounts, no billing;
- non-streaming answers (SSE is a natural next step);
- local filesystem storage only (S3/GCS adapters are straightforward to add);
- synchronous ingestion (fine for individual documents; use the CLI for batches);
- no OCR of scanned images inside PDFs.

## License

MIT — see [LICENSE](LICENSE).
