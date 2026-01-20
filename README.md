# Live RAG Console (Pathway + Chroma + Ollama)

A live-ingestion Retrieval-Augmented Generation stack with Pathway streaming into Chroma, Ollama for embeddings + generation, and a React/Vite frontend that cites sources.

##  Demo video
- Record a 2–3 minute screen recording following [docs/DEMO.md](docs/DEMO.md).
- Upload the `.mp4` somewhere accessible (recommended: GitHub Release asset), then paste the public URL here.

Demo URL: <ADD_YOUR_VIDEO_URL_HERE>

## Features
- Live Pathway file watcher (`data/`) → chunked ingestion → Chroma persistent collection (Docker volume).
- Grounded answers with citations (doc + page; chunk IDs are hidden in the UI).
- FastAPI backend with health/metrics and CORS open for local dev.
- React/Vite UI with status pills, latency display, depth slider (`k`), and source cards.
- Dockerized backend for reproducible builds; frontend runs locally via Vite.

## Prerequisites
- Docker Desktop
- Ollama running locally
	- Pull models once: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

## Quick Start

### Backend (Docker)
From the repo root:
```bash
docker compose up --build
```

Backend listens on `http://localhost:8000`.

### Frontend (Vite)
```bash
cd frontend
npm install
npm run dev -- --host
```

Frontend listens on `http://localhost:5173`.

## Live ingestion without any external API (recommended for demo)
This repo includes a built-in simulator that continuously **adds, updates, and deletes** files inside the watched `data/` folder.

Run everything:
```bash
docker compose up --build
```

The `simulator` container will create/update/delete files like `breaking_news.txt`.
This lets you demonstrate “Live AI” liveness without NewsAPI/GNews/etc.

Tuning (optional) in [docker-compose.yml](docker-compose.yml) (simulator env vars):
- `SIM_MODE=scripted` (predictable add → update → delete) or `SIM_MODE=random`
- `SIM_INTERVAL_S=15` (seconds between events)
- `SIM_DELETE_PROB=0.15` (only used in random mode)

## Configuration (optional)
- `VITE_BACKEND_URL` in `frontend/.env` or shell (defaults to `http://localhost:8000`).
- Ollama endpoints: `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434` inside Docker).
- Models: `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`), `OLLAMA_GEN_MODEL` (default `llama3.2`).

## Ingest documents
- Drop PDFs or text files into `data/`. Pathway streams them and writes chunks to Chroma with embeddings.

## Health
- `GET /health` returns `{ status, watching, collection_count, expected_dim }`.

## Project Structure
- `rag_backend.py` — Pathway pipeline, chunking, retrieval, FastAPI routes.
- `frontend/` — React/Vite UI (App, status pills, chat messages, styles).
- `data/` — Watched input docs.
- `docker-compose.yml`, `Dockerfile` — Backend container + simulator.

## Troubleshooting
- Backend up but empty results: confirm files exist in `data/` and `collection_count > 0` via `/health`.
- Ollama unreachable: start Ollama and pull models `llama3.2` and `nomic-embed-text`.
- Frontend errors: ensure `VITE_BACKEND_URL` matches backend host/port.

## License
MIT 
