# Live RAG Console (Pathway + Chroma + Ollama)

A live-ingestion Retrieval-Augmented Generation stack with Pathway streaming to Chroma, Ollama for embeddings + generation, and a React/Vite frontend that cites doc sources.

## 🎥 Demo (GitHub overlay)
[![Watch the demo](docs/demo-thumb.png)](https://github.com/Pebin-Joseph/DataQuest_Hack/assets/PLACEHOLDER_DEMO_VIDEO)

- Replace `PLACEHOLDER_DEMO_VIDEO` with your uploaded GitHub asset URL (an `.mp4` plays inline/overlay on GitHub). 
- Optionally add `docs/demo-thumb.png` as a thumbnail poster.

## Features
- Live Pathway file watcher (`/data`) → chunked ingestion → Chroma persistent collection (`vector_db`).
- Hybrid retrieval: semantic (Ollama `nomic-embed-text`) + lexical rerank for vague queries; citations include doc + page.
- FastAPI backend with health/metrics and CORS open for local dev.
- React/Vite UI with status pills, latency display, depth slider (k), and source cards.
- Dockerized backend for reproducible builds; frontend runs locally via Vite.

## Quick Start
```bash
# Backend (Docker)
cd "C:\Users\HOME\Downloads\New Hackathon"
docker compose up --build
# Backend listens on http://localhost:8000

# Frontend
cd frontend
npm install
npm run dev -- --host  # Vite on http://localhost:5173
```

### Configure (optional)
- `VITE_BACKEND_URL` in `frontend/.env` or shell (defaults to `http://localhost:8000`).
- Ollama endpoints: `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`).
- Chroma paths: `CHROMA_PATH` (`vector_db`), `CHROMA_COLLECTION` (`live_docs`).

### Ingest documents
- Drop PDFs or text files into `data/`. Pathway streams them and writes chunks to Chroma with embeddings.
- Reset the index by removing `vector_db/` before a rebuild (already handled in recent runs).

### Querying
- Frontend textbox → `/query` with adjustable `k` (default 6, clamped 1–12).
- Backend prompt is context-only: if not found, it responds "I cannot find that information in the documents."

### Health
- `GET /health` returns `{ status, watching, collection_count, expected_dim }`.

## Project Structure
- `rag_backend.py` — Pathway pipeline, chunking, hybrid retrieval, FastAPI routes.
- `frontend/` — React/Vite UI (App, status pills, chat messages, styles).
- `data/` — Watched input docs.
- `vector_db/` — Chroma persistent store (generated at runtime).
- `docker-compose.yml`, `Dockerfile` — Backend container.

## Retrieval Quality Tips
- Increase the depth slider (k) for vague queries.
- Ask with doc names/sections when possible; smaller chunks (220 tokens, 60 overlap) improve recall.
- If embeddings fail (e.g., Ollama offline), a lexical fallback still surfaces matches.

## Troubleshooting
- Backend up but empty results: confirm files exist in `data/` and `collection_count > 0` via `/health`.
- Dimension mismatch: ensured by custom query embeddings; if issues persist, clear `vector_db/` and rebuild.
- Frontend CORS/errors: ensure `VITE_BACKEND_URL` matches backend host/port.
- Ollama unreachable: start Ollama and pull models `llama3.2` and `nomic-embed-text`.

## Deploy/Publish
- Commit and push to `main`:
```bash
git add .
git commit -m "docs: add README and UX tweaks"
git push origin main
```

## License
MIT (or your preferred license).
