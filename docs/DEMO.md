# Demo Guide (No External APIs)

This demo is designed to score well on the hackathon’s **Real-Time Capability & Dynamism** criteria.
It uses the built-in simulator (no News API keys).

## What you’ll show (2–3 minutes)
1. Backend is running and watching a folder
2. A new fact appears (file ADD)
3. The fact changes (file UPDATE) and the answer changes
4. The source is deleted (file DELETE) and the answer becomes “not found”

## Pre-reqs (local)
- Docker Desktop running
- Ollama running on your machine (Windows): `http://localhost:11434`
  - Models pulled once:
    - `nomic-embed-text` (embeddings)
    - `llama3.2` (generation)

## Start the system
From repo root:

```bash
docker compose up --build
```

Open:
- Backend health: `http://localhost:8000/health`
- Frontend (optional):
  - `cd frontend`
  - `npm install`
  - `npm run dev -- --host`

## Demo script (what to say + do)
### 1) Prove liveness is enabled
- Visit `GET /health` and point out:
  - `watching` is `/app/data`
  - `collection_count` increases over time

### 2) Ask the question while the simulator is in v1
- Ask:
  - "When is Product Alpha launching?"
- Expected:
  - Answer cites `breaking_news.txt` (launch date appears)

### 3) Wait for the UPDATE event and ask again
- Wait ~15s for the simulator to update `breaking_news.txt`
- Ask the same question again
- Expected:
  - Answer changes to the corrected launch date

### 4) Wait for the DELETE event and ask again
- Wait ~15s for the simulator to delete `breaking_news.txt`
- Ask the same question again
- Expected:
  - "I cannot find that information in the documents."

## Quick troubleshooting (during recording)
- If queries are slow, reduce generation size in `docker-compose.yml`:
  - `OLLAMA_GEN_MAX_PREDICT=128`
  - `OLLAMA_GEN_NUM_CTX=1024`
- If you want more time between events, increase:
  - `SIM_INTERVAL_S=20` or `30`

## Upload the video to GitHub (easy + public)
Option A (recommended): GitHub Release asset
1. Push the repo
2. Create a Release in GitHub
3. Upload `demo.mp4` as an asset
4. Paste the asset URL into the README at `Demo URL: ...`

Option B: attach video to a GitHub Issue/PR
1. Open an Issue in your repo
2. Drag-and-drop the `.mp4` into the comment box
3. GitHub will upload and give you a URL
4. Paste that URL into README
