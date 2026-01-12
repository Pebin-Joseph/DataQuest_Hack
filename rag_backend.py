import hashlib
import json
import os
import threading
import time as time_module
from pathlib import Path
from typing import List, Dict

import pathway as pw
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.errors import InvalidArgumentError

# ------------
# Parsing helpers
# ------------

def parse_document(content: bytes, filepath: str) -> List[Dict[str, str]]:
    path_obj = Path(filepath)
    name = (path_obj.name or "unknown").strip().strip('"').strip("'") or "unknown"
    rows: List[Dict[str, str]] = []

    def chunk_text(
        text: str,
        chunk_size_words: int = 120,
        overlap_words: int = 30,
        max_chars: int = 1200,
    ) -> List[str]:
        words = text.split()
        if len(words) <= 1:
            step = max_chars - 200 if max_chars > 400 else max_chars
            return [text[i : i + max_chars] for i in range(0, len(text), step) if text[i : i + max_chars].strip()]

        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size_words
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                if len(chunk) > max_chars:
                    chunk = chunk[:max_chars]
                chunks.append(chunk)
            start = end - overlap_words
            if start < 0:
                start = 0
        return chunks

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("utf-16", errors="ignore")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")
    text = text.strip()
    if text:
        for chunk in chunk_text(text):
            rows.append({"text": chunk, "page": 1, "doc": name})
    return rows


def _path_to_str(val) -> str:
    if isinstance(val, bytes):
        try:
            s = val.decode("utf-8", errors="ignore")
            s = (s or "").strip().strip('"').strip("'")
            return s or "unknown"
        except Exception:
            return "unknown"
    try:
        s = os.fspath(val)
        s = (s or "").strip().strip('"').strip("'")
        return s or "unknown"
    except Exception:
        s = str(val) if val is not None else "unknown"
        s = (s or "").strip().strip('"').strip("'")
        return s or "unknown"


def _row_field(row, key: str, default=None):
    """Best-effort access for Pathway rows which can be dict-like or attribute-like."""
    if row is None:
        return default
    try:
        if isinstance(row, dict):
            return row.get(key, default)
    except Exception:
        pass
    try:
        # Some Pathway row objects support __getitem__ but not .get
        return row[key]
    except Exception:
        pass
    try:
        if hasattr(row, key):
            return getattr(row, key)
    except Exception:
        pass
    try:
        getter = getattr(row, "get", None)
        if callable(getter):
            return getter(key, default)
    except Exception:
        pass
    return default

# ------------
# Embedding + vector store helpers
# ------------

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "60"))
OLLAMA_GEN_TIMEOUT = int(os.getenv("OLLAMA_GEN_TIMEOUT", "45"))
OLLAMA_GEN_MAX_PREDICT = int(os.getenv("OLLAMA_GEN_MAX_PREDICT", "256"))
OLLAMA_GEN_TEMPERATURE = float(os.getenv("OLLAMA_GEN_TEMPERATURE", "0.2"))
OLLAMA_GEN_NUM_CTX = int(os.getenv("OLLAMA_GEN_NUM_CTX", "2048"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "vector_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "live_docs")

_http = requests.Session()
_http.mount(
    "http://",
    HTTPAdapter(
        max_retries=Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
    ),
)
_http.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
    ),
)

# Track indexing activity so queries can reduce load while ingesting
_INGESTING = threading.Event()
_LAST_INGEST_TS = 0.0

# Avoid multiple concurrent /api/generate calls (helps on CPU-only Ollama)
_GEN_SEMAPHORE = threading.Semaphore(int(os.getenv("OLLAMA_GEN_CONCURRENCY", "1")))

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
_expected_dim = None
_CHROMA_LOCK = threading.Lock()

# Best-effort cleanup for legacy ingests where path metadata could not be extracted.
try:
    with _CHROMA_LOCK:
        _collection.delete(where={"doc": "unknown"})
except Exception:
    pass

def embed_text(text: str) -> List[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    for attempt in range(1, 4):
        try:
            resp = _http.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json=payload,
                timeout=(10, OLLAMA_EMBED_TIMEOUT),
            )
            resp.raise_for_status()
            emb = resp.json().get("embedding", [])
            if emb:
                return emb
            else:
                print(
                    f"EMBED_EMPTY attempt={attempt} model={OLLAMA_EMBED_MODEL} len={len(text)}"
                )
        except Exception as e:
            print(
                f"EMBED_FAIL attempt={attempt} model={OLLAMA_EMBED_MODEL} len={len(text)} err={e}"
            )
        time_module.sleep(0.25 * attempt)
    return []


def _get_expected_dim() -> int | None:
    global _expected_dim
    if _expected_dim is not None:
        return _expected_dim
    try:
        peek = _collection.peek(1) or {}
        embeddings = peek.get("embeddings", [])
        if embeddings:
            _expected_dim = len(embeddings[0])
            return _expected_dim
    except Exception:
        return None
    return None


def _lexical_score(text: str, query: str) -> int:
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return 0
    lower_text = text.lower()
    return sum(lower_text.count(t) for t in terms)


def _fallback_scan(query: str, limit: int):
    try:
        data = _collection.get(include=["documents", "metadatas", "ids"], limit=200)
    except Exception:
        return [], []

    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    ids = data.get("ids", []) or []

    scored = []
    for i, text in enumerate(docs):
        score = _lexical_score(text or "", query)
        if score == 0:
            continue
        meta = metas[i] if i < len(metas) else {}
        cid = ids[i] if i < len(ids) else ""
        scored.append((score, text, meta, cid))

    if not scored:
        return [], []

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]

    contexts = []
    sources = []
    for _, text, meta, cid in top:
        doc = meta.get("doc", "") if isinstance(meta, dict) else ""
        page = meta.get("page", 1) if isinstance(meta, dict) else 1
        sources.append({"doc": doc, "page": int(page), "chunk_id": cid})
        contexts.append(f"[doc={doc} page={page}] {text}")
    return contexts, sources

# ------------
# Pathway pipeline
# ------------

@pw.udf
def parse_udf(content: bytes, path: str):
    return parse_document(content, path)

def make_chunk_id(doc: str, page: int, text: str) -> str:
    digest = hashlib.sha1()
    digest.update(doc.encode("utf-8"))
    digest.update(str(page).encode("utf-8"))
    digest.update(text[:2000].encode("utf-8"))
    return digest.hexdigest()

def sink_to_chroma(chunk_id: str, doc: str, page: int, text: str):
    if len(text) > 1200:
        print(
            f"TRUNCATE chunk={chunk_id[:8]} doc={doc} page={page} from_len={len(text)} to=1200"
        )
        text = text[:1200]

    embedding = embed_text(text)
    if not embedding:
        # If Ollama embedding failed, skip this chunk to avoid Chroma auto-downloading a different model
        print(f"SKIP_EMBED: no embedding for chunk={chunk_id[:8]} doc={doc} page={page} len={len(text)}")
        return

    expected_dim = _get_expected_dim()
    if expected_dim is None:
        # Establish expected dimension from first successful embedding
        expected_dim = len(embedding)
        globals()["_expected_dim"] = expected_dim
    if len(embedding) != expected_dim:
        # Skip mismatched embeddings to avoid collection errors
        print(
            f"SKIP_DIM: chunk={chunk_id[:8]} doc={doc} page={page} len={len(text)} "
            f"embed_dim={len(embedding)} expected={expected_dim}"
        )
        return

    _metadata = {"doc": doc, "page": page}
    try:
        with _CHROMA_LOCK:
            _collection.upsert(
                ids=[chunk_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[_metadata],
            )
    except InvalidArgumentError:
        # Defensive: skip rows that still violate dimension or other schema issues
        return
    except Exception as e:
        # Chroma can raise InternalError on some filesystems (e.g., disk I/O on bind mounts).
        # Don't crash the ingestion thread; log and skip.
        print(f"CHROMA_UPSERT_FAIL chunk={chunk_id[:8]} doc={doc} page={page} err={e}")
        return


def sink_ingest(row):
    # row has fields: data (bytes), _metadata
    path_val = None
    try:
        path_val = row._metadata["path"]
    except Exception:
        try:
            path_val = row._metadata.path
        except Exception:
            path_val = "unknown"

    for chunk in parse_document(row.data, path_val):
        cid = make_chunk_id(chunk["doc"], chunk["page"], chunk["text"])
        sink_to_chroma(cid, chunk["doc"], chunk["page"], chunk["text"])


class SinkIngestObserver(pw.io.python.ConnectorObserver):
    def on_change(self, key, row, time, is_addition):
        # Pathway may surface file path either as a top-level `path` column
        # or inside `_metadata.path` depending on version/connector.
        path_val = _row_field(row, "path", "unknown")
        if path_val in (None, "unknown"):
            metadata = _row_field(row, "_metadata", {}) or {}
            path_val = _row_field(metadata, "path", "unknown")

        path_val = _path_to_str(path_val)
        if path_val not in (None, "unknown"):
            path_val = os.path.basename(path_val).strip().strip('"').strip("'")

        if not is_addition:
            # Remove all chunks for this doc when the source file is deleted
            try:
                with _CHROMA_LOCK:
                    _collection.delete(where={"doc": path_val})
            except Exception:
                pass
            return

        # If this is an update event, Pathway will still surface it as an addition.
        # To avoid stale chunks lingering (old content still being retrieved),
        # remove existing vectors for this doc before re-ingesting.
        try:
            with _CHROMA_LOCK:
                _collection.delete(where={"doc": path_val})
        except Exception:
            pass

        global _LAST_INGEST_TS
        _INGESTING.set()
        _LAST_INGEST_TS = time_module.time()

        data_bytes = _row_field(row, "data", b"") or b""

        try:
            for chunk in parse_document(data_bytes, path_val):
                cid = make_chunk_id(chunk["doc"], chunk["page"], chunk["text"])
                sink_to_chroma(cid, chunk["doc"], chunk["page"], chunk["text"])
        finally:
            _LAST_INGEST_TS = time_module.time()
            _INGESTING.clear()

def build_pipeline(data_dir: str = "data"):
    files = pw.io.fs.read(data_dir, format="binary", mode="streaming", with_metadata=True)
    # Simpler path: write each file row to a Python sink that parses and ingests
    pw.io.python.write(files, SinkIngestObserver())

# ------------
# FastAPI service
# ------------

app = FastAPI(title="Live RAG Backend", version="0.1")

# CORS for local dev (React/Streamlit frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    k: int = 6

class Source(BaseModel):
    doc: str
    page: int
    chunk_id: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


def _retrieve(query: str, k: int):
    # Use our own embedding to avoid dimension mismatch with Chroma defaults
    q_emb = embed_text(query)
    if not q_emb:
        return _fallback_scan(query, k)

    n_results = max(k * 2, k + 2)
    results = _collection.query(query_embeddings=[q_emb], n_results=n_results)
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return _fallback_scan(query, k)

    scored = []
    for i, text in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        doc = meta.get("doc", "")
        page = meta.get("page", 1)
        chunk_id = ids[i] if i < len(ids) else ""
        lexical = _lexical_score(text or "", query)
        semantic = 0.0
        if distances and i < len(distances):
            try:
                semantic = 1.0 / (1.0 + float(distances[i]))
            except Exception:
                semantic = 0.0
        combined = semantic + 0.35 * lexical
        scored.append((combined, text, doc, page, chunk_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    sources = []
    contexts = []
    for _, text, doc, page, chunk_id in top:
        sources.append({"doc": doc, "page": int(page), "chunk_id": chunk_id})
        contexts.append(f"[doc={doc} page={page}] {text}")
    return contexts, sources


def _generate_answer(question: str, context: str) -> str:
    prompt = (
        "Role: You are a helpful assistant. Answer ONLY with information found in the context. "
        "If the answer is not in the context, reply: I cannot find that information in the documents. "
        "Cite doc name and page in parentheses when possible.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    # If indexing is running, reduce generation size to improve responsiveness.
    max_predict = 128 if _INGESTING.is_set() else OLLAMA_GEN_MAX_PREDICT
    payload = {
        "model": OLLAMA_GEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": OLLAMA_GEN_TEMPERATURE,
            "num_predict": max_predict,
            "num_ctx": OLLAMA_GEN_NUM_CTX,
        },
    }
    try:
        with _GEN_SEMAPHORE:
            resp = _http.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=(10, OLLAMA_GEN_TIMEOUT),
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.Timeout:
        raise HTTPException(status_code=503, detail="LLM timeout: generation took too long")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM request failed: {e}")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    k = max(1, min(body.k, 12))
    contexts, sources = _retrieve(body.question, k)
    if not contexts:
        return {
            "answer": "I cannot find that information in the documents.",
            "sources": [],
        }

    context_block = "\n\n".join(contexts)
    # Cap context to avoid oversized prompts hanging the LLM
    if len(context_block) > 6000:
        context_block = context_block[:6000]

    try:
        answer = _generate_answer(body.question, context_block)
    except HTTPException:
        # Pass through as-is so FastAPI returns the intended status/detail
        raise
    except Exception as e:
        # Catch-all to prevent 500s on unexpected LLM errors
        raise HTTPException(status_code=503, detail=f"LLM generate failed: {e}")

    return {"answer": answer, "sources": sources}

@app.get("/health")
def health():
    ollama = {"ok": False}
    try:
        r = _http.get(f"{OLLAMA_BASE_URL}/api/version", timeout=(2, 2))
        if r.ok:
            v = r.json() if "application/json" in (r.headers.get("content-type") or "") else {}
            ollama = {"ok": True, "version": v.get("version")}
        else:
            ollama = {"ok": False, "status_code": r.status_code}
    except Exception as e:
        ollama = {"ok": False, "error": str(e)}

    try:
        count = _collection.count()
    except Exception:
        count = None
    return {
        "status": "ok",
        "watching": os.path.abspath("data"),
        "collection_count": count,
        "expected_dim": _get_expected_dim(),
        "ingesting": _INGESTING.is_set(),
        "last_ingest_ts": _LAST_INGEST_TS,
        "ollama": ollama,
    }

# ------------
# Entry point
# ------------

def _run_pathway():
    build_pipeline("data")
    pw.run()

if __name__ == "__main__":
    threading.Thread(target=_run_pathway, daemon=True).start()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
