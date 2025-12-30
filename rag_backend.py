import hashlib
import json
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Dict

import pathway as pw
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.errors import InvalidArgumentError

# ------------
# Parsing helpers
# ------------

def _parse_pdf_with_unstructured(content: bytes, filename: str) -> List[Tuple[str, int]]:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        return []
    elements = partition_pdf(file=BytesIO(content), include_page_breaks=True, strategy="hi_res")
    chunks: List[Tuple[str, int]] = []
    for el in elements:
        page_num = getattr(el.metadata, "page_number", None) or 1
        text = el.text.strip() if hasattr(el, "text") else ""
        if text:
            chunks.append((text, page_num))
    return chunks

def _parse_pdf_fallback(content: bytes, filename: str) -> List[Tuple[str, int]]:
    try:
        import pdfplumber
    except ImportError:
        return []
    chunks: List[Tuple[str, int]] = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append((text, i))
    return chunks

def parse_document(content: bytes, filepath: str) -> List[Dict[str, str]]:
    path_obj = Path(filepath)
    name = path_obj.name
    suffix = path_obj.suffix.lower()
    rows: List[Dict[str, str]] = []

    def chunk_text(text: str, chunk_size: int = 220, overlap: int = 60) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    if suffix == ".pdf":
        pdf_chunks = _parse_pdf_with_unstructured(content, name)
        if not pdf_chunks:
            pdf_chunks = _parse_pdf_fallback(content, name)
        for text, page in pdf_chunks:
            for chunk in chunk_text(text):
                rows.append({"text": chunk, "page": page, "doc": name})
    else:
        # Treat everything else as utf-8 text
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
            return val.decode("utf-8", errors="ignore") or "unknown"
        except Exception:
            return "unknown"
    try:
        return os.fspath(val)
    except Exception:
        return str(val) if val is not None else "unknown"

# ------------
# Embedding + vector store helpers
# ------------

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PATH = os.getenv("CHROMA_PATH", "vector_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "live_docs")

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
_expected_dim = None

def embed_text(text: str) -> List[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except Exception:
        # If the embed endpoint fails, skip embedding so ingestion continues.
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
    embedding = embed_text(text)
    expected_dim = _get_expected_dim()
    if embedding:
        if expected_dim is None:
            # Establish expected dimension from first successful embedding
            expected_dim = len(embedding)
            globals()["_expected_dim"] = expected_dim
        if len(embedding) != expected_dim:
            # Skip mismatched embeddings to avoid collection errors
            return
    _metadata = {"doc": doc, "page": page}
    try:
        _collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding] if embedding else None,
            metadatas=[_metadata],
        )
    except InvalidArgumentError:
        # Defensive: skip rows that still violate dimension or other schema issues
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
        if not is_addition:
            return

        metadata = row.get("_metadata", {}) or {}
        if isinstance(metadata, dict):
            path_val = metadata.get("path", "unknown")
        else:
            path_val = getattr(metadata, "path", "unknown")

        path_val = _path_to_str(path_val)
        if path_val not in (None, "unknown"):
            path_val = os.path.basename(path_val)

        data_bytes = row.get("data", b"") or b""

        for chunk in parse_document(data_bytes, path_val):
            cid = make_chunk_id(chunk["doc"], chunk["page"], chunk["text"])
            sink_to_chroma(cid, chunk["doc"], chunk["page"], chunk["text"])

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
    payload = {"model": "llama3.2", "prompt": prompt, "stream": False}
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    k = max(1, min(body.k, 12))
    contexts, sources = _retrieve(body.question, k)
    context_block = "\n\n".join(contexts) if contexts else ""
    answer = _generate_answer(body.question, context_block)
    return {"answer": answer, "sources": sources}

@app.get("/health")
def health():
    try:
        count = _collection.count()
    except Exception:
        count = None
    return {
        "status": "ok",
        "watching": os.path.abspath("data"),
        "collection_count": count,
        "expected_dim": _get_expected_dim(),
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
