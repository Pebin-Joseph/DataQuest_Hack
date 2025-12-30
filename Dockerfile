# Backend container for Live RAG (Pathway + FastAPI + Chroma)
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps for PDF parsing
# Switch apt sources to HTTPS to avoid 403s in some networks
RUN sed -i 's|http://deb.debian.org/debian|https://deb.debian.org/debian|g' /etc/apt/sources.list.d/debian.sources \
  && sed -i 's|http://security.debian.org/debian-security|https://deb.debian.org/debian-security|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential poppler-utils ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy code and mount points
COPY rag_backend.py ./
COPY data ./data
# Ensure runtime dirs exist; vector_db will be mounted/created at runtime
RUN mkdir -p data vector_db

# Python deps
RUN PIP_NO_CACHE_DIR=1 pip install \
  --default-timeout=600 --retries 10 --use-deprecated=legacy-resolver \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  -i https://pypi.org/simple \
  fastapi==0.110.0 \
  uvicorn==0.23.2 \
  pathway==0.27.1 \
  chromadb==1.3.7 \
  unstructured==0.10.30 \
  pdfplumber==0.11.0 \
  requests==2.31.0

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_EMBED_MODEL=nomic-embed-text \
    CHROMA_PATH=vector_db \
    CHROMA_COLLECTION=live_docs

EXPOSE 8000
CMD ["python", "rag_backend.py"]
