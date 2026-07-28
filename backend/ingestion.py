from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.settings import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS
from backend.vector_store import add_documents

_DOCS: dict[str, dict[str, Any]] = {}
_LOCK = RLock()


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in (
        "document_id", "filename", "status",
        "chunks_indexed", "message", "error", "uploaded_at",
    )}


def _update(document_id: str, **fields: Any) -> None:
    with _LOCK:
        if document_id in _DOCS:
            _DOCS[document_id].update(fields)


# ---------------------------------------------------------------------------
# Public registry API
# ---------------------------------------------------------------------------

def register_document(document_id: str, filename: str, stored_path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "document_id":   document_id,
        "filename":      filename,
        "stored_path":   str(stored_path),
        "status":        "queued",
        "chunks_indexed": 0,
        "message":       "Document queued for ingestion.",
        "error":         None,
        "uploaded_at":   _now(),
    }
    with _LOCK:
        _DOCS[document_id] = record
    return _public(record)


def get_document_status(document_id: str) -> dict[str, Any] | None:
    with _LOCK:
        record = _DOCS.get(document_id)
        return _public(record) if record else None


def list_document_statuses() -> list[dict[str, Any]]:
    with _LOCK:
        return [_public(r) for r in _DOCS.values()]


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")


def _safe_value(v: Any) -> str | int | float | bool:
    return v if isinstance(v, (str, int, float, bool)) else ("" if v is None else str(v))


def _chunk(documents: list[Document], document_id: str, filename: str) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks, ids = [], []
    for i, raw in enumerate(splitter.split_documents(documents)):
        text = "\n".join(ln.strip() for ln in raw.page_content.splitlines() if ln.strip())
        if not text:
            continue
        meta = {k: _safe_value(v) for k, v in raw.metadata.items()}
        meta.update(document_id=document_id, source=filename, chunk=i)
        if isinstance(meta.get("page"), int):
            meta["page"] += 1          # LangChain uses 0-based page numbers
        chunks.append(Document(page_content=text, metadata=meta))
        ids.append(f"{document_id}:{i}")
    return chunks, ids


def ingest_document(document_id: str, stored_path: Path, filename: str) -> None:
    """Load, chunk, embed, and index one uploaded document (runs in background)."""
    try:
        _update(document_id, status="processing", message="Extracting text and building embeddings.", error=None)
        docs = _load(stored_path)
        chunks, ids = _chunk(docs, document_id, filename)
        if not chunks:
            raise ValueError("No extractable text found in the uploaded document.")
        count = add_documents(chunks, ids)
        _update(document_id, status="complete", chunks_indexed=count, message=f"Indexed {count} chunks.", error=None)
    except Exception as exc:
        _update(document_id, status="failed", message="Ingestion failed.", error=str(exc))
