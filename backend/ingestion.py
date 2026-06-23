from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.settings import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS
from backend.vector_store import add_documents


_DOCUMENTS: dict[str, dict[str, Any]] = {}
_DOCUMENT_LOCK = RLock()


def clean_text(text: str) -> str:
    """Normalize whitespace without changing the meaning of the source text."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _public_status(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": record["document_id"],
        "filename": record["filename"],
        "status": record["status"],
        "chunks_indexed": record.get("chunks_indexed", 0),
        "message": record.get("message"),
        "error": record.get("error"),
        "uploaded_at": record["uploaded_at"],
    }


def register_document(document_id: str, filename: str, stored_path: Path) -> dict[str, Any]:
    record = {
        "document_id": document_id,
        "filename": filename,
        "stored_path": str(stored_path),
        "status": "queued",
        "chunks_indexed": 0,
        "message": "Document queued for ingestion.",
        "error": None,
        "uploaded_at": _now(),
    }
    with _DOCUMENT_LOCK:
        _DOCUMENTS[document_id] = record
    return _public_status(record)


def update_document_status(document_id: str, **updates: Any) -> None:
    with _DOCUMENT_LOCK:
        if document_id in _DOCUMENTS:
            _DOCUMENTS[document_id].update(updates)


def get_document_status(document_id: str) -> dict[str, Any] | None:
    with _DOCUMENT_LOCK:
        record = _DOCUMENTS.get(document_id)
        return _public_status(record) if record else None


def list_document_statuses() -> list[dict[str, Any]]:
    with _DOCUMENT_LOCK:
        return [_public_status(record) for record in _DOCUMENTS.values()]


def _load_documents(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}. Supported types: {sorted(SUPPORTED_EXTENSIONS)}")


def _metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return str(value)


def _prepare_chunks(documents: list[Document], document_id: str, filename: str) -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(documents)

    chunks: list[Document] = []
    ids: list[str] = []
    for index, chunk in enumerate(split_docs):
        text = clean_text(chunk.page_content)
        if not text:
            continue

        metadata = {key: _metadata_value(value) for key, value in chunk.metadata.items()}
        metadata["document_id"] = document_id
        metadata["source"] = filename
        metadata["chunk"] = index

        if "page" in metadata and isinstance(metadata["page"], int):
            metadata["page"] = metadata["page"] + 1

        chunks.append(Document(page_content=text, metadata=metadata))
        ids.append(f"{document_id}:{index}")

    return chunks, ids


def ingest_document(document_id: str, stored_path: Path, filename: str) -> None:
    """Load, chunk, embed, and index one uploaded document."""
    try:
        update_document_status(
            document_id,
            status="processing",
            message="Extracting text and building embeddings.",
            error=None,
        )
        documents = _load_documents(stored_path)
        chunks, ids = _prepare_chunks(documents, document_id, filename)
        if not chunks:
            raise ValueError("No extractable text was found in the uploaded document.")

        indexed_count = add_documents(chunks, ids)
        update_document_status(
            document_id,
            status="complete",
            chunks_indexed=indexed_count,
            message=f"Indexed {indexed_count} chunks.",
            error=None,
        )
    except Exception as exc:
        update_document_status(
            document_id,
            status="failed",
            message="Document ingestion failed.",
            error=str(exc),
        )

