"""BM25 must stay in sync with Chroma (the authoritative chunk store).

Regression tests for:
- deleted documents lingering in BM25 (and in sparse/hybrid answers)
- startup self-heal when the two indexes drift apart
- BM25 save/load honouring a custom index_dir
- BM25 search never returning chunks with no lexical match
"""
import uuid
from pathlib import Path

from langchain_core.documents import Document

from backend.bm25_index import (
    BM25Index,
    ensure_bm25_consistent,
    get_bm25_index,
    rebuild_bm25_from_vector_store,
)
from backend.models import Chunk
from backend.vector_store import add_documents, delete_by_document


def _chroma_ids() -> set[str]:
    from backend.vector_store import get_vector_store
    return set(get_vector_store()._collection.get(include=[])["ids"])


def _add_doc(doc_id: str, text: str) -> list[str]:
    ids = [f"{doc_id}:0"]
    add_documents(
        [Document(page_content=text, metadata={"document_id": doc_id, "chunk": 0, "source": f"{doc_id}.txt"})],
        ids,
    )
    return ids


def test_rebuild_from_vector_store_matches_chroma_ids():
    _add_doc(f"sync_{uuid.uuid4().hex[:8]}", "zebracorn telemetry pipeline notes")
    rebuild_bm25_from_vector_store()
    assert get_bm25_index().get_chunk_ids() == _chroma_ids()


def test_ensure_consistent_heals_drift():
    rebuild_bm25_from_vector_store()
    doc_id = f"drift_{uuid.uuid4().hex[:8]}"
    _add_doc(doc_id, "quokkaflux appears only in this drifted chunk")  # Chroma only
    assert f"{doc_id}:0" not in get_bm25_index().get_chunk_ids()

    report = ensure_bm25_consistent()

    assert report["status"] == "rebuilt"
    assert f"{doc_id}:0" in get_bm25_index().get_chunk_ids()
    hits = get_bm25_index().search("quokkaflux", 5)
    assert hits and hits[0]["chunk_id"] == f"{doc_id}:0"
    assert ensure_bm25_consistent()["status"] == "ok"


def test_delete_document_removes_chunks_from_bm25(tmp_path):
    from backend.ingestion import delete_document, register_document

    doc_id = f"del_{uuid.uuid4().hex[:8]}"
    stored = tmp_path / "doc.txt"
    stored.write_text("x")
    register_document(doc_id, "doc.txt", stored)
    _add_doc(doc_id, "wombatrix is a word only this deleted document contains")
    rebuild_bm25_from_vector_store()
    assert get_bm25_index().search("wombatrix", 5)

    delete_document(doc_id)

    assert f"{doc_id}:0" not in get_bm25_index().get_chunk_ids()
    assert get_bm25_index().search("wombatrix", 5) == []
    assert get_bm25_index().get_chunk_ids() == _chroma_ids()


def test_save_load_uses_custom_index_dir(tmp_path: Path):
    idx = BM25Index(index_dir=tmp_path)
    idx.build([Chunk.create("d", 0, "alpha beta gamma", "h")])
    idx.save()
    assert (tmp_path / "bm25_index.pkl").exists()
    reloaded = BM25Index(index_dir=tmp_path)
    assert reloaded.load()
    assert reloaded.get_chunk_ids() == {"d:0"}


def test_search_excludes_non_matching_chunks(tmp_path: Path):
    idx = BM25Index(index_dir=tmp_path)
    idx.build([
        Chunk.create("a", 0, "alpha retrieval evidence", "ha"),
        Chunk.create("b", 0, "beta storage notes", "hb"),
        Chunk.create("c", 0, "gamma network config", "hc"),
    ])
    hits = idx.search("alpha", 10)
    assert [h["chunk_id"] for h in hits] == ["a:0"]
    assert idx.search("nonexistentterm", 10) == []


def test_add_chunks_replaces_same_id_without_duplicates(tmp_path: Path):
    idx = BM25Index(index_dir=tmp_path)
    idx.build([Chunk.create("a", 0, "old text", "h1")])
    idx.add_chunks([Chunk.create("a", 0, "new text", "h2")])
    assert idx._chunk_ids.count("a:0") == 1
    assert idx.search("new", 5)[0]["content"] == "new text"
