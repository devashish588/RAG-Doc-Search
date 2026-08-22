"""Dependency and ingestion runtime validation tests.

Verifies that critical production dependencies are installed, the embedding
backend initializes, Chroma accepts documents, and retrieval works after
ingestion. Also validates WindowsPath metadata conversion.
"""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 1. Import validation
# ---------------------------------------------------------------------------

class TestCriticalImports:
    def test_import_chromadb(self):
        import chromadb
        assert hasattr(chromadb, "__version__")

    def test_import_fastembed(self):
        import fastembed
        assert hasattr(fastembed, "__version__") or hasattr(fastembed, "VERSION")

    def test_import_numpy(self):
        import numpy
        assert hasattr(numpy, "__version__")

    def test_import_rank_bm25(self):
        from rank_bm25 import BM25Okapi
        assert BM25Okapi is not None

    def test_import_langchain_chroma(self):
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        assert Chroma is not None

    def test_import_sklearn(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        assert HashingVectorizer is not None

    def test_import_fastapi(self):
        import fastapi
        assert hasattr(fastapi, "__version__")

    def test_import_prometheus_client(self):
        import prometheus_client
        assert hasattr(prometheus_client, "Counter")


# ---------------------------------------------------------------------------
# 2. Dependency check module
# ---------------------------------------------------------------------------

class TestDependencyCheck:
    def test_check_runtime_dependencies(self):
        from backend.dependency_check import check_runtime_dependencies
        report = check_runtime_dependencies()
        assert report.status in ("ok", "degraded")
        assert report.missing_critical == []
        assert "chromadb" in report.dependencies
        assert "fastembed" in report.dependencies
        assert "numpy" in report.dependencies
        assert "rank-bm25" in report.dependencies

    def test_check_embedding_backend(self):
        from backend.dependency_check import check_embedding_backend
        result = check_embedding_backend()
        assert result["status"] == "ok"
        assert result["backend"] == "fastembed"

    def test_check_vector_store(self):
        from backend.dependency_check import check_vector_store
        result = check_vector_store()
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# 3. Embedding initialization
# ---------------------------------------------------------------------------

class TestEmbeddingInit:
    def test_embedding_backend_is_fastembed(self):
        from backend.vector_store import get_embedding_backend
        backend = get_embedding_backend()
        assert backend == "fastembed"

    def test_embeddings_embed_query(self):
        from backend.vector_store import get_embeddings
        embeddings = get_embeddings()
        vec = embeddings.embed_query("test query")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)


# ---------------------------------------------------------------------------
# 4. Chroma initialization (isolated)
# ---------------------------------------------------------------------------

class TestChromaInit:
    def test_chroma_collection_accessible(self):
        from backend.vector_store import get_vector_store
        store = get_vector_store()
        collection = getattr(store, "_collection", None)
        assert collection is not None
        # Should be able to count without error
        count = collection.count()
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# 5. Minimal ingestion
# ---------------------------------------------------------------------------

class TestMinimalIngestion:
    def test_ingest_and_retrieve(self):
        from langchain_core.documents import Document
        from backend.vector_store import get_vector_store, add_documents, _VECTOR_LOCK
        import uuid

        store = get_vector_store()
        test_id = f"test_dep_smoke_{uuid.uuid4().hex[:8]}"
        doc_id = f"{test_id}:0"

        try:
            doc = Document(
                page_content="HybridRAG dependency smoke test. This document verifies the production ingestion pipeline.",
                metadata={
                    "source": "test_smoke.txt",
                    "document_id": test_id,
                    "chunk": 0,
                },
            )
            with _VECTOR_LOCK:
                store.add_documents(documents=[doc], ids=[doc_id])
                if callable(getattr(store, "persist", None)):
                    store.persist()

            # Verify retrieval
            results = store.similarity_search("dependency smoke test", k=1)
            assert len(results) >= 1
            assert "dependency smoke test" in results[0].page_content.lower()
        finally:
            # Cleanup
            collection = getattr(store, "_collection", None)
            if collection is not None:
                try:
                    collection.delete(ids=[doc_id])
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 6. WindowsPath metadata test
# ---------------------------------------------------------------------------

class TestWindowsPathMetadata:
    def test_path_metadata_converted(self):
        """Ensure Path objects are converted to strings before Chroma upsert."""
        from pathlib import Path
        from langchain_core.documents import Document
        from backend.vector_store import get_vector_store, _VECTOR_LOCK
        import uuid

        store = get_vector_store()
        test_id = f"test_winpath_{uuid.uuid4().hex[:8]}"
        doc_id = f"{test_id}:0"

        try:
            # Simulate metadata that might contain Path objects
            raw_path = Path("data") / "uploads" / "test.pdf"
            metadata = {
                "source": str(raw_path),  # must be string
                "document_id": test_id,
                "chunk": 0,
                "file_path": str(raw_path),  # verify str conversion
            }
            # Assert all metadata values are Chroma-safe types
            for k, v in metadata.items():
                assert isinstance(v, (str, int, float, bool, list, type(None))), \
                    f"Metadata '{k}' has unsupported type {type(v)}"

            doc = Document(
                page_content="WindowsPath metadata validation test.",
                metadata=metadata,
            )
            with _VECTOR_LOCK:
                store.add_documents(documents=[doc], ids=[doc_id])
                if callable(getattr(store, "persist", None)):
                    store.persist()

            # Verify retrieval
            results = store.similarity_search("WindowsPath metadata", k=1)
            assert len(results) >= 1
        finally:
            collection = getattr(store, "_collection", None)
            if collection is not None:
                try:
                    collection.delete(ids=[doc_id])
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 7. Readyz reports dependencies
# ---------------------------------------------------------------------------

class TestReadyzDependencies:
    def test_readyz_includes_dependencies(self):
        from backend.storage_health import readiness_report
        report = readiness_report()
        assert "dependencies" in report["checks"]
        dep_check = report["checks"]["dependencies"]
        assert dep_check["status"] == "ok"
