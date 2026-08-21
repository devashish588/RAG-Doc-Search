"""Phase 2 tests: loaders, normalizer, chunking, deduplication."""
import tempfile
from pathlib import Path
import pytest

from backend.loaders import (
    PDFLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    get_loader,
    supported_extensions,
)
from backend.normalizer import TextNormalizer, MinimalNormalizer, get_normalizer
from backend.chunking import (
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
    available_strategies,
)
from backend.deduplication import Deduplicator, compute_document_hash, compute_chunk_hash
from backend.models import Chunk
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Loader Tests
# ---------------------------------------------------------------------------

def test_pdf_loader():
    """Test PDF loader with a simple PDF."""
    # Use a valid minimal PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Minimal valid PDF (single page, empty)
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Test) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n295\n%%EOF")
        path = Path(f.name)

    loader = PDFLoader()
    assert ".pdf" in loader.supported_extensions()
    docs = loader.load(path)
    assert isinstance(docs, list)
    # Should have at least one document
    path.unlink()


def test_text_loader():
    """Test text file loader."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world\nThis is a test document.\n")
        path = Path(f.name)

    loader = TextLoader()
    assert ".txt" in loader.supported_extensions()
    docs = loader.load(path)
    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata.get("page") == 1
    path.unlink()


def test_markdown_loader():
    """Test Markdown loader with section hierarchy."""
    md_content = """# Introduction

This is the introduction.

## Section 1

Content of section 1.

### Subsection 1.1

Content of subsection.
"""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(md_content)
        path = Path(f.name)

    loader = MarkdownLoader()
    assert ".md" in loader.supported_extensions()
    docs = loader.load(path)
    assert len(docs) >= 1
    # Check that sections are extracted
    assert "sections" in docs[0].metadata
    sections = docs[0].metadata["sections"]
    assert "# Introduction" in sections
    assert "## Section 1" in sections
    assert "### Subsection 1.1" in sections
    path.unlink()


def test_html_loader():
    """Test HTML loader."""
    html_content = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<h1>Main Heading</h1>
<p>Paragraph text.</p>
<script>console.log('ignored');</script>
</body>
</html>"""
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write(html_content)
        path = Path(f.name)

    loader = HTMLLoader()
    assert ".html" in loader.supported_extensions()
    docs = loader.load(path)
    assert len(docs) >= 1
    # Script content should be filtered out
    content = docs[0].page_content
    assert "Main Heading" in content or "Paragraph" in content
    assert "console.log" not in content or "ignored" not in content
    path.unlink()


def test_loader_registry():
    """Test loader registry returns correct loaders."""
    assert get_loader(Path("test.pdf")).__class__ == PDFLoader
    assert get_loader(Path("test.txt")).__class__ == TextLoader
    assert get_loader(Path("test.md")).__class__ == MarkdownLoader
    assert get_loader(Path("test.html")).__class__ == HTMLLoader

    exts = supported_extensions()
    assert ".pdf" in exts
    assert ".txt" in exts
    assert ".md" in exts
    assert ".html" in exts


def test_unsupported_extension():
    """Test unsupported extension raises error."""
    with pytest.raises(ValueError):
        get_loader(Path("test.xyz"))


# ---------------------------------------------------------------------------
# Normalizer Tests
# ---------------------------------------------------------------------------

def test_text_normalizer_whitespace():
    """Test whitespace normalization."""
    normalizer = TextNormalizer()
    text = "  Hello   world  \n\n\n  Test  "
    normalized = normalizer.normalize(text)
    assert "Hello world" in normalized
    assert "Test" in normalized
    # Multiple newlines collapsed
    assert "\n\n\n" not in normalized


def test_text_normalizer_preserves_paragraphs():
    """Test paragraph preservation."""
    normalizer = TextNormalizer()
    text = "Para 1\n\nPara 2\n\n\nPara 3"
    normalized = normalizer.normalize(text)
    # Should have at most double newlines
    assert "\n\n\n" not in normalized
    assert "Para 1" in normalized
    assert "Para 2" in normalized
    assert "Para 3" in normalized


def test_text_normalizer_code_blocks():
    """Test code block preservation."""
    normalizer = TextNormalizer()
    text = "Before\n```python\ndef foo():\n    return 42\n```\nAfter"
    normalized = normalizer.normalize(text)
    assert "def foo():" in normalized
    assert "return 42" in normalized


def test_minimal_normalizer():
    """Test minimal normalizer only does basic cleanup."""
    normalizer = MinimalNormalizer()
    text = "  Hello   world  \r\n\r\nTest  "
    normalized = normalizer.normalize(text)
    assert "Hello   world" in normalized  # spaces preserved
    assert "Test" in normalized
    assert "\r" not in normalized  # line endings normalized


def test_normalizer_factory():
    """Test normalizer factory function."""
    std = get_normalizer("standard")
    assert isinstance(std, TextNormalizer)

    minimal = get_normalizer("minimal")
    assert isinstance(minimal, MinimalNormalizer)


# ---------------------------------------------------------------------------
# Chunking Tests
# ---------------------------------------------------------------------------

def make_test_docs() -> list[Document]:
    """Create test documents."""
    return [
        Document(page_content="This is the first document. " * 50, metadata={"page": 1}),
        Document(page_content="This is the second document. " * 30, metadata={"page": 2}),
    ]


def test_fixed_chunker_deterministic():
    """Test fixed chunker produces deterministic output."""
    chunker = FixedChunker(chunk_size=500, chunk_overlap=50)
    docs = make_test_docs()

    chunks1 = chunker.chunk(docs, "doc123", "test.txt")
    chunks2 = chunker.chunk(docs, "doc123", "test.txt")

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.id == c2.id
        assert c1.content == c2.content
        assert c1.chunk_strategy == "fixed"


def test_fixed_chunker_size_config():
    """Test fixed chunker respects chunk size."""
    chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
    docs = [Document(page_content="A" * 500, metadata={})]

    chunks = chunker.chunk(docs, "doc1", "test.txt")

    # Each chunk should be roughly chunk_size (allowing for overlap)
    for chunk in chunks:
        assert len(chunk.content) <= 100 + 10  # size + overlap buffer
        assert chunk.chunk_strategy == "fixed"


def test_recursive_chunker_deterministic():
    """Test recursive chunker produces deterministic output."""
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    docs = make_test_docs()

    chunks1 = chunker.chunk(docs, "doc123", "test.txt")
    chunks2 = chunker.chunk(docs, "doc123", "test.txt")

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.id == c2.id
        assert c1.content == c2.content
        assert c1.chunk_strategy == "recursive"


def test_semantic_chunker_available():
    """Test semantic chunker can be instantiated."""
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
    assert chunker.strategy_name == "semantic"


def test_chunker_factory():
    """Test chunker factory returns correct instances."""
    fixed = get_chunker("fixed", 500, 50)
    assert isinstance(fixed, FixedChunker)

    recursive = get_chunker("recursive", 500, 50)
    assert isinstance(recursive, RecursiveChunker)

    semantic = get_chunker("semantic", 500, 50)
    assert isinstance(semantic, SemanticChunker)

    strategies = available_strategies()
    assert "fixed" in strategies
    assert "recursive" in strategies
    assert "semantic" in strategies


def test_unknown_strategy():
    """Test unknown strategy raises error."""
    with pytest.raises(ValueError):
        get_chunker("unknown", 500, 50)


def test_chunk_metadata_fields():
    """Test chunks have all required metadata fields."""
    chunker = FixedChunker(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content="Test content. " * 50, metadata={"page": 1, "section": "Intro"})]

    chunks = chunker.chunk(docs, "doc123", "test.txt")

    for chunk in chunks:
        assert chunk.id.startswith("doc123:")
        assert chunk.document_id == "doc123"
        assert chunk.chunk_index >= 0
        assert chunk.content
        assert chunk.character_count == len(chunk.content)
        assert chunk.token_count is not None
        assert chunk.content_hash
        assert chunk.chunk_strategy == "fixed"
        assert chunk.metadata.get("source") == "test.txt"


def test_chunk_canonical_id_format():
    """Test canonical chunk IDs follow {doc_id}:{index} format."""
    chunker = FixedChunker(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content="A" * 1000, metadata={})]

    chunks = chunker.chunk(docs, "abc123", "test.txt")

    assert chunks[0].id == "abc123:0"
    assert chunks[1].id == "abc123:1"
    # IDs should be sequential
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"abc123:{i}"


# ---------------------------------------------------------------------------
# Deduplication Tests
# ---------------------------------------------------------------------------

def test_exact_document_hash():
    """Test document hash computation."""
    content = b"test content"
    hash1 = compute_document_hash(content)
    hash2 = compute_document_hash(content)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex

    # Different content = different hash
    hash3 = compute_document_hash(b"different")
    assert hash1 != hash3


def test_chunk_hash():
    """Test chunk hash computation."""
    content = "test chunk content"
    hash1 = compute_chunk_hash(content)
    hash2 = compute_chunk_hash(content)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_exact_chunk_deduplication():
    """Test exact chunk deduplication."""
    dedup = Deduplicator()

    chunk1 = Chunk.create(
        document_id="doc1",
        chunk_index=0,
        content="Identical content",
        content_hash=compute_chunk_hash("Identical content"),
    )
    chunk2 = Chunk.create(
        document_id="doc1",
        chunk_index=1,
        content="Identical content",
        content_hash=compute_chunk_hash("Identical content"),
    )

    result1 = dedup.check_exact_chunk(chunk1)
    assert not result1.is_duplicate

    # Register the first chunk
    dedup.register_chunk(chunk1)

    result2 = dedup.check_exact_chunk(chunk2)
    assert result2.is_duplicate
    assert result2.duplicate_type == "exact_chunk"


def test_near_duplicate_detection():
    """Test near-duplicate detection with mock embedder."""
    dedup = Deduplicator(near_duplicate_threshold=0.9)

    import numpy as np

    def mock_embedder(text: str) -> np.ndarray:
        # Simple deterministic embedding based on text length
        arr = np.ones(384, dtype=np.float32) * (len(text) / 100.0)
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    chunk1 = Chunk.create(
        document_id="doc1",
        chunk_index=0,
        content="This is a test document about machine learning.",
        content_hash=compute_chunk_hash("This is a test document about machine learning."),
    )
    chunk2 = Chunk.create(
        document_id="doc1",
        chunk_index=1,
        content="This is a test document about machine learning.",  # Same
        content_hash=compute_chunk_hash("This is a test document about machine learning."),
    )

    result1 = dedup.check_near_duplicate(chunk1, mock_embedder)
    assert not result1.is_duplicate

    # Register first chunk
    dedup.register_chunk(chunk1, mock_embedder(chunk1.content))

    result2 = dedup.check_near_duplicate(chunk2, mock_embedder)
    assert result2.is_duplicate
    assert result2.duplicate_type == "near_chunk"
    assert result2.similarity >= 0.9


def test_deduplicate_chunks_function():
    """Test the deduplicate_chunks convenience function."""
    from backend.deduplication import deduplicate_chunks
    import numpy as np

    def mock_embedder(text: str) -> np.ndarray:
        arr = np.ones(384, dtype=np.float32) * (hash(text) % 100 / 100.0)
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    chunks = [
        Chunk.create("doc1", 0, "Unique content A", compute_chunk_hash("Unique content A")),
        Chunk.create("doc1", 1, "Unique content A", compute_chunk_hash("Unique content A")),  # Exact dup
        Chunk.create("doc1", 2, "Unique content B", compute_chunk_hash("Unique content B")),
    ]

    kept, results = deduplicate_chunks(chunks, embedder_fn=mock_embedder)

    assert len(kept) == 2  # One exact duplicate removed
    assert results[0].is_duplicate == False
    assert results[1].is_duplicate == True  # Exact duplicate
    assert results[2].is_duplicate == False


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

def test_full_ingestion_pipeline_deterministic():
    """Test full ingestion pipeline is deterministic."""
    from backend.ingestion_v2 import ingest_document_v2, get_ingestion_config_snapshot

    # Create test file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("This is a test document for ingestion testing. " * 20)
        path = Path(f.name)

    try:
        doc_id = "test_doc_123"
        result1 = ingest_document_v2(doc_id, path, "test.txt")
        result2 = ingest_document_v2(doc_id, path, "test.txt")

        # Should be deterministic (same chunks indexed)
        assert result1["chunks_indexed"] == result2["chunks_indexed"]
    finally:
        path.unlink()


def test_config_snapshot_captured():
    """Test ingestion config snapshot is captured."""
    from backend.ingestion_v2 import get_ingestion_config_snapshot

    snapshot = get_ingestion_config_snapshot()

    assert "chunking_strategy" in snapshot
    assert "chunk_size" in snapshot
    assert "chunk_overlap" in snapshot
    assert "near_duplicate_threshold" in snapshot
    assert "embedding_model" in snapshot
    assert "embedding_backend" in snapshot

    # Values should match settings
    assert snapshot["chunk_size"] == 1200
    assert snapshot["chunk_overlap"] == 200
    assert snapshot["chunking_strategy"] in ("fixed", "recursive", "semantic")