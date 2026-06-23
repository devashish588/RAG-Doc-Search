"""Standalone example for metadata filtering with Chroma.

This file is intentionally small and can be adapted if you later switch from FAISS
to a Chroma-backed vector store.
"""


def example_filter_query() -> str:
    return """
    # Example idea:
    # collection.query(query_texts=["your question"], where={"source": "your_file.pdf"})
    """
