from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def build_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create the embedding model used by FAISS."""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(documents: List[Document], embeddings, persist_dir: str = "vector_store") -> FAISS:
    """Build and persist a FAISS index from document chunks."""
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(persist_dir)
    return vector_store


def load_vector_store(embeddings, persist_dir: str = "vector_store") -> FAISS:
    """Load an existing FAISS index from disk."""
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


def retrieve_context(vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
    """Retrieve the most relevant context chunks for the question."""
    return vector_store.similarity_search(query, k=k)
