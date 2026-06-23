from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf_documents(data_dir: str) -> List[Document]:
    """Load all PDF files from a folder and return LangChain documents."""
    docs = []
    for path in sorted(Path(data_dir).glob("*.pdf")):
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())

    for doc in docs:
        doc.metadata.setdefault("source", str(doc.metadata.get("source", "unknown")))
    return docs


def clean_text(text: str) -> str:
    """Basic text cleanup for better chunk quality."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def has_extractable_text(document: Document) -> bool:
    """Return True when a document has real text content after cleaning."""
    return bool(clean_text(document.page_content).strip())


def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    """Split documents into smaller chunks for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    cleaned_chunks = []
    for chunk in chunks:
        text = clean_text(chunk.page_content)
        if text.strip():
            cleaned_chunks.append(Document(page_content=text, metadata=chunk.metadata))
    return cleaned_chunks
