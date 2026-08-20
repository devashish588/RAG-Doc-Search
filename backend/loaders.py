"""Document loaders for various formats.

Implements the BaseLoader interface for PDF, TXT, Markdown, and HTML files.
"""
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.interfaces import BaseLoader

log = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """Load PDF documents using PyPDF."""

    def load(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        # Preserve page metadata from PyPDFLoader
        for i, doc in enumerate(docs):
            if "page" not in doc.metadata:
                doc.metadata["page"] = i + 1
        return docs

    def supported_extensions(self) -> set[str]:
        return {".pdf"}


class TextLoader(BaseLoader):
    """Load plain text files."""

    def load(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        # Add page metadata for consistency
        for doc in docs:
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
        return docs

    def supported_extensions(self) -> set[str]:
        return {".txt"}


class MarkdownLoader(BaseLoader):
    """Load Markdown files with section hierarchy preservation."""

    def load(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()

        # Parse Markdown headers to extract section hierarchy
        for doc in docs:
            content = doc.page_content
            sections = self._extract_sections(content)
            doc.metadata["sections"] = sections
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1

        return docs

    def _extract_sections(self, content: str) -> list[str]:
        """Extract section headers from Markdown content."""
        # Find all headers (# ## ### etc.)
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        headers = header_pattern.findall(content)
        return [f"{'#' * len(level)} {title}" for level, title in headers]

    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown"}


class HTMLLoader(BaseLoader):
    """Load HTML files, extracting meaningful text content."""

    def load(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        try:
            loader = UnstructuredHTMLLoader(str(path))
            docs = loader.load()
        except Exception as exc:
            log.warning("UnstructuredHTMLLoader failed for %s, falling back to basic extraction: %s", path, exc)
            docs = self._basic_html_load(path)

        # Add page metadata for consistency
        for doc in docs:
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1

        return docs

    def _basic_html_load(self, path: Path) -> list[Document]:
        """Basic HTML text extraction as fallback."""
        import html
        from html.parser import HTMLParser

        class SimpleHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.ignore_tags = {"script", "style", "meta", "link", "noscript"}
                self.current_tag = None

            def handle_starttag(self, tag, attrs):
                self.current_tag = tag.lower()

            def handle_endtag(self, tag):
                self.current_tag = None

            def handle_data(self, data):
                if self.current_tag not in self.ignore_tags and data.strip():
                    self.text_parts.append(data.strip())

        content = path.read_text(encoding="utf-8", errors="ignore")
        parser = SimpleHTMLParser()
        parser.feed(content)
        text = "\n\n".join(parser.text_parts)

        if not text.strip():
            text = "[No extractable text content]"

        return [Document(page_content=text, metadata={"source": str(path)})]

    def supported_extensions(self) -> set[str]:
        return {".html", ".htm"}


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

_LOADER_REGISTRY: dict[str, BaseLoader] = {
    ".pdf": PDFLoader(),
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
}


def get_loader(path: Path) -> BaseLoader:
    """Get appropriate loader for file extension."""
    ext = path.suffix.lower()
    if ext not in _LOADER_REGISTRY:
        raise ValueError(f"No loader for extension '{ext}'. Supported: {sorted(ext for ext in _LOADER_REGISTRY)}")
    return _LOADER_REGISTRY[ext]


def register_loader(ext: str, loader: BaseLoader) -> None:
    """Register a custom loader (for testing/extension)."""
    _LOADER_REGISTRY[ext.lower()] = loader


def supported_extensions() -> set[str]:
    """Return all supported file extensions."""
    return set(_LOADER_REGISTRY.keys())