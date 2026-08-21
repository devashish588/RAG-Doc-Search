"""Text normalization for document ingestion.

Implements the BaseNormalizer interface for cleaning and normalizing
raw document text before chunking.
"""
import logging
import re
from typing import Any

from backend.interfaces import BaseNormalizer

log = logging.getLogger(__name__)


class TextNormalizer(BaseNormalizer):
    """Normalize document text while preserving semantic structure."""

    def __init__(
        self,
        normalize_whitespace: bool = True,
        preserve_headings: bool = True,
        preserve_paragraphs: bool = True,
        preserve_code_blocks: bool = True,
    ):
        self.normalize_whitespace = normalize_whitespace
        self.preserve_headings = preserve_headings
        self.preserve_paragraphs = preserve_paragraphs
        self.preserve_code_blocks = preserve_code_blocks

    def normalize(self, text: str) -> str:
        """Normalize text according to configuration."""
        if not text:
            return ""

        # Preserve code blocks first (they have special formatting)
        if self.preserve_code_blocks:
            text, code_blocks = self._extract_code_blocks(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Preserve headings
        if self.preserve_headings:
            text = self._preserve_headings(text)

        # Preserve paragraph boundaries
        if self.preserve_paragraphs:
            text = self._preserve_paragraphs(text)

        # Restore code blocks
        if self.preserve_code_blocks:
            text = self._restore_code_blocks(text, code_blocks)

        return text.strip()

    def _extract_code_blocks(self, text: str) -> tuple[str, list[str]]:
        """Extract code blocks and replace with placeholders."""
        code_blocks = []

        # Match fenced code blocks (```...``` or ~~~...~~~)
        def replace_code_block(match):
            code_blocks.append(match.group(0))
            return f"{{{{CODE_BLOCK_{len(code_blocks) - 1}}}}}"

        text = re.sub(r"(```[\s\S]*?```|~~~[\s\S]*?~~~)", replace_code_block, text)

        # Match indented code blocks (4+ spaces)
        def replace_indented_code(match):
            code_blocks.append(match.group(0))
            return f"{{{{CODE_BLOCK_{len(code_blocks) - 1}}}}}"

        text = re.sub(r"(?m)^( {4}|\t).+$", replace_indented_code, text)

        return text, code_blocks

    def _restore_code_blocks(self, text: str, code_blocks: list[str]) -> str:
        """Restore code blocks from placeholders."""
        for i, block in enumerate(code_blocks):
            text = text.replace(f"{{{{CODE_BLOCK_{i}}}}}", block)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving intentional structure."""
        # Replace tabs with spaces
        text = text.replace("\t", "    ")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse multiple blank lines to at most 2 (preserves paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize spaces within lines (but not at start for indentation)
        def normalize_line_spaces(line):
            if not line.strip():
                return ""
            # Preserve leading indentation
            match = re.match(r"^(\s*)", line)
            indent = match.group(1) if match else ""
            content = line[len(indent):]
            # Collapse multiple spaces
            content = re.sub(r" {2,}", " ", content)
            return indent + content

        lines = [normalize_line_spaces(line) for line in text.split("\n")]
        text = "\n".join(lines)

        return text

    def _preserve_headings(self, text: str) -> str:
        """Ensure headings are well-formed and preserved."""
        # Markdown headings
        text = re.sub(r"(?m)^(#{1,6})\s*(.+)$", r"\1 \2", text)

        # Potential heading lines (all caps, short, followed by content)
        # This is heuristic - be conservative
        return text

    def _preserve_paragraphs(self, text: str) -> str:
        """Ensure paragraph boundaries are clear."""
        # Already handled by whitespace normalization (max 2 newlines)
        return text


class MinimalNormalizer(BaseNormalizer):
    """Minimal normalizer that only does basic cleanup."""

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Only normalize line endings and remove trailing whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()


def get_normalizer(name: str = "standard") -> BaseNormalizer:
    """Get normalizer by name."""
    if name == "minimal":
        return MinimalNormalizer()
    return TextNormalizer()