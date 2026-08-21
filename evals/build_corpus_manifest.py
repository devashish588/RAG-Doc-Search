import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MANIFEST_PATH = Path(__file__).resolve().parent / "corpus_manifest.json"


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifest() -> dict[str, Any]:
    files_info = []
    total_chars = 0
    total_documents = 0

    for p in sorted(CORPUS_DIR.glob("*")):
        if p.is_dir() or p.name.startswith("."):
            continue

        sha256 = hash_file(p)
        ext = p.suffix.lower()

        if ext == ".pdf":
            docs = PyPDFLoader(str(p)).load()
            page_count = len(docs)
            content = "\n".join(d.page_content for d in docs)
        elif ext in {".txt", ".md"}:
            docs = TextLoader(str(p), encoding="utf-8").load()
            page_count = 1
            content = "\n".join(d.page_content for d in docs)
        else:
            continue

        char_count = len(content)
        total_chars += char_count
        total_documents += 1

        files_info.append({
            "filename": p.name,
            "path": str(p.relative_to(CORPUS_DIR.parent)),
            "extension": ext,
            "size_bytes": p.stat().st_size,
            "sha256": sha256,
            "page_count": page_count,
            "char_count": char_count,
        })

    manifest = {
        "version": "1.0",
        "corpus_name": "fixed_baseline_corpus",
        "total_documents": total_documents,
        "total_characters": total_chars,
        "documents": files_info,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {MANIFEST_PATH} with {total_documents} documents.")
    return manifest


if __name__ == "__main__":
    generate_manifest()
