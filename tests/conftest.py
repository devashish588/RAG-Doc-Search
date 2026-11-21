"""Shared fixtures for all test modules."""
import os
import shutil
import tempfile
from pathlib import Path

# Isolate test data BEFORE any backend module is imported: backend.settings
# resolves DATA_DIR at import time. Without this the suite reads and mutates
# the developer's real data/ directory (Chroma DB, BM25 index, uploads), so a
# local index built by a different chromadb version breaks ~100 tests, and the
# tests pollute the real index.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="rag_test_data_"))
os.environ["DATA_DIR"] = str(_TEST_DATA_DIR)
# Reuse the shared embedding-model cache so tests don't re-download the model.
os.environ.setdefault(
    "FASTEMBED_CACHE_DIR", str(_REPO_ROOT / "data" / "fastembed_cache")
)

import pytest  # noqa: E402

from backend.middleware import reset_limiter  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the in-memory rate limiter before every test to avoid cross-module 429s."""
    reset_limiter()
    yield


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)
