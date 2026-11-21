"""Upload filenames are client-controlled and must be sanitized."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.ingestion import safe_upload_name
from backend.main import app
from backend.middleware import reset_limiter


@pytest.mark.parametrize("raw, expected", [
    ("report.pdf", "report.pdf"),
    ("../../etc/passwd.txt", "passwd.txt"),
    ("C:\\Users\\me\\notes.md", "notes.md"),
    ('a<b>c:"d"|e?f*.txt', "a_b_c__d__e_f_.txt"),
    ("Report.PDF", "Report.pdf"),
    ("   .txt", "document.txt"),
    ("x" * 300 + ".txt", "x" * 150 + ".txt"),
])
def test_safe_upload_name(raw, expected):
    assert safe_upload_name(raw) == expected


def test_upload_with_unsafe_name_is_accepted_and_sanitized():
    reset_limiter()
    client = TestClient(app)
    # No lifespan here, so no ingestion worker: stub the queue.
    with patch("backend.main.submit_job", return_value=True):
        resp = client.post(
            "/upload",
            files={"file": ('evil <img src=x onerror=alert(1)>.txt', b"harmless content here", "text/plain")},
        )
    assert resp.status_code == 202, resp.text
    name = resp.json()["filename"]
    assert "<" not in name and ">" not in name
    assert name.endswith(".txt")
    client.delete(f"/documents/{resp.json()['document_id']}")
