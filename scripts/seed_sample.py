from pathlib import Path
import time

import requests


API = "http://127.0.0.1:8000"
FILE_PATH = Path("data") / "DATABASE MANAGEMENT SYSTEMS.pdf"


def main() -> None:
    file_bytes = FILE_PATH.read_bytes()
    files = {"file": (FILE_PATH.name, file_bytes, "application/pdf")}

    response = requests.post(f"{API}/upload", files=files, timeout=60)
    response.raise_for_status()
    document_id = response.json()["document_id"]
    print(f"uploaded {document_id}")

    for _ in range(60):
        status = requests.get(f"{API}/documents/{document_id}", timeout=10).json()
        print(status["status"], status["chunks_indexed"])
        if status["status"] == "complete":
            break
        time.sleep(2)

    search = requests.post(
        f"{API}/search",
        json={"query": "What is a database management system?", "top_k": 3},
        timeout=60,
    )
    search.raise_for_status()
    print(search.json()["answer"])


if __name__ == "__main__":
    main()
