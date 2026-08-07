"""In-process ingestion queue.

Decouples the API from the (slow, memory-hungry) embedding step without
spawning a second service. A single daemon worker consumes jobs off a bounded
queue, so uploads return immediately and ingestion happens on one thread at a
time — keeping peak RAM low on the single-instance free plan.
"""
import os
import queue
import threading

from backend.ingestion import ingest_document

_QUEUE_SIZE = int(os.getenv("INGESTION_QUEUE_SIZE", "8"))

_q = queue.Queue(maxsize=_QUEUE_SIZE)
_thread: threading.Thread | None = None


def _run() -> None:
    while True:
        document_id, stored_path, filename = _q.get()
        try:
            ingest_document(document_id, stored_path, filename)
        finally:
            _q.task_done()


def start() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_run, name="ingestion-worker", daemon=True)
        _thread.start()


def submit(document_id: str, stored_path, filename: str) -> bool:
    """Queue a job. Returns False if the queue is full (backpressure)."""
    try:
        _q.put_nowait((document_id, stored_path, filename))
        return True
    except (queue.Full,):
        return False