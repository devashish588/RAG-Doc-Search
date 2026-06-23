from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from backend.ingestion import (
    get_document_status,
    ingest_document,
    list_document_statuses,
    register_document,
)
from backend.retrieval import run_search
from backend.schemas import DocumentStatus, HealthResponse, SearchRequest, SearchResponse, UploadResponse
from backend.settings import CHROMA_DIR, MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS, UPLOAD_DIR, ensure_runtime_dirs
from backend.webapp import build_index_html
from backend.vector_store import get_embedding_backend


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_runtime_dirs()
    yield


app = FastAPI(
    title="RAG Document Search API",
    description="FastAPI backend for semantic document search with LangChain, HuggingFace embeddings, and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(build_index_html())


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _save_upload(upload_file: UploadFile, target_path: Path) -> None:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0

    with target_path.open("wb") as buffer:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File is larger than {MAX_UPLOAD_MB} MB.",
                )
            buffer.write(chunk)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store=str(CHROMA_DIR),
        embedding_backend=get_embedding_backend(),
    )


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")

    original_filename = Path(file.filename).name
    extension = Path(original_filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    ensure_runtime_dirs()
    document_id = uuid4().hex
    stored_path = UPLOAD_DIR / f"{document_id}_{original_filename}"
    await _save_upload(file, stored_path)

    register_document(document_id=document_id, filename=original_filename, stored_path=stored_path)
    background_tasks.add_task(ingest_document, document_id, stored_path, original_filename)

    return UploadResponse(
        document_id=document_id,
        filename=original_filename,
        status="queued",
        message="Upload accepted. Ingestion is running in the background.",
    )


@app.get("/documents", response_model=list[DocumentStatus])
def documents() -> list[DocumentStatus]:
    return list_document_statuses()


@app.get("/documents/{document_id}", response_model=DocumentStatus)
def document_status(document_id: str) -> DocumentStatus:
    record = get_document_status(document_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentStatus(**record)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return run_search(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Search failed: {exc}") from exc
