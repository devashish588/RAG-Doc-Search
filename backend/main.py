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
from backend.vector_store import get_embedding_backend
from backend.webapp import build_index_html


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_runtime_dirs()
    yield


app = FastAPI(
    title="RAG Document Search",
    description="Semantic document search with LangChain, HuggingFace embeddings, and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(build_index_html())


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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

    name = Path(file.filename).name
    ext  = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    ensure_runtime_dirs()
    doc_id = uuid4().hex
    dest   = UPLOAD_DIR / f"{doc_id}_{name}"
    await _save_upload(file, dest)

    register_document(document_id=doc_id, filename=name, stored_path=dest)
    background_tasks.add_task(ingest_document, doc_id, dest, name)

    return UploadResponse(
        document_id=doc_id,
        filename=name,
        status="queued",
        message="Upload accepted. Ingestion running in the background.",
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


# ---------------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile, dest: Path) -> None:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written   = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
                )
            fh.write(chunk)
