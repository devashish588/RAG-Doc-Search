"""Structured API error handling."""
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: ErrorDetail


class APIError(Exception):
    """Base exception for structured API errors."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=self.code,
                    message=self.message,
                    details=self.details,
                )
            ).model_dump(),
        )


class DocumentNotFoundError(APIError):
    def __init__(self, document_id: str):
        super().__init__(
            code="DOCUMENT_NOT_FOUND",
            message="Document was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"document_id": document_id},
        )


class InvalidFileTypeError(APIError):
    def __init__(self, filename: str, allowed: list[str]):
        super().__init__(
            code="INVALID_FILE_TYPE",
            message=f"Unsupported file type: {filename}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"filename": filename, "allowed_extensions": allowed},
        )


class DuplicateDocumentError(APIError):
    def __init__(self, existing_filename: str):
        super().__init__(
            code="DUPLICATE_DOCUMENT",
            message="Document with identical content already exists.",
            status_code=status.HTTP_409_CONFLICT,
            details={"existing_filename": existing_filename},
        )


class QueueFullError(APIError):
    def __init__(self):
        super().__init__(
            code="QUEUE_FULL",
            message="Ingestion queue is full. Please try again shortly.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class MalformedRequestError(APIError):
    def __init__(self, detail: str):
        super().__init__(
            code="MALFORMED_REQUEST",
            message=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class InvalidIdError(APIError):
    def __init__(self, id_value: str, id_type: str = "document"):
        super().__init__(
            code="INVALID_ID",
            message=f"Invalid {id_type} ID format.",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={f"{id_type}_id": id_value},
        )


class InternalError(APIError):
    def __init__(self, detail: str = "An internal error occurred."):
        super().__init__(
            code="INTERNAL_ERROR",
            message=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ---------------------------------------------------------------------------
# Exception handlers for FastAPI
# ---------------------------------------------------------------------------

async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return exc.to_response()


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    # Convert standard HTTPExceptions to structured format
    code = "HTTP_ERROR"
    if exc.status_code == 404:
        code = "NOT_FOUND"
    elif exc.status_code == 400:
        code = "BAD_REQUEST"
    elif exc.status_code == 409:
        code = "CONFLICT"
    elif exc.status_code >= 500:
        code = "INTERNAL_ERROR"

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=str(exc.detail),
                details={},
            )
        ).model_dump(),
    )


def register_error_handlers(app) -> None:
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)