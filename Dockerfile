FROM python:3.12-slim

ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install production dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app
RUN mkdir -p /app/data/uploads /app/data/chroma_db /app/data/bm25_index && \
    chown -R app:app /app
USER app

EXPOSE 9826

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9826/healthz')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9826", "--workers", "1"]
