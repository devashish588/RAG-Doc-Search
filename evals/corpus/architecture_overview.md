# RAG Document Search - System Architecture & Component Specification

## 1. System Overview
The RAG Document Search system is a decoupled web application designed for semantic document retrieval and AI-powered question answering. The backend is built using FastAPI, LangChain, fastembed (ONNX runtime), and ChromaDB.

## 2. Ingestion Engine & Queue Lifecycle
- **Document Loaders**: PyPDFLoader for `.pdf` files, TextLoader (UTF-8 encoding) for `.txt` and `.md` files.
- **Chunking Strategy**: `RecursiveCharacterTextSplitter` configured with `CHUNK_SIZE=1200` characters and `CHUNK_OVERLAP=200` characters. Separators used are `["\n\n", "\n", ". ", " ", ""]`.
- **Deduplication**: Files uploaded are hashed using SHA-256 (`content_hash`). If an active upload with the exact same content hash exists in `queued` or `processing` status, the endpoint returns a `409 CONFLICT` response code.
- **Worker Queue**: Ingestion jobs are pushed to an in-process bounded worker queue (`ingestion_queue.py`). Document status transitions through `queued` -> `processing` -> `complete` (or `failed`).

## 3. Retrieval Pipeline & MMR Parameters
- **Embedding Backend**: Primary backend is `FastEmbedEmbeddings` utilizing `BAAI/bge-small-en-v1.5` (ONNX format). When offline or when fastembed is unavailable, it gracefully falls back to `HashingEmbeddings` (scikit-learn `HashingVectorizer` with 1024 features).
- **Retrieval Strategy**: Dense vector similarity with Maximal Marginal Relevance (`max_marginal_relevance_search`).
- **MMR Parameters**: Initial candidate fetch `fetch_k=20`, target result count `k=8`, diversity balancing coefficient `lambda_mult=0.5`.
- **Score Filtering**: Low-relevance chunks below `MIN_RELEVANCE_SCORE=0.35` are filtered out.
- **Adjacent Chunk Expansion**: For every retrieved chunk with index `N`, the retrieval engine automatically attempts to fetch adjacent chunk `N+1` from ChromaDB to prevent boundary cutoff for multi-page lists.

## 4. LLM Integration & OpenRouter Protocol
- **Generation Model**: OpenRouter API (`OPENROUTER_MODEL=openai/gpt-4o-mini`, `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`).
- **Fallback Behavior**: When `OPENROUTER_API_KEY` is empty or API calls fail, the engine falls back to plain context synthesis listing top matching sources.
- **Generation Parameters**: `temperature=0.0`, `max_tokens=768`.
