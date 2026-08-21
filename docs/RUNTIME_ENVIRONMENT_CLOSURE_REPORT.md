# Runtime Environment Closure Report

**Date:** 2026-08-21
**Scope:** Fix runtime dependency/environment mismatch (wrong Python interpreter)

## Root Cause

The user's machine had global Python 3.14 at `C:\Users\hp\AppData\Local\Programs\Python\Python314\`. When `python -m uvicorn` was run without activating `.venv`, it used the global Python which lacks chromadb/fastembed/etc. The project `.venv` has Python 3.12 with all dependencies.

## Changes Made

### `backend/dependency_check.py`
- Added `log_environment_info()` — logs executable, Python version, prefix, base_prefix, virtualenv status at startup
- `startup_validation()` now prints interpreter info and fails with actionable instructions when wrong interpreter detected
- Error messages distinguish between "wrong interpreter" (not in venv) vs "missing deps in venv"

### `evals/runtime_environment_diagnostic.py` (NEW)
- Standalone diagnostic script: prints interpreter info, virtualenv status, all dependency statuses
- Usage: `.venv\Scripts\python.exe evals/runtime_environment_diagnostic.py`
- Detects wrong interpreter and prints canonical start command

### `tests/test_runtime_environment.py` (NEW)
- 17 tests covering:
  - Interpreter info and virtualenv detection
  - Package import verification (chromadb, fastembed, numpy, rank_bm25, prometheus_client, flashrank)
  - Dependency checker uses imports (not requirements.txt)
  - Missing package detection (mocked chromadb, fastembed)
  - /readyz dependency health reporting
  - /readyz returns 503 when chromadb missing
  - /healthz remains lightweight (no dependency checks)
  - Production backend is fastembed (no silent hashing fallback)
  - WindowsPath metadata regression test

### `evals/test_real_ingestion.py` (NEW)
- Integration test: starts server in-process, uploads real document, verifies Chroma chunks, tests search
- Verifies .venv Python 3.12.13 is used for server startup

### `README.md`
- Updated "Run" section with canonical start command using `.venv\Scripts\python.exe`
- Added warning about wrong interpreter

### `docs/DEPENDENCY_RUNTIME_GUIDE.md`
- Updated validation section to reference diagnostic script
- Updated startup section with canonical start commands
- Added warning about wrong interpreter

## Verification Results

### Test Suite
```
378 passed, 0 failed (17 new from test_runtime_environment.py)
```

### Diagnostic Script
```
executable:    .venv\Scripts\python.exe
python:        3.12.13
virtualenv:    True
chromadb:      OK (1.5.9)
fastembed:     OK (0.8.0)
numpy:         OK (2.5.0)
rank_bm25:     OK
ALL DEPENDENCIES OK
```

### Real Ingestion Test
```
Server started with .venv Python 3.12.13
/healthz -> 200
/readyz -> 200: ready (dependencies=ok, chroma=ok, bm25=ok, disk=ok)
Ingestion: 13 chunks indexed (1 unique after dedup)
Search: 200 (dense retrieval returns 10 results)
```

### Startup Validation Log
```
Runtime environment:
  executable:     .venv\Scripts\python.exe
  python:         3.12.13
  prefix:         .venv
  base_prefix:    ...global python...
  virtualenv:     True
  site-packages:  .venv
Startup validation OK - chromadb=1.5.9, embedding=fastembed, vector_store=langchain_chroma.Chroma
```

## Key Findings

1. **Root cause confirmed**: Global Python 3.14 at `C:\Users\hp\AppData\Local\Programs\Python\Python314\` was being used instead of `.venv` Python 3.12
2. **All .venv dependencies are installed**: chromadb 1.5.9, fastembed 0.8.0, numpy 2.5.0, rank_bm25, prometheus_client, flashrank
3. **Chroma collection exists**: `rag_documents_fastembed_bge_small_en_v1_5` with chunks
4. **FastEmbed is the active embedding backend** (not hashing fallback)
5. **/readyz correctly reports dependency health** and returns 503 when critical deps are missing
6. **The fix is runtime detection**, not code changes to the ingestion pipeline

## What Was Skipped

- Docker verification (not needed for local fix)
- Clean environment test (would require recreating venv)
- Phase 13 — not started

## Canonical Start Command

```bash
# Windows:
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 9826

# macOS/Linux:
source .venv/bin/activate && uvicorn backend.main:app --host 127.0.0.1 --port 9826
```
