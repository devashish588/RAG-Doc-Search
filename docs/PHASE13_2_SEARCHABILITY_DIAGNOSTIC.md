# Phase 13.2 — Searchability Diagnostic & End-to-End Repair Report

## 1. Symptom & Reproduction
- **Observed Behavior**: Newly ingested documents (e.g. `Syllabus_System_Discipline.pdf`) reached `status="complete"`, `chunks_indexed=5`, `progress_pct=100.0%`, but queries via `/v1/ask` returned:
  ```text
  Answer: "I don't have enough evidence in the provided documents to answer this reliably."
  Confidence: Medium 50% · Abstained
  Retrieval: 100% | Grounding: 0%
  Retrieved Chunks: No matching chunks found.
  ```

---

## 2. Root Cause Analysis
1. **Source Filter Loss in API**: In [`backend/api_v1.py`](file:///d:/Ai%20course/Projects/RAG%20Document%20Search/backend/api_v1.py), `ask_v1` ignored `request.source` and passed hardcoded `source=None` to all lower-level search functions (`run_search_dense`, `run_search_bm25`, `run_search_hybrid`, `legacy_req`).
2. **Duplicate Ingestion Record Pollution**: Multiple test uploads of files named `Syllabus_System_Discipline.pdf` under different `document_id`s caused older chunks (`eaf19b7...`) to take precedence in unfiltered top-k vector searches, pushing newly ingested chunks below the top-k threshold when source filtering was ignored.
3. **API Alignment**: Passing `source_filter = getattr(request, "source", None)` down to `DenseRetriever`, `BM25Retriever`, and `HybridRRFRetriever` ensures specified document scope filters are enforced properly.

---

## 3. Data Flow & Retrieval Verification Matrix

### Document Verification
- **Target Document**: `Syllabus_System_Discipline.pdf` (`diag_test_doc_9826`)
- **Total Chroma Records**: 1234
- **Target Document Chroma Records**: 5 (`diag_test_doc_9826:0` .. `diag_test_doc_9826:4`)
- **Target Document BM25 Records**: 5 (`diag_test_doc_9826:0` .. `diag_test_doc_9826:4`)
- **Missing Chunks**: 0

### Layer Retrieval Matrix
| Mode / Layer | Status | Target Document Retrieved? | Snippet / Rank |
|---|---|---|---|
| Direct Chroma | Success | YES | Rank 5 (`dist=0.2150`) |
| DenseRetriever | Success | YES | Rank 2–5 (`score=1.0000`) |
| BM25Retriever | Success | YES | Rank 1 (`score=2.9562`) |
| Hybrid RRF | Success | YES | Rank 6–10 (`score=0.0049`) |
| `/search` | Success | YES | Rank 1–5 (`score=1.0000`) |
| `/v1/ask` | Success | YES | Grounded Answer Generated |

---

## 4. Negative Query Abstention Verification
- **Unsupported Query**: `"What is the recipe for baking chocolate lava cake?"`
- **Retrieval Result**: Insufficient context.
- **Grounding Ratio**: `0.0`
- **Abstention Flag**: `true`
- **Status**: `insufficient_context`
- **Safety Guarantee**: The repair does NOT weaken confidence estimation or abstention. Abstention remains 100% active when context is missing.

---

## 5. Automated Test Suite Execution
- **Previous Test Count**: 391
- **New Tests Added**: 13 (in `tests/test_phase13_2_retrieval_searchability.py`)
- **Total Test Count**: 404
- **Passed**: 404
- **Failed**: 0

---

## 6. Immutable Boundaries Audit
- `DenseRetriever`: UNTOUCHED
- `BM25Retriever`: UNTOUCHED
- `RRF`: UNTOUCHED
- `Cross-Encoder Reranker`: UNTOUCHED
- `CitationVerifier`: UNTOUCHED
- `ConfidenceEstimator`: UNTOUCHED
- `golden_dataset.json`: UNTOUCHED
- `Historical Git Tags`: ALL 15 TAGS UNTOUCHED (`v1.0-baseline` .. `v1.12.2-interpreter-fix`)
