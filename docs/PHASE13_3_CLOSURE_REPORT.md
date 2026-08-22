# Phase 13.3 Completion Report

## 1. Reproduced Issue

The UI displayed "No matching chunks found." in the Retrieved Chunks section despite the Retrieval Trace showing non-zero counts (Dense 5, BM25 5, Hybrid RRF 10, Reranker 5).

## 2. Root Cause

**Field mismatch between backend response and frontend rendering.**

The frontend's `renderCitations()` function read from `data.citations`. But `AskResponse.citations` was built from *citation verification* (grounding), NOT from retrieval. When grounding ratio was 0% and abstention fired, citations was empty — even though retrieval returned real chunks in `retrieval_trace.dense/bm25/rrf/reranker`.

The `AskResponse` schema had NO field for retrieved chunks. The retrieval data existed in `retrieval_trace` but was not exposed as a top-level field the frontend could render.

## 3. Evidence

Runtime diagnostic captured before fix:

```
Response top-level keys: ['answer', 'status', 'citations', 'confidence', 'grounding_metrics', 'retrieval_trace']
retrieved_chunks field present: False
citations: 0 items
retrieval_trace.dense: 10 items  ← chunks exist here
```

Frontend code (app.js:120-124):
```javascript
var cites = data.citations || [];
if (!cites.length) {
  resultsBox.innerHTML = '<div class="muted">No matching chunks found.</div>';
  return;
}
```

## 4. Fix

| File | Change |
|------|--------|
| `backend/api_v1.py` | Added `RetrievedChunk` model, `_extract_retrieved_chunks()` helper, `retrieved_chunks` field to `AskResponse`, observability logging |
| `frontend/app.js` | Replaced `renderCitations()` with `renderAnswer()` + `renderChunks()` reading from `data.retrieved_chunks` |

**Smallest safe fix**: Added one new response field and one frontend renderer. No retrieval algorithms, grounding, or confidence logic changed.

## 5. Backend Response Contract

```json
{
  "answer": "...",
  "status": "answered|insufficient_context",
  "citations": [...],
  "retrieved_chunks": [
    {
      "chunk_id": "doc1:5",
      "source": "Syllabus.pdf",
      "text": "...",
      "score": 0.87,
      "rank": 1,
      "page": 3,
      "metadata": {...}
    }
  ],
  "confidence": {
    "overall_score": 0.5,
    "level": "medium",
    "retrieval_confidence": 1.0,
    "grounding_confidence": 0.0,
    "abstention_flag": true,
    "signals": {...}
  },
  "grounding_metrics": {...},
  "retrieval_trace": {
    "dense": [...],
    "bm25": [...],
    "rrf": [...],
    "reranker": [...]
  }
}
```

## 6. Frontend Rendering Contract

`renderChunks(data)` reads `data.retrieved_chunks` and renders each chunk with:
- Source filename
- Page number (if available)
- Score (formatted to 4 decimal places)
- Full text content

## 7. Grounding Data Flow

```
retrieval (dense/bm25/rrf/reranker)
    → eval_context (built from search_res.results)
    → verifier.verify(answer, eval_context)
    → citations (from verification)
    → confidence_estimator.estimate(...)
    → abstention decision
    → response.citations = schema_citations (may be empty if abstained)
    → response.retrieved_chunks = _extract_retrieved_chunks(...)  ← NEW: always populated from retrieval
```

## 8. Four-Mode Verification

| Mode | Retrieval | API Chunks | UI Chunks | Grounding |
|------|-----------|------------|-----------|-----------|
| Dense | PASS | PASS (7) | PASS | PASS |
| BM25 | PASS | PASS (5) | PASS | PASS |
| Hybrid | PASS | PASS (12) | PASS | PASS |
| Hybrid Rerank | PASS | PASS (5) | PASS | PASS |

## 9. Positive Query

Query: "What is the attendance policy?"
- Dense: 7 retrieved chunks returned
- BM25: 5 retrieved chunks returned
- Hybrid: 12 retrieved chunks returned
- Hybrid Rerank: 5 retrieved chunks returned

## 10. Negative Query

Query: "What is the recipe for baking chocolate lava cake?"
- Response still valid with `retrieved_chunks` field present
- Abstention logic unchanged

## 11. Source Filter Verification

Source filter payload accepted by all four modes without error.

## 12. Tests

- Existing tests: 404
- New tests: 21
- Total: 425
- Passed: 425
- Failed: 0

## 13. Performance Impact

- One additional list comprehension per request (negligible)
- No additional database calls
- No additional retrieval execution

## 14. Historical Integrity

Confirmed:
- Retrieval algorithms unchanged
- Grounding unchanged
- Confidence thresholds unchanged
- Abstention logic unchanged
- golden_dataset unchanged
- Historical tags unchanged

## 15. Commit

Pending — ready to commit after verification.

## 16. Release Decision

Phase 13.3 is ready for release after all acceptance criteria pass:
- ✅ Backend returns `retrieved_chunks` for all four modes
- ✅ Frontend renders from `retrieved_chunks`
- ✅ "No matching chunks found" only appears when chunks are genuinely empty
- ✅ All 425 tests pass
- ✅ No retrieval/grounding/confidence algorithms modified
- ✅ Observability logging added
