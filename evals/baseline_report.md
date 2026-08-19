# HybridRAG Phase 0 Baseline Evaluation Report

## 1. Executive Summary
- **Evaluation Phase**: Phase 0 - Baseline Freeze & Measurement
- **Git Baseline Tag**: `v1.0-baseline`
- **Execution Timestamp**: `2026-08-19T18:42:22Z`
- **Fixed Corpus Files**: `api_reference.txt`, `architecture_overview.md`, `database_and_storage_spec.txt`, `troubleshooting_guide.pdf`
- **Total Golden Questions Evaluated**: 50
- **Chunking Configuration**: `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200`
- **Retrieval Configuration**: Dense Vector Search with MMR (`fetch_k=20`, `k=8`, `lambda_mult=0.5`, `MIN_RELEVANCE_SCORE=0.35`)

## 2. Summary Metrics Table

| Metric Category | Metric | Baseline Value | Standard / Target |
| :--- | :--- | :--- | :--- |
| **Retrieval** | **Recall@1** | `78.0%` | ≥ 70.0% |
| **Retrieval** | **Recall@5** | `100.0%` | ≥ 85.0% |
| **Retrieval** | **Recall@10** | `100.0%` | ≥ 90.0% |
| **Retrieval** | **MRR (Mean Reciprocal Rank)** | `0.8867` | ≥ 0.7500 |
| **Generation** | **Answer Correctness Score** | `0.2909` | ≥ 0.8000 |
| **Generation** | **Faithfulness / Grounding** | `100.0%` | 100% |
| **Citation** | **Citation Coverage** | `78.0%` | ≥ 80.0% |
| **Abstention** | **Abstention Accuracy** | `16.7%` | ≥ 90.0% |
| **System** | **Mean Latency (ms)** | `2118.71 ms` | ≤ 500 ms |
| **System** | **P95 Latency (ms)** | `3806.21 ms` | ≤ 1000 ms |

## 3. Detailed Category Breakdown

| Category | Questions | Recall@1 | Recall@5 | MRR | Answer Correctness | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `single-hop` | 15 | `66.7%` | `100.0%` | `0.8333` | `0.4614` | `2158.04 ms` |
| `exact-term` | 10 | `70.0%` | `100.0%` | `0.8333` | `0.1309` | `1637.91 ms` |
| `multi-hop` | 10 | `100.0%` | `100.0%` | `1.0000` | `0.2989` | `2722.48 ms` |
| `unanswerable` | 10 | `100.0%` | `100.0%` | `1.0000` | `0.1777` | `1931.65 ms` |
| `ambiguous` | 5 | `40.0%` | `100.0%` | `0.7000` | `0.3097` | `2128.89 ms` |

## 4. Failure Mode Analysis & Baseline Limitations

1. **Exact-Term Mismatch on Dense Vectors**:
   - Technical identifiers, environment flags (e.g. `TOKENIZERS_PARALLELISM`), and exact error detail strings occasionally rank lower in dense vector similarity.
   - *Phase 4 Remedy*: Introduce BM25 sparse keyword index with Reciprocal Rank Fusion (RRF).

2. **Chunk Boundary Splitting**:
   - Although adjacent chunk expansion (`_fetch_adjacent_chunks`) retrieves chunk `N+1`, complex multi-hop evidence spanning disparate sections requires semantic chunking.
   - *Phase 2 & Phase 3 Remedy*: Upgrade chunking engine and introduce reranking.

3. **Abstention Guardrails**:
   - Queries for unsupported domain topics currently rely on LLM system prompt instructions or context fallback messages.
   - *Phase 6 Remedy*: Implement explicit abstention classifiers and confidence estimation.

## 5. Exit Gate Validation Status
- [x] Baseline tag `v1.0-baseline` created
- [x] Fixed evaluation corpus versioned in `evals/corpus/`
- [x] 50+ golden questions annotated in `evals/golden_dataset.json`
- [x] Baseline evaluation pipeline executed cleanly
- [x] Results saved to `evals/results/baseline_results.json`
- [x] Baseline report generated
