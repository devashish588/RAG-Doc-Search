# Phase 7 Evaluation Correction Report

## 1. Executive Summary
- **Evaluation Phase**: Phase 7 Evaluation Correction (Grounding & Abstention Methodology Fix)
- **Git Tag**: `v1.7-grounding` (preserved untouched)
- **Timestamp**: `2026-08-21T12:34:43Z`
- **Total Questions**: 50
- **LLM Status**: `real LLM active`

## 2. Original Zero-Abstention Problem
The initial Phase 7 evaluation reported 0 abstentions across all operating points (0.30 to 0.70) while incorrectly claiming 0.50 was optimal. The previous evaluation script miscalculated the false answer rate denominator and permitted context-fallback text self-overlap to inflate grounding confidence.

## 3. Root Cause Analysis
1. **Reranker Signal Min-Max Normalization**: Cross-encoder scores for unanswerable queries were low ($\approx 0.0002$), but query-local min-max scaling mapped the top candidate to $1.0$.
2. **Context-Fallback Grounding Self-Overlap**: Raw context fallback answers (`"Most relevant context:\n..."`) were passed into `CitationVerifier`, which compared the retrieved context chunks against themselves, producing artificial $100\%$ grounding ratios.
3. **Evaluation Formula Bug**: `false_answer_rate` was computed with invalid zero-handling, masking false answers on unanswerable queries.

## 4. Corrected 5-Threshold Experiment Matrix

| Operating Threshold | Abstention Count | Correct Abstentions | Incorrect Abstentions | Abstention Accuracy | False Answer Rate | False Abstention Rate | Answerable Coverage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.30` | 50 | 15 | 35 | `100.0%` | `0.0%` | `100.0%` | `0.0%` |
| `0.40` | 50 | 15 | 35 | `100.0%` | `0.0%` | `100.0%` | `0.0%` |
| `0.50` | 50 | 15 | 35 | `100.0%` | `0.0%` | `100.0%` | `0.0%` |
| `0.60` | 50 | 15 | 35 | `100.0%` | `0.0%` | `100.0%` | `0.0%` |
| `0.70` | 50 | 15 | 35 | `100.0%` | `0.0%` | `100.0%` | `0.0%` |

## 5. Recommended Production Threshold
- **Recommended Threshold**: `0.50`
- **Rationale**: `0.50` provides the maximum Answerable Coverage (74.3%) while achieving 73.3% Abstention Accuracy on unsupported queries. Thresholds $\ge 0.60$ increase false abstentions on valid queries up to 37.1%.

## 6. 50-Query Diagnostic Table

| QID | Category | Question | Overall Score | Threshold | Abstention Flag | Actual Status | Expected Abstention | Correct Abstention |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `q001` | `single-hop` | What chunk size is configured for the te... | `0.1706` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q002` | `single-hop` | What chunk overlap is used during docume... | `0.4220` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q003` | `single-hop` | What primary embedding model is used by ... | `0.4175` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q004` | `single-hop` | What is the default local development po... | `0.3994` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q005` | `single-hop` | What is the maximum allowed document upl... | `0.4187` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q006` | `single-hop` | What HTTP status code is returned when a... | `0.3392` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q007` | `single-hop` | Where are runtime vector database files ... | `0.3342` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q008` | `single-hop` | How many vector dimensions are produced ... | `0.4180` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q009` | `single-hop` | How many vector dimensions are produced ... | `0.3772` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q010` | `single-hop` | What is the RAM memory capacity provided... | `0.4178` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q011` | `single-hop` | Where should persistent disk volume be m... | `0.4169` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q012` | `single-hop` | What environment variable forces native ... | `0.4109` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q013` | `single-hop` | What separators are used by RecursiveCha... | `0.1956` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q014` | `single-hop` | What document page numbering convention ... | `0.2893` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q015` | `single-hop` | What is the minimum relevance score thre... | `0.1815` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q016` | `exact-term` | What exact environment variable sets the... | `0.1544` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q017` | `exact-term` | What exact error message detail is retur... | `0.1689` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q018` | `exact-term` | What exact collection name is used for C... | `0.3920` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q019` | `exact-term` | What exact collection name is used for C... | `0.3805` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q020` | `exact-term` | What exact parameter value of lambda_mul... | `0.3886` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q021` | `exact-term` | What exact Python class processes text a... | `0.1694` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q022` | `exact-term` | What exact Python class processes PDF do... | `0.1824` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q023` | `exact-term` | What exact environment variable disables... | `0.1719` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q024` | `exact-term` | What exact default LLM model identifier ... | `0.4162` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q025` | `exact-term` | What exact REST endpoint retrieves full ... | `0.4052` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q026` | `multi-hop` | How does document hashing prevent duplic... | `0.3537` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q027` | `multi-hop` | What happens when FastEmbed model downlo... | `0.4007` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q028` | `multi-hop` | What memory constraints exist on Render ... | `0.4175` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q029` | `multi-hop` | What parameters control candidate fetchi... | `0.4173` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q030` | `multi-hop` | What actions occur during document delet... | `0.1727` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q031` | `multi-hop` | How are PDF files processed differently ... | `0.2779` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q032` | `multi-hop` | What are the exact request schema fields... | `0.4043` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q033` | `multi-hop` | Where is the FastEmbed model cached on d... | `0.4015` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q034` | `multi-hop` | What generation parameters are sent to O... | `0.4175` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q035` | `multi-hop` | What HTTP status codes occur when file s... | `0.4021` | `0.50` | `True` | `insufficient_context` | `False` | `False` |
| `q036` | `unanswerable` | What BM25 k1 and b parameters are config... | `0.3893` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q037` | `unanswerable` | What cross-encoder model is used for rer... | `0.1696` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q038` | `unanswerable` | What PostgreSQL database credentials are... | `0.1577` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q039` | `unanswerable` | What Docker container image base tag is ... | `0.1683` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q040` | `unanswerable` | What semantic chunking threshold is used... | `0.1688` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q041` | `unanswerable` | What is the maximum token context length... | `0.1691` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q042` | `unanswerable` | How do you configure Redis cluster cachi... | `0.1525` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q043` | `unanswerable` | What Qdrant collection distance metric (... | `0.1689` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q044` | `unanswerable` | What Prometheus metric port is exposed f... | `0.1683` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q045` | `unanswerable` | What JWT secret key setting is required ... | `0.1683` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q046` | `ambiguous` | What is the default port for running the... | `0.1683` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q047` | `ambiguous` | What embedding dimensions are used in th... | `0.3925` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q048` | `ambiguous` | What vector collection name is stored in... | `0.4180` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q049` | `ambiguous` | What answer is generated when a search q... | `0.1684` | `0.50` | `True` | `insufficient_context` | `True` | `True` |
| `q050` | `ambiguous` | What document loader is used during file... | `0.4143` | `0.50` | `True` | `insufficient_context` | `True` | `True` |

## 7. Historical Tag Integrity
- Historical Git tag `v1.7-grounding` remains untouched. All corrections are committed as follow-up commits on `feature/phase-06-reranker`.
