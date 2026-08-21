"""Citation verification and grounding layer (Phase 7).

Provides:
- Claim extraction from generated answers
- Evidence verification against retrieved context
- Citation mapping for supported claims
- Grounding metrics (ratio, citation coverage, citation accuracy)
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from backend.interfaces import BaseVerifier, RetrievalResult
from backend.vector_store import get_embeddings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A single factual claim extracted from an answer."""
    claim_id: str
    claim: str
    has_inline_citation: bool = False


@dataclass
class VerifiedClaim:
    """A claim after verification against evidence."""
    claim_id: str
    claim: str
    support_score: float
    verdict: str  # "supported" | "unsupported"
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    has_inline_citation: bool = False


@dataclass
class GroundingMetrics:
    """Aggregated grounding metrics for an answer."""
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    grounding_ratio: float = 0.0
    citation_coverage: float = 0.0
    citation_accuracy: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INLINE_CITATION_PATTERN = re.compile(r'\[(\d+)\]|\(source:\s*([^)]+)\)|chunk[_\s]?id\s*[=:]\s*([^\s,]+)')

_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'their', 'his', 'her', 'our',
    'your', 'my', 'we', 'you', 'they', 'he', 'she', 'i', 'me', 'us',
    'them', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
    'why', 'how', 'there', 'here', 'then', 'than', 'so', 'if', 'because'
}


def _tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from text."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _has_inline_citation(text: str) -> bool:
    """Check if text contains an inline citation marker."""
    return bool(_INLINE_CITATION_PATTERN.search(text))


def _strip_inline_citations(text: str) -> str:
    """Remove inline citation markers from text."""
    # Remove [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove (source: ...) and [source: ...] patterns
    text = re.sub(r'[\(\[]source:\s*[^\)\]]+[\)\]]', '', text)
    # Remove [filename, page X] patterns
    text = re.sub(r'\[[^\]]+\.md[^\]]*\]', '', text)
    text = re.sub(r'\[[^\]]+\.txt[^\]]*\]', '', text)
    text = re.sub(r'\[[^\]]+\.pdf[^\]]*\]', '', text)
    # Remove chunk_id=... patterns
    text = re.sub(r'chunk[_\s]?id\s*[=:]\s*[^\s,]+', '', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing punctuation artifacts
    text = re.sub(r'\s+[.,;]+$', '', text)
    return text


def _split_claims(answer: str) -> list[Claim]:
    """Extract individual factual claims from an answer.

    Splits on sentence boundaries, bullet points, numbered lists.
    Filters out purely conversational content (greetings, hedging).
    """
    if not answer or not answer.strip():
        return []

    # Context fallback check: raw context fallback is not an LLM generated answer
    answer_clean = answer.strip().lower()
    if answer_clean.startswith("most relevant context:"):
        return []

    claims = []

    # Remove code blocks and very short fragments
    text = re.sub(r'```[\s\S]*?```', '', answer)
    text = re.sub(r'`[^`]+`', '', text)

    # Split into candidate sentences/lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Remove leading bullet/number markers
        line = re.sub(r'^[\s]*([-*•]|\d+[\.)])\s*', '', line)
        if line:
            lines.append(line)

    # Split all lines on sentence boundaries
    candidates = []
    for line in lines:
        parts = re.split(r'(?<=[.!?])\s+', line)
        candidates.extend(p for p in parts if p.strip())

    # Filter out conversational hedging, refusals, and fallback text
    conversational_starts = (
        'i think', 'i believe', 'it seems', 'it appears', 'perhaps',
        'maybe', 'probably', 'possibly', 'likely', 'generally',
        'typically', 'usually', 'often', 'sometimes', 'note that',
        'please note', 'important:', 'note:', 'disclaimer',
        'here are', 'here is', 'the following', 'below are', 'key points',
        'i cannot answer', 'i am unable to answer', 'i don\'t have',
        'i do not have', 'there is no information', 'the provided text does not',
        'no information is provided', 'based on the provided documents',
    )

    for i, cand in enumerate(candidates):
        cand_lower = cand.lower().strip()
        # Skip very short fragments
        if len(cand_lower) < 10:
            continue
        # Skip conversational hedging and refusals
        if any(cand_lower.startswith(cs) for cs in conversational_starts):
            continue
        # Skip pure questions
        if cand_lower.endswith('?'):
            continue
        # Skip pure greetings/closings
        if cand_lower in ('hello', 'hi', 'thanks', 'thank you', 'you\'re welcome', 'goodbye'):
            continue

        claims.append(Claim(
            claim_id=f"claim_{i+1}",
            claim=_strip_inline_citations(cand),
            has_inline_citation=_has_inline_citation(cand)
        ))

    return claims


def _claims_to_dicts(claims: list[Claim]) -> list[dict]:
    """Convert Claim objects to dicts for API compatibility."""
    return [{"claim_id": c.claim_id, "claim": c.claim, "has_inline_citation": c.has_inline_citation} for c in claims]


def _vc_to_dict(vc: VerifiedClaim) -> dict[str, Any]:
    return {
        "claim_id": vc.claim_id,
        "claim": vc.claim,
        "support_score": vc.support_score,
        "verdict": vc.verdict,
        "source": vc.source,
        "page": vc.page,
        "chunk_id": vc.chunk_id,
        "has_inline_citation": vc.has_inline_citation,
    }


# ---------------------------------------------------------------------------
# Standalone normalization functions (for test compatibility)
# ---------------------------------------------------------------------------

def normalize_dense_score(score: float) -> float:
    """Normalize dense score to [0, 1] via (score + 1) / 2 clamped."""
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def normalize_bm25_scores(scores: list[float]) -> list[float]:
    """Normalize BM25 scores to [0, 1] using min-max."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s <= min_s:
        return [1.0] * len(scores)
    return [max(0.0, min(1.0, (s - min_s) / (max_s - min_s))) for s in scores]


def normalize_rrf_score(score: float) -> float:
    """Normalize RRF score to [0, 1] using theoretical max."""
    from backend.settings import RRF_K, RRF_DENSE_WEIGHT, RRF_SPARSE_WEIGHT
    max_possible = (RRF_DENSE_WEIGHT + RRF_SPARSE_WEIGHT) / (RRF_K + 1)
    if max_possible <= 0:
        return 0.0
    return max(0.0, min(1.0, score / max_possible))


def normalize_reranker_score(score: Optional[float]) -> Optional[float]:
    """Normalize reranker score - if None, return None."""
    if score is None:
        return None
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Citation Verifier
# ---------------------------------------------------------------------------

class CitationVerifier(BaseVerifier):
    """Verifies factual claims against retrieved evidence.

    Uses the existing embedding model (FastEmbed) to compute semantic
    similarity between claims and evidence chunks. No new retrieval is performed.
    """

    def __init__(
        self,
        support_threshold: float = 0.65,
        embedding_model: Any = None,
    ):
        self.support_threshold = support_threshold
        self._embeddings = embedding_model

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def extract_claims(self, answer: str) -> list[dict]:
        """Extract factual claims from an answer. Returns list of dicts."""
        claims = _split_claims(answer)
        return _claims_to_dicts(claims)

    def verify(
        self,
        answer: str,
        context: list[RetrievalResult],
    ) -> dict[str, Any]:
        """Verify claims against evidence and compute grounding metrics."""
        claims = self.extract_claims(answer)
        verified = self._verify_claims(claims, context)
        metrics = self._compute_metrics(verified)

        # Build citation map for supported claims
        citations = []
        for vc in verified:
            if vc.verdict == "supported" and vc.chunk_id:
                citations.append({
                    "claim": vc.claim,
                    "source": vc.source,
                    "page": vc.page,
                    "chunk_id": vc.chunk_id,
                    "verdict": vc.verdict,
                })
            elif vc.verdict == "unsupported":
                citations.append({
                    "claim": vc.claim,
                    "source": None,
                    "page": None,
                    "chunk_id": None,
                    "verdict": vc.verdict,
                })

        return {
            "claims": [_vc_to_dict(vc) for vc in verified],
            "citations": citations,
            "metrics": self._metrics_to_dict(metrics),
        }

    def _verify_claims(
        self,
        claims: list[dict],
        context: list[RetrievalResult],
    ) -> list[VerifiedClaim]:
        """Verify each claim against the evidence context."""
        if not claims:
            return []

        if not context:
            # No evidence = all unsupported
            return [
                VerifiedClaim(
                    claim_id=c["claim_id"],
                    claim=c["claim"],
                    support_score=0.0,
                    verdict="unsupported",
                    has_inline_citation=c.get("has_inline_citation", False),
                )
                for c in claims
            ]

        # Prepare evidence texts and embeddings
        evidence_texts = [c.content for c in context]
        claim_texts = [c["claim"] for c in claims]

        try:
            # Embed claims and evidence (reuse existing embedder)
            claim_embeddings = self.embeddings.embed_documents(claim_texts)
            evidence_embeddings = self.embeddings.embed_documents(evidence_texts)
        except Exception as exc:
            log.warning("Embedding failed during verification: %s", exc)
            # Fallback: lexical token overlap
            return self._verify_lexical(claims, context)

        verified = []
        for i, claim in enumerate(claims):
            claim_emb = claim_embeddings[i]
            best_score = 0.0
            best_idx = -1

            claim_lower = claim["claim"].lower().strip()
            for j, ev_emb in enumerate(evidence_embeddings):
                score = self._cosine_similarity(claim_emb, ev_emb)
                ev_lower = context[j].content.lower()

                # Exact claim substring in evidence chunk
                if len(claim_lower) >= 8 and claim_lower in ev_lower:
                    score = max(score, 1.0)

                if score > best_score:
                    best_score = score
                    best_idx = j

            # ponytail: dual gate — cosine high AND lexical overlap >= 20%
            # prevents unrelated short sentences from passing on cosine alone
            claim_tokens = _tokenize(claim["claim"])
            ev_tokens = _tokenize(context[best_idx].content) if best_idx >= 0 else set()
            overlap_ratio = len(claim_tokens & ev_tokens) / len(claim_tokens) if claim_tokens else 0.0
            lexically_grounded = overlap_ratio >= 0.20
            verdict = "supported" if (best_score >= self.support_threshold and lexically_grounded) else "unsupported"
            ev = context[best_idx] if best_idx >= 0 else None

            verified.append(VerifiedClaim(
                claim_id=claim["claim_id"],
                claim=claim["claim"],
                support_score=round(best_score, 4),
                verdict=verdict,
                source=ev.source if ev else None,
                page=ev.metadata.get("page") if ev else None,
                chunk_id=ev.chunk_id if ev else None,
                has_inline_citation=claim.get("has_inline_citation", False),
            ))

        return verified

    def _verify_lexical(
        self,
        claims: list[dict],
        context: list[RetrievalResult],
    ) -> list[VerifiedClaim]:
        """Fallback lexical verification using token overlap."""
        verified = []
        for claim in claims:
            claim_tokens = _tokenize(claim["claim"])
            if not claim_tokens:
                verified.append(VerifiedClaim(
                    claim_id=claim["claim_id"],
                    claim=claim["claim"],
                    support_score=0.0,
                    verdict="unsupported",
                    has_inline_citation=claim.get("has_inline_citation", False),
                ))
                continue

            best_score = 0.0
            best_ev = None
            for ev in context:
                ev_tokens = _tokenize(ev.content)
                if not ev_tokens:
                    continue
                overlap = len(claim_tokens & ev_tokens)
                score = overlap / len(claim_tokens)
                if score > best_score:
                    best_score = score
                    best_ev = ev

            verdict = "supported" if best_score >= self.support_threshold else "unsupported"
            verified.append(VerifiedClaim(
                claim_id=claim["claim_id"],
                claim=claim["claim"],
                support_score=round(best_score, 4),
                verdict=verdict,
                source=best_ev.source if best_ev else None,
                page=best_ev.metadata.get("page") if best_ev else None,
                chunk_id=best_ev.chunk_id if best_ev else None,
                has_inline_citation=claim.get("has_inline_citation", False),
            ))
        return verified

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _compute_metrics(self, verified: list[VerifiedClaim]) -> GroundingMetrics:
        """Compute grounding metrics from verified claims."""
        total = len(verified)
        if total == 0:
            return GroundingMetrics()

        supported = sum(1 for v in verified if v.verdict == "supported")
        unsupported = total - supported

        # Cited claims: supported OR has inline citation
        cited = sum(1 for v in verified if v.verdict == "supported" or v.has_inline_citation)
        supported_cited = sum(1 for v in verified if v.verdict == "supported")

        return GroundingMetrics(
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            grounding_ratio=round(supported / total, 4) if total > 0 else 0.0,
            citation_coverage=round(cited / total, 4) if total > 0 else 0.0,
            citation_accuracy=round(supported_cited / cited, 4) if cited > 0 else 0.0,
        )

    def _metrics_to_dict(self, m: GroundingMetrics) -> dict[str, Any]:
        return {
            "total_claims": m.total_claims,
            "supported_claims": m.supported_claims,
            "unsupported_claims": m.unsupported_claims,
            "grounding_ratio": m.grounding_ratio,
            "citation_coverage": m.citation_coverage,
            "citation_accuracy": m.citation_accuracy,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VERIFIER_INSTANCE: Optional[CitationVerifier] = None


def get_verifier() -> CitationVerifier:
    """Get or create the singleton CitationVerifier instance."""
    global _VERIFIER_INSTANCE
    if _VERIFIER_INSTANCE is None:
        _VERIFIER_INSTANCE = CitationVerifier()
    return _VERIFIER_INSTANCE


def reset_verifier() -> None:
    """Reset the singleton (mainly for testing)."""
    global _VERIFIER_INSTANCE
    _VERIFIER_INSTANCE = None