import json
from urllib import error, request

from backend.schemas import SearchResult
from backend.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


def llm_available() -> bool:
    return bool(OPENROUTER_API_KEY)


def _messages(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"--- CHUNK [{i + 1}] (Source: {r.source}, Page: {r.page or 'N/A'}) ---\n{r.text}"
        for i, r in enumerate(results)  # Pass all retrieved chunks to the LLM
    )
    system = (
        "You are an exact, document-bound QA assistant. You MUST answer the user's question "
        "using ONLY the provided Context below.\n\n"
        "STRICT RULES:\n"
        "1. Do NOT use outside knowledge or make assumptions.\n"
        "2. If the answer is not directly stated in the Context, respond ONLY with: "
        "'I cannot answer this question based on the provided document.'\n"
        "3. Do not invent details or extrapolate.\n"
        "4. Cite your sources using [1], [2], etc., corresponding to the chunk numbers."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def generate_answer(query: str, results: list[SearchResult]) -> str | None:
    ...
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": _messages(query, results),
        "max_tokens": 768,
        "temperature": 0.0,  # <-- Set to 0.0 for deterministic factual responses
    }
    ...
    req = request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (error.HTTPError, error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return None
