import json
from urllib import error, request

from backend.schemas import SearchResult
from backend.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


def llm_available() -> bool:
    return bool(OPENROUTER_API_KEY)


def _messages(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{i + 1}] {r.source}" + (f" (page {r.page})" if r.page else "") + f":\n{r.text}"
        for i, r in enumerate(results)
    )
    system = (
        "You are a factual QA assistant. Answer the user's question clearly and concisely "
        "using ONLY the provided Context.\n\n"
        "FORMATTING RULES:\n"
        "1. Do NOT use bullet points starting with asterisks (*) or hyphens (-).\n"
        "2. Use bold titles for headings and major key points (e.g., **1. Section Title:**).\n"
        "3. Keep descriptions in numbered lists or clear paragraphs under bold headers.\n"
        "4. Always cite sources as [1], [2], etc."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def generate_answer(query: str, results: list[SearchResult]) -> str | None:
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
