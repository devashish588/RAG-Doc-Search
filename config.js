// RAG Document Search — API base URL, auto-detected.
//
// Handles GitHub Pages, same-origin Render serving, and local dev, so no manual
// URL edits are needed per environment. The backend serves the frontend itself
// locally on port 9826 (see main.py).
window.API_BASE = (() => {
  const host = window.location.hostname;

  // Local Development (uvicorn runs on port 9826 in this repo)
  if (host === "localhost" || host === "127.0.0.1") {
    return "http://localhost:9826";
  }

  // Served directly from the Render web service -> same-origin, no CORS
  if (host.includes("onrender.com")) {
    return "";
  }

  // GitHub Pages / External Frontend -> Canonical Render Backend
  return "https://rag-doc-search-1.onrender.com";
})();