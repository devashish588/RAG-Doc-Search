from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>RAG Document Search</title></head>
<body>
  <h1>RAG Document Search</h1>
  <p>Frontend files not found. Run the static frontend from the <code>frontend/</code>
  directory, or serve it separately.</p>
</body>
</html>"""


def build_index_html() -> str:
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        html = index.read_text(encoding="utf-8")
        # Served by this backend -> same-origin API, whatever port it runs on.
        return html.replace(
            '<script src="config.js"></script>',
            '<script>window.API_BASE = "";</script>\n  <script src="config.js"></script>',
            1,
        )
    return FALLBACK_HTML