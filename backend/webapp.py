def build_index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Document Search</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6fb;
      --panel: #ffffff;
      --text: #142033;
      --muted: #5e6a7d;
      --line: #d9e0ea;
      --accent: #2156f3;
      --accent-2: #153ea9;
      --good: #117a43;
      --bad: #b42318;
      --shadow: 0 12px 30px rgba(18,34,66,0.08);
      --radius: 14px;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      background: linear-gradient(180deg, #eef3ff 0%, var(--bg) 18%);
      color: var(--text);
    }
    .shell { max-width: 1280px; margin: 0 auto; padding: 24px; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 28px; }
    .subtle { color: var(--muted); font-size: 14px; margin-top: 6px; }
    .grid {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .stack { display: grid; gap: 12px; }
    label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
    input[type="text"],
    input[type="number"],
    textarea,
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 11px 12px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    textarea { min-height: 108px; resize: vertical; }
    button {
      border: 0;
      border-radius: 10px;
      padding: 11px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.08s ease, opacity 0.2s ease;
    }
    button:hover  { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }
    .primary   { background: var(--accent);  color: #fff; }
    .secondary { background: #e8eefc; color: var(--accent-2); }
    .status {
      padding: 10px 12px;
      border-radius: 10px;
      background: #f7f9fc;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      min-height: 42px;
      white-space: pre-wrap;
    }
    .cards { display: grid; gap: 12px; }
    .doc { border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: #fbfcfe; }
    .doc-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 6px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid var(--line);
      background: #fff;
    }
    .ok   { color: var(--good); }
    .fail { color: var(--bad);  }
    .muted { color: var(--muted); }
    .answer { white-space: pre-wrap; line-height: 1.6; font-size: 15px; }
    .results { display: grid; gap: 12px; margin-top: 12px; }
    .result { border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: #fff; }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .result-text { white-space: pre-wrap; line-height: 1.55; font-size: 14px; }
    .row { display: grid; grid-template-columns: 1fr 100px; gap: 10px; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    .divider { height: 1px; background: var(--line); margin: 16px 0; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .row  { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<div class="shell">
  <header>
    <div>
      <h1>RAG Document Search</h1>
      <div class="subtle">Upload documents · index into ChromaDB · search by meaning.</div>
    </div>
    <div class="pill" id="healthPill">API: checking…</div>
  </header>

  <div class="grid">
    <section class="panel stack">
      <div>
        <label for="fileInput">Upload document</label>
        <input id="fileInput" type="file" accept=".pdf,.txt,.md" />
      </div>
      <button class="primary" id="uploadBtn">Upload</button>
      <div class="status" id="uploadStatus">Ready.</div>

      <div class="divider"></div>

      <div>
        <label for="queryInput">Search query</label>
        <textarea id="queryInput" placeholder="How does AI learn from examples?"></textarea>
      </div>
      <div class="row">
        <div>
          <label for="topKInput">Top results</label>
          <input id="topKInput" type="number" min="1" max="20" value="5" />
        </div>
        <div>
          <label for="sourceSelect">Source</label>
          <select id="sourceSelect">
            <option value="">All</option>
          </select>
        </div>
      </div>
      <div class="actions">
        <button class="primary"   id="searchBtn">Search</button>
        <button class="secondary" id="refreshBtn">Refresh Docs</button>
      </div>
      <div class="status" id="searchStatus">Results will appear on the right.</div>
    </section>

    <section class="stack">
      <div class="panel">
        <div class="pill" style="margin-bottom:10px">Answer</div>
        <div class="answer" id="answerBox">Ask a question after indexing a document.</div>
      </div>
      <div class="panel">
        <div class="pill" style="margin-bottom:10px">Retrieved Chunks</div>
        <div class="results" id="resultsBox"></div>
      </div>
      <div class="panel">
        <div class="pill" style="margin-bottom:10px">Documents</div>
        <div class="cards" id="docsBox"></div>
      </div>
    </section>
  </div>
</div>

<script>
  const $ = id => document.getElementById(id);
  const healthPill   = $('healthPill');
  const uploadStatus = $('uploadStatus');
  const searchStatus = $('searchStatus');
  const answerBox    = $('answerBox');
  const resultsBox   = $('resultsBox');
  const docsBox      = $('docsBox');
  const sourceSelect = $('sourceSelect');

  async function api(path, options) {
    const res = await fetch(path, options);
    const ct  = res.headers.get('content-type') || '';
    const body = ct.includes('application/json') ? await res.json() : await res.text();
    if (!res.ok) throw new Error(body?.detail ?? `Request failed (${res.status})`);
    return body;
  }

  function statusClass(s) {
    return s === 'complete' ? 'ok' : s === 'failed' ? 'fail' : 'muted';
  }

  function renderDocs(docs) {
    docsBox.innerHTML = '';
    sourceSelect.innerHTML = '<option value="">All</option>';
    if (!docs.length) {
      docsBox.innerHTML = '<div class="muted">No documents yet.</div>';
      return;
    }
    const sources = new Set();
    docs.forEach(doc => {
      if (doc.status === 'complete') sources.add(doc.filename);
      const el = document.createElement('div');
      el.className = 'doc';
      el.innerHTML = `
        <div class="doc-top">
          <strong>${doc.filename}</strong>
          <span class="pill ${statusClass(doc.status)}">${doc.status} · ${doc.chunks_indexed ?? 0} chunks</span>
        </div>
        <div class="muted">${doc.message ?? ''}</div>
        ${doc.error ? `<div class="fail" style="margin-top:8px">${doc.error}</div>` : ''}
      `;
      docsBox.appendChild(el);
    });
    [...sources].sort().forEach(src => {
      const opt = document.createElement('option');
      opt.value = opt.textContent = src;
      sourceSelect.appendChild(opt);
    });
  }

  function renderSearch(data) {
    answerBox.textContent = data.answer || 'No answer returned.';
    resultsBox.innerHTML  = '';
    if (!data.results?.length) {
      resultsBox.innerHTML = '<div class="muted">No matching chunks found.</div>';
      return;
    }
    data.results.forEach((item, i) => {
      const card = document.createElement('div');
      card.className = 'result';
      card.innerHTML = `
        <div class="result-head">
          <div>#${i + 1} · ${item.source}${item.page ? ' · page ' + item.page : ''}</div>
          <div>score ${item.score.toFixed(2)}</div>
        </div>
        <div class="result-text">${item.text}</div>
      `;
      resultsBox.appendChild(card);
    });
  }

  async function loadHealth() {
    try {
      const d = await api('/health');
      healthPill.textContent = `API ok · ${d.embedding_backend}`;
    } catch (e) {
      healthPill.textContent = `API error · ${e.message}`;
    }
  }

  async function loadDocuments() {
    try {
      renderDocs(await api('/documents'));
    } catch (e) {
      docsBox.innerHTML = `<div class="fail">${e.message}</div>`;
    }
  }

  $('uploadBtn').addEventListener('click', async () => {
    const file = $('fileInput').files[0];
    if (!file) { uploadStatus.textContent = 'Choose a file first.'; return; }
    $('uploadBtn').disabled = true;
    uploadStatus.textContent = 'Uploading…';
    try {
      const fd = new FormData();
      fd.append('file', file);
      const d = await api('/upload', { method: 'POST', body: fd });
      uploadStatus.textContent = `${d.filename} queued.`;
      await loadDocuments();
    } catch (e) {
      uploadStatus.textContent = `Upload failed: ${e.message}`;
    } finally {
      $('uploadBtn').disabled = false;
    }
  });

  $('searchBtn').addEventListener('click', async () => {
    const query  = $('queryInput').value.trim();
    const top_k  = Number($('topKInput').value || 5);
    const source = sourceSelect.value;
    if (!query) { searchStatus.textContent = 'Type a question first.'; return; }
    $('searchBtn').disabled = true;
    searchStatus.textContent = 'Searching…';
    try {
      const payload = { query, top_k, ...(source && { source }) };
      const d = await api('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      renderSearch(d);
      searchStatus.textContent = `Done in ${d.latency_ms} ms`;
    } catch (e) {
      searchStatus.textContent = `Search failed: ${e.message}`;
    } finally {
      $('searchBtn').disabled = false;
    }
  });

  $('refreshBtn').addEventListener('click', loadDocuments);

  loadHealth();
  loadDocuments();
  setInterval(loadHealth,    15_000);
  setInterval(loadDocuments, 15_000);
</script>
</body>
</html>"""
