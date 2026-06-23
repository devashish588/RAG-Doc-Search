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
      --shadow: 0 12px 30px rgba(18, 34, 66, 0.08);
      --radius: 14px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: linear-gradient(180deg, #eef3ff 0%, var(--bg) 18%, var(--bg) 100%);
      color: var(--text);
    }

    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
    }

    .subtle {
      color: var(--muted);
      font-size: 14px;
      margin-top: 6px;
    }

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

    .stack {
      display: grid;
      gap: 12px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
    }

    input[type="text"], input[type="number"], textarea, select {
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
      transition: transform 0.08s ease, opacity 0.2s ease, background 0.2s ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

    .primary { background: var(--accent); color: white; }
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

    .cards {
      display: grid;
      gap: 12px;
    }

    .doc {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fbfcfe;
    }

    .doc-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
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

    .ok { color: var(--good); }
    .fail { color: var(--bad); }

    .muted { color: var(--muted); }
    .answer {
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 15px;
    }

    .results {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }

    .result {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fff;
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
    }

    .result-text {
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 14px;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 100px;
      gap: 10px;
    }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .divider {
      height: 1px;
      background: var(--line);
      margin: 16px 0;
    }

    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>RAG Document Search</h1>
        <div class="subtle">Upload documents, index them into ChromaDB, and search by meaning from one port.</div>
      </div>
      <div class="pill" id="healthPill">API: checking...</div>
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
              <option value="">All documents</option>
            </select>
          </div>
        </div>
        <div class="actions">
          <button class="primary" id="searchBtn">Search</button>
          <button class="secondary" id="refreshBtn">Refresh Docs</button>
        </div>
        <div class="status" id="searchStatus">Search results will appear on the right.</div>
      </section>

      <section class="stack">
        <div class="panel">
          <div class="pill" style="margin-bottom: 10px;">Answer</div>
          <div class="answer" id="answerBox">Ask a question after indexing a document.</div>
        </div>

        <div class="panel">
          <div class="pill" style="margin-bottom: 10px;">Retrieved Chunks</div>
          <div class="results" id="resultsBox"></div>
        </div>

        <div class="panel">
          <div class="pill" style="margin-bottom: 10px;">Documents</div>
          <div class="cards" id="docsBox"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const healthPill = document.getElementById('healthPill');
    const uploadBtn = document.getElementById('uploadBtn');
    const searchBtn = document.getElementById('searchBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const searchStatus = document.getElementById('searchStatus');
    const answerBox = document.getElementById('answerBox');
    const resultsBox = document.getElementById('resultsBox');
    const docsBox = document.getElementById('docsBox');
    const sourceSelect = document.getElementById('sourceSelect');

    async function api(path, options) {
      const response = await fetch(path, options);
      const contentType = response.headers.get('content-type') || '';
      const payload = contentType.includes('application/json') ? await response.json() : await response.text();
      if (!response.ok) {
        const detail = payload && payload.detail ? payload.detail : (typeof payload === 'string' ? payload : `Request failed (${response.status})`);
        throw new Error(detail);
      }
      return payload;
    }

    function statusClass(status) {
      return status === 'complete' ? 'ok' : (status === 'failed' ? 'fail' : 'muted');
    }

    function renderDocs(docs) {
      docsBox.innerHTML = '';
      const sourceOptions = new Set();
      sourceSelect.innerHTML = '<option value="">All documents</option>';

      if (!docs.length) {
        docsBox.innerHTML = '<div class="muted">No uploaded documents yet.</div>';
        return;
      }

      docs.forEach((doc) => {
        const source = doc.filename || 'unknown';
        if (doc.status === 'complete') {
          sourceOptions.add(source);
        }

        const el = document.createElement('div');
        el.className = 'doc';
        el.innerHTML = `
          <div class="doc-top">
            <strong>${source}</strong>
            <span class="pill ${statusClass(doc.status)}">${doc.status} · ${doc.chunks_indexed || 0} chunks</span>
          </div>
          <div class="muted">${doc.message || ''}</div>
          ${doc.error ? `<div class="fail" style="margin-top: 8px;">${doc.error}</div>` : ''}
        `;
        docsBox.appendChild(el);
      });

      [...sourceOptions].sort().forEach((source) => {
        const opt = document.createElement('option');
        opt.value = source;
        opt.textContent = source;
        sourceSelect.appendChild(opt);
      });
    }

    function renderSearch(data) {
      answerBox.textContent = data.answer || 'No answer returned.';
      resultsBox.innerHTML = '';

      if (!data.results || data.results.length === 0) {
        resultsBox.innerHTML = '<div class="muted">No matching chunks were found.</div>';
        return;
      }

      data.results.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'result';
        card.innerHTML = `
          <div class="result-head">
            <div>#${index + 1} · ${item.source}${item.page ? ' · page ' + item.page : ''}</div>
            <div>score ${Number(item.score).toFixed(2)}</div>
          </div>
          <div class="result-text">${item.text}</div>
        `;
        resultsBox.appendChild(card);
      });
    }

    async function loadHealth() {
      try {
        const data = await api('/health');
        healthPill.textContent = `API ok · embedding: ${data.embedding_backend}`;
      } catch (err) {
        healthPill.textContent = `API error · ${err.message}`;
      }
    }

    async function loadDocuments() {
      const docs = await api('/documents');
      renderDocs(docs);
    }

    uploadBtn.addEventListener('click', async () => {
      const fileInput = document.getElementById('fileInput');
      const file = fileInput.files[0];
      if (!file) {
        uploadStatus.textContent = 'Choose a file first.';
        return;
      }

      uploadBtn.disabled = true;
      uploadStatus.textContent = 'Uploading...';
      try {
        const formData = new FormData();
        formData.append('file', file);
        const data = await api('/upload', { method: 'POST', body: formData });
        uploadStatus.textContent = `${data.filename} queued. Refreshing document list...`;
        await loadDocuments();
      } catch (err) {
        uploadStatus.textContent = `Upload failed: ${err.message}`;
      } finally {
        uploadBtn.disabled = false;
      }
    });

    searchBtn.addEventListener('click', async () => {
      const query = document.getElementById('queryInput').value.trim();
      const topK = Number(document.getElementById('topKInput').value || 5);
      const source = sourceSelect.value;
      if (!query) {
        searchStatus.textContent = 'Type a question first.';
        return;
      }

      searchBtn.disabled = true;
      searchStatus.textContent = 'Searching...';
      try {
        const payload = { query, top_k: topK };
        if (source) payload.source = source;
        const data = await api('/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        renderSearch(data);
        searchStatus.textContent = `Done in ${data.latency_ms} ms`;
      } catch (err) {
        searchStatus.textContent = `Search failed: ${err.message}`;
      } finally {
        searchBtn.disabled = false;
      }
    });

    refreshBtn.addEventListener('click', loadDocuments);

    loadHealth();
    loadDocuments();
    setInterval(loadHealth, 15000);
    setInterval(loadDocuments, 15000);
  </script>
</body>
</html>"""

