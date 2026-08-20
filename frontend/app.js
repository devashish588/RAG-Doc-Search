const API_BASE = (window.API_BASE || "").replace(/\/+$/, "");

const $ = id => document.getElementById(id);
const healthPill   = $('healthPill');
const uploadStatus = $('uploadStatus');
const searchStatus = $('searchStatus');
const answerBox    = $('answerBox');
const resultsBox   = $('resultsBox');
const traceBox     = $('traceBox');
const modeSelect   = $('modeSelect');
const topKFinalInput = $('topKFinalInput');
const docsBox      = $('docsBox');
const sourceSelect = $('sourceSelect');

const MAX_RETRIES = 3;

const sleep = ms => new Promise(r => setTimeout(r, ms));

function isRetryable(status) {
  return status === 502 || status === 503 || status === 504;
}

async function api(path, options, attempt = 1) {
  let res, body;
  try {
    res = await fetch(API_BASE + path, options);
    const ct  = res.headers.get('content-type') || '';
    body = ct.includes('application/json')
      ? await res.json()
      : await res.text();
  } catch (err) {
    // Network-level failure (e.g. Render cold start drops the connection).
    if (attempt < MAX_RETRIES) {
      await sleep(3000 + attempt * 4000);
      return api(path, options, attempt + 1);
    }
    throw err;
  }
  // Server was waking up / momentarily unavailable — retry.
  if (isRetryable(res.status) && attempt < MAX_RETRIES) {
    await sleep(3000 + attempt * 4000);
    return api(path, options, attempt + 1);
  }
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
      <div style="margin-top:8px">
        <button class="danger" data-delete-id="${doc.document_id}" data-filename="${doc.filename}">Delete</button>
      </div>
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

function renderCitations(data) {
  answerBox.textContent = data.answer || 'No answer returned.';
  resultsBox.innerHTML  = '';
  const cites = data.citations || [];
  if (!cites.length) {
    resultsBox.innerHTML = '<div class="muted">No matching chunks found.</div>';
    return;
  }
  cites.forEach((c, i) => {
    const card = document.createElement('div');
    card.className = 'result';
    card.innerHTML = `
      <div class="result-head">
        <div>#${i + 1} · ${c.source || '?'}${c.page ? ' · page ' + c.page : ''}</div>
      </div>
      <div class="result-text">${c.text_snippet || ''}</div>
    `;
    resultsBox.appendChild(card);
  });
}

function traceStage(label, items, meta) {
  if (!items) return '';
  const n = items.length;
  const extra = meta && items[0]?.metadata?.[meta];
  return `<div class="trace-row"><span class="trace-label">${label}</span>`
       + `<span class="trace-val">${n}${extra ? ' · ' + extra : ''}</span></div>`;
}

function renderTrace(trace) {
  if (!trace) { traceBox.innerHTML = '<div class="muted">—</div>'; return; }
  let html = '';
  html += traceStage('Dense', trace.dense);
  html += traceStage('BM25', trace.bm25);
  html += traceStage('Hybrid RRF', trace.rrf);
  html += traceStage('Reranker', trace.reranker, 'reranker_status');
  traceBox.innerHTML = html;
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

async function deleteDoc(documentId, filename) {
  if (!confirm(`Delete "${filename}" and remove its indexed chunks?`)) return;
  try {
    await api(`/documents/${documentId}`, { method: 'DELETE' });
    await loadDocuments();
    uploadStatus.textContent = `${filename} deleted.`;
  } catch (e) {
    uploadStatus.textContent = `Delete failed: ${e.message}`;
  }
}

docsBox.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-delete-id]');
  if (btn) deleteDoc(btn.dataset.deleteId, btn.dataset.filename);
});

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
  const question = $('queryInput').value.trim();
  const top_k    = Number($('topKInput').value || 8);
  const top_k_final = Number($('topKFinalInput').value || 5);
  const mode     = modeSelect.value;
  const source   = sourceSelect.value;
  if (!question) { searchStatus.textContent = 'Type a question first.'; return; }
  $('searchBtn').disabled = true;
  searchStatus.textContent = 'Searching…';
  try {
    const payload = {
      question,
      retrieval_mode: mode,
      top_k_final,
      top_k,
      ...(source && { source }),
    };
    const d = await api('/v1/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderCitations(d);
    renderTrace(d.retrieval_trace);
    searchStatus.textContent = `Status: ${d.status}`;
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