const API_BASE = (window.API_BASE || "").replace(/\/+$/, "");

const $ = id => document.getElementById(id);
const healthPill     = $('healthPill');
const uploadStatus   = $('uploadStatus');
const searchStatus   = $('searchStatus');
const answerBox      = $('answerBox');
const resultsBox     = $('resultsBox');
const traceBox       = $('traceBox');
const modeSelect     = $('modeSelect');
const topKFinalInput = $('topKFinalInput');
const docsBox        = $('docsBox');
const sourceSelect   = $('sourceSelect');
const confidenceBox  = $('confidenceBox');
const queryInput     = $('queryInput');

const DASH = '—';
const MAX_RETRIES = 3;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const IN_PROGRESS = new Set(['queued', 'processing', 'indexing']);

// Escape anything that comes from users or documents before it touches
// innerHTML: filenames and chunk text are attacker-controlled on a public
// deployment (an uploaded file could otherwise inject script).
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function isRetryable(status) {
  return status === 502 || status === 503 || status === 504;
}

async function api(path, options, attempt = 1, maxRetries = MAX_RETRIES) {
  let res, body;
  try {
    res = await fetch(API_BASE + path, options);
    const ct = res.headers.get('content-type') || '';
    body = ct.includes('application/json') ? await res.json() : await res.text();
  } catch (err) {
    if (attempt < maxRetries) {
      await sleep(3000 + attempt * 4000);
      return api(path, options, attempt + 1, maxRetries);
    }
    throw err;
  }
  if (isRetryable(res.status) && attempt < maxRetries) {
    await sleep(3000 + attempt * 4000);
    return api(path, options, attempt + 1, maxRetries);
  }
  if (res.status === 429) {
    const retryAfter = res.headers.get('Retry-After') || '';
    throw new Error('Too many requests. ' + (retryAfter ? 'Retry after ' + retryAfter + 's.' : 'Please wait and try again.'));
  }
  if (!res.ok) throw new Error(body?.error?.message || body?.detail || 'Request failed (' + res.status + ')');
  return body;
}

function statusClass(s) {
  if (s === 'complete' || s === 'indexed') return 'ok';
  if (IN_PROGRESS.has(s)) return 'indexing';
  if (s === 'failed' || s === 'error') return 'fail';
  return 'muted';
}

function safeText(v) {
  if (v === null || v === undefined || v === '' || v === 'undefined' || v === 'null' || v === 'NaN') return DASH;
  return String(v);
}

function pct(v) {
  return typeof v === 'number' ? (v * 100).toFixed(0) + '%' : DASH;
}

function renderDocs(docs) {
  const selected = sourceSelect.value;
  docsBox.innerHTML = '';
  sourceSelect.innerHTML = '<option value="">All</option>';
  if (!docs.length) {
    docsBox.innerHTML = '<div class="muted">No documents yet. Upload a PDF, TXT or MD file to get started.</div>';
    return;
  }
  const sources = new Set();
  docs.forEach(doc => {
    if (doc.status === 'complete' || doc.status === 'indexed') sources.add(doc.filename);
    const el = document.createElement('div');
    el.className = 'doc';
    const hasPct = doc.progress_pct !== null && doc.progress_pct !== undefined;
    const chunksText = doc.total_chunks
      ? doc.chunks_indexed + '/' + doc.total_chunks + ' chunks' + (hasPct ? ' (' + Math.round(doc.progress_pct) + '%)' : '')
      : (doc.chunks_indexed ?? 0) + ' chunks';
    const progressBar = (hasPct && IN_PROGRESS.has(doc.status))
      ? '<div class="progress-bar-container"><div class="progress-bar-fill" style="width:' + Math.min(100, Math.max(0, Number(doc.progress_pct) || 0)) + '%;"></div></div>'
      : '';
    el.innerHTML =
      '<div class="doc-top">' +
        '<strong class="doc-name">' + esc(doc.filename) + '</strong>' +
        '<span class="pill ' + statusClass(doc.status) + '">' + esc(doc.status) + ' · ' + esc(chunksText) + '</span>' +
      '</div>' +
      progressBar +
      '<div class="muted" style="margin-top:4px;">' + esc(doc.message) + '</div>' +
      '<div style="margin-top:8px">' +
        '<button class="danger" data-delete-id="' + esc(doc.document_id) + '" data-filename="' + esc(doc.filename) + '">Delete</button>' +
      '</div>' +
      (doc.error ? '<div class="fail" style="margin-top:8px">' + esc(doc.error) + '</div>' : '');
    docsBox.appendChild(el);
  });
  [...sources].sort().forEach(src => {
    const opt = document.createElement('option');
    opt.value = opt.textContent = src;
    sourceSelect.appendChild(opt);
  });
  // Keep the user's source filter across the 30s auto-refresh.
  if (selected && sources.has(selected)) sourceSelect.value = selected;
}

function renderConfidence(conf) {
  if (!conf) { confidenceBox.innerHTML = '<div class="muted">' + DASH + '</div>'; return; }
  const level = String(conf.level || 'unknown');
  const color = level === 'high' ? 'ok' : level === 'medium' ? '' : 'fail';
  const abstain = conf.abstention_flag ? ' · <span class="fail">Abstained</span>' : '';
  confidenceBox.innerHTML =
    '<div class="doc-top">' +
      '<span class="pill ' + color + '">' + esc(level.charAt(0).toUpperCase() + level.slice(1)) + '</span>' +
      '<strong>' + pct(conf.overall_score) + '</strong>' +
      abstain +
    '</div>' +
    '<div class="muted">Retrieval: ' + pct(conf.retrieval_confidence) +
      ' · Grounding: ' + pct(conf.grounding_confidence) + '</div>';
}

function renderAnswer(data) {
  answerBox.textContent = data.answer || 'No answer returned.';
}

function renderChunks(data) {
  resultsBox.innerHTML = '';
  const chunks = data.retrieved_chunks || [];
  if (!chunks.length) {
    resultsBox.innerHTML = '<div class="muted">No matching chunks found.</div>';
    return;
  }
  chunks.forEach((c, i) => {
    const card = document.createElement('div');
    card.className = 'result';
    const pageStr = c.page ? ' · page ' + esc(c.page) : '';
    const scoreStr = typeof c.score === 'number' ? c.score.toFixed(4) : esc(c.score);
    card.innerHTML =
      '<div class="result-head">' +
        '<div>#' + (i + 1) + ' · ' + esc(safeText(c.source)) + pageStr + '</div>' +
        '<span class="pill">Score: ' + scoreStr + '</span>' +
      '</div>' +
      '<div class="result-text">' + esc(safeText(c.text)) + '</div>';
    resultsBox.appendChild(card);
  });
}

function traceStage(label, items, meta) {
  if (!items || !items.length) return '';
  const extra = meta && items[0] && items[0].metadata && items[0].metadata[meta];
  return '<div class="trace-row"><span class="trace-label">' + label + '</span>' +
       '<span class="trace-val">' + items.length + (extra ? ' · ' + esc(extra) : '') + '</span></div>';
}

function renderTrace(trace) {
  if (!trace) { traceBox.innerHTML = '<div class="muted">' + DASH + '</div>'; return; }
  let html = '';
  html += traceStage('Dense', trace.dense);
  html += traceStage('BM25', trace.bm25);
  html += traceStage('Hybrid RRF', trace.rrf);
  html += traceStage('Reranker', trace.reranker, 'reranker_status');
  traceBox.innerHTML = html || '<div class="muted">' + DASH + '</div>';
}

function setHealth(text, cls) {
  healthPill.textContent = text;
  healthPill.className = 'pill' + (cls ? ' ' + cls : '');
}

let apiReady = false;
async function loadHealth() {
  if (!apiReady) setHealth('Connecting… (a sleeping server can take ~1 min to wake)', 'indexing');
  try {
    // Extra retries on the first call to ride out a Render cold start.
    const d = await api('/health', undefined, 1, apiReady ? MAX_RETRIES : 8);
    apiReady = true;
    setHealth('API online · ' + d.embedding_backend, 'ok');
  } catch (e) {
    apiReady = false;
    setHealth('API unreachable · ' + e.message, 'fail');
  }
}

async function loadDocuments() {
  try {
    renderDocs(await api('/documents'));
  } catch (e) {
    docsBox.innerHTML = '<div class="fail">' + esc(e.message) + '</div>';
  }
}

// Poll one upload until ingestion finishes (large PDFs can take minutes).
async function pollSingleDocProgress(docId) {
  for (let attempt = 0; attempt < 200; attempt++) {
    await sleep(3000);
    let doc;
    try {
      doc = await api('/documents/' + docId);
    } catch (e) {
      continue;
    }
    await loadDocuments();
    if (IN_PROGRESS.has(doc.status)) {
      uploadStatus.textContent = doc.filename + ': ' + (doc.message || doc.status);
      continue;
    }
    uploadStatus.textContent = doc.status === 'complete'
      ? doc.filename + ' is ready. Ask a question about it.'
      : doc.filename + ' ' + doc.status + (doc.error ? ': ' + doc.error : '.');
    return;
  }
}

async function deleteDoc(documentId, filename) {
  if (!confirm('Delete "' + filename + '" and remove its indexed chunks?')) return;
  try {
    await api('/documents/' + documentId, { method: 'DELETE' });
    await loadDocuments();
    uploadStatus.textContent = filename + ' deleted.';
  } catch (e) {
    uploadStatus.textContent = 'Delete failed: ' + e.message;
  }
}

docsBox.addEventListener('click', e => {
  const btn = e.target.closest('[data-delete-id]');
  if (btn) deleteDoc(btn.dataset.deleteId, btn.dataset.filename);
});

$('uploadBtn').addEventListener('click', async () => {
  const file = $('fileInput').files[0];
  if (!file) { uploadStatus.textContent = 'Choose a file first.'; return; }
  $('uploadBtn').disabled = true;
  uploadStatus.textContent = 'Uploading ' + file.name + '…';
  try {
    const fd = new FormData();
    fd.append('file', file);
    const d = await api('/upload', { method: 'POST', body: fd });
    uploadStatus.textContent = d.filename + ' queued. Indexing…';
    $('fileInput').value = '';
    await loadDocuments();
    pollSingleDocProgress(d.document_id);
  } catch (e) {
    uploadStatus.textContent = 'Upload failed: ' + e.message;
  } finally {
    $('uploadBtn').disabled = false;
  }
});

async function runSearch() {
  const question = queryInput.value.trim();
  const top_k    = Number($('topKInput').value || 10);
  const top_k_final = Number(topKFinalInput.value || 5);
  const mode     = modeSelect.value;
  const source   = sourceSelect.value;
  if (!question) { searchStatus.textContent = 'Type a question first.'; return; }
  $('searchBtn').disabled = true;
  searchStatus.textContent = 'Searching…';
  try {
    const payload = {
      question: question,
      retrieval_mode: mode,
      top_k_dense: top_k,
      top_k_sparse: top_k,
      top_k_fused: Math.min(top_k * 2, 50),
      top_k_final: top_k_final,
    };
    if (source) payload.source = source;
    const d = await api('/v1/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderAnswer(d);
    renderChunks(d);
    renderConfidence(d.confidence);
    renderTrace(d.retrieval_trace);
    searchStatus.textContent = 'Status: ' + d.status;
  } catch (e) {
    searchStatus.textContent = 'Search failed: ' + e.message;
    answerBox.textContent = '';
    resultsBox.innerHTML = '';
    traceBox.innerHTML = '';
    confidenceBox.innerHTML = '';
  } finally {
    $('searchBtn').disabled = false;
  }
}

$('searchBtn').addEventListener('click', runSearch);

// Enter searches; Shift+Enter inserts a newline.
queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    runSearch();
  }
});

// "Final k" only applies to the reranker.
function syncModeControls() {
  topKFinalInput.disabled = modeSelect.value !== 'hybrid_rerank';
}
modeSelect.addEventListener('change', syncModeControls);
syncModeControls();

$('refreshBtn').addEventListener('click', loadDocuments);

loadHealth().then(loadDocuments);
setInterval(loadHealth, 30000);
setInterval(loadDocuments, 30000);
