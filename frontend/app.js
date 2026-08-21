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

const MAX_RETRIES = 3;
const sleep = ms => new Promise(r => setTimeout(r, ms));

function isRetryable(status) {
  return status === 502 || status === 503 || status === 504;
}

async function api(path, options, attempt = 1) {
  let res, body;
  try {
    res = await fetch(API_BASE + path, options);
    const ct = res.headers.get('content-type') || '';
    body = ct.includes('application/json') ? await res.json() : await res.text();
  } catch (err) {
    if (attempt < MAX_RETRIES) {
      await sleep(3000 + attempt * 4000);
      return api(path, options, attempt + 1);
    }
    throw err;
  }
  if (isRetryable(res.status) && attempt < MAX_RETRIES) {
    await sleep(3000 + attempt * 4000);
    return api(path, options, attempt + 1);
  }
  if (res.status === 429) {
    const retryAfter = res.headers.get('Retry-After') || '';
    throw new Error('Too many requests. ' + (retryAfter ? 'Retry after ' + retryAfter + 's.' : 'Please wait and try again.'));
  }
  if (!res.ok) throw new Error(body?.error?.message || body?.detail || 'Request failed (' + res.status + ')');
  return body;
}

function statusClass(s) {
  return s === 'complete' ? 'ok' : s === 'failed' ? 'fail' : 'muted';
}

function safeText(v) {
  if (v === null || v === undefined || v === '' || v === 'undefined' || v === 'null' || v === 'NaN') return '—';
  return String(v);
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
    el.innerHTML =
      '<div class="doc-top">' +
        '<strong>' + doc.filename + '</strong>' +
        '<span class="pill ' + statusClass(doc.status) + '">' + doc.status + ' · ' + (doc.chunks_indexed ?? 0) + ' chunks</span>' +
      '</div>' +
      '<div class="muted">' + (doc.message ?? '') + '</div>' +
      '<div style="margin-top:8px">' +
        '<button class="danger" data-delete-id="' + doc.document_id + '" data-filename="' + doc.filename + '">Delete</button>' +
      '</div>' +
      (doc.error ? '<div class="fail" style="margin-top:8px">' + doc.error + '</div>' : '');
    docsBox.appendChild(el);
  });
  [...sources].sort().forEach(src => {
    const opt = document.createElement('option');
    opt.value = opt.textContent = src;
    sourceSelect.appendChild(opt);
  });
}

function renderConfidence(conf) {
  if (!conf) { confidenceBox.innerHTML = '<div class="muted">—</div>'; return; }
  var level = conf.level || 'unknown';
  var color = level === 'high' ? 'ok' : level === 'medium' ? '' : 'fail';
  var score = typeof conf.overall_score === 'number' ? (conf.overall_score * 100).toFixed(0) + '%' : '—';
  var abstain = conf.abstention_flag ? ' · <span class="fail">Abstained</span>' : '';
  var html =
    '<div class="doc-top">' +
      '<span class="pill ' + color + '">' + level.charAt(0).toUpperCase() + level.slice(1) + '</span>' +
      '<strong>' + score + '</strong>' +
      abstain +
    '</div>' +
    '<div class="muted">' +
      'Retrieval: ' + (typeof conf.retrieval_confidence === 'number' ? (conf.retrieval_confidence * 100).toFixed(0) + '%' : '—') +
      ' · Grounding: ' + (typeof conf.grounding_confidence === 'number' ? (conf.grounding_confidence * 100).toFixed(0) + '%' : '—') +
    '</div>';
  confidenceBox.innerHTML = html;
}

function renderCitations(data) {
  answerBox.textContent = data.answer || 'No answer returned.';
  resultsBox.innerHTML = '';
  var cites = data.citations || [];
  if (!cites.length) {
    resultsBox.innerHTML = '<div class="muted">No matching chunks found.</div>';
    return;
  }
  cites.forEach(function(c, i) {
    var verdictClass = c.verdict === 'supported' ? 'ok' : c.verdict === 'unsupported' ? 'fail' : 'muted';
    var verdictLabel = c.verdict ? c.verdict.charAt(0).toUpperCase() + c.verdict.slice(1) : '';
    var card = document.createElement('div');
    card.className = 'result';
    card.innerHTML =
      '<div class="result-head">' +
        '<div>#' + (i + 1) + ' · ' + safeText(c.source) + (c.page ? ' · page ' + c.page : '') + '</div>' +
        (verdictLabel ? '<span class="pill ' + verdictClass + '">' + verdictLabel + '</span>' : '') +
      '</div>' +
      '<div class="result-text">' + safeText(c.text_snippet || c.claim) + '</div>';
    resultsBox.appendChild(card);
  });
}

function traceStage(label, items, meta) {
  if (!items || !items.length) return '';
  var n = items.length;
  var extra = meta && items[0] && items[0].metadata && items[0].metadata[meta];
  return '<div class="trace-row"><span class="trace-label">' + label + '</span>' +
       '<span class="trace-val">' + n + (extra ? ' · ' + extra : '') + '</span></div>';
}

function renderTrace(trace) {
  if (!trace) { traceBox.innerHTML = '<div class="muted">—</div>'; return; }
  var html = '';
  html += traceStage('Dense', trace.dense);
  html += traceStage('BM25', trace.bm25);
  html += traceStage('Hybrid RRF', trace.rrf);
  html += traceStage('Reranker', trace.reranker, 'reranker_status');
  if (!html) html = '<div class="muted">—</div>';
  traceBox.innerHTML = html;
}

async function loadHealth() {
  try {
    var d = await api('/health');
    healthPill.textContent = 'API ok · ' + d.embedding_backend;
  } catch (e) {
    healthPill.textContent = 'API error · ' + e.message;
  }
}

async function loadDocuments() {
  try {
    renderDocs(await api('/documents'));
  } catch (e) {
    docsBox.innerHTML = '<div class="fail">' + e.message + '</div>';
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

docsBox.addEventListener('click', function(e) {
  var btn = e.target.closest('[data-delete-id]');
  if (btn) deleteDoc(btn.dataset.deleteId, btn.dataset.filename);
});

$('uploadBtn').addEventListener('click', async function() {
  var file = $('fileInput').files[0];
  if (!file) { uploadStatus.textContent = 'Choose a file first.'; return; }
  $('uploadBtn').disabled = true;
  uploadStatus.textContent = 'Uploading…';
  try {
    var fd = new FormData();
    fd.append('file', file);
    var d = await api('/upload', { method: 'POST', body: fd });
    uploadStatus.textContent = d.filename + ' queued.';
    await loadDocuments();
  } catch (e) {
    uploadStatus.textContent = 'Upload failed: ' + e.message;
  } finally {
    $('uploadBtn').disabled = false;
  }
});

$('searchBtn').addEventListener('click', async function() {
  var question = $('queryInput').value.trim();
  var top_k    = Number($('topKInput').value || 10);
  var top_k_final = Number($('topKFinalInput').value || 5);
  var mode     = modeSelect.value;
  var source   = sourceSelect.value;
  if (!question) { searchStatus.textContent = 'Type a question first.'; return; }
  $('searchBtn').disabled = true;
  searchStatus.textContent = 'Searching…';
  try {
    var payload = {
      question: question,
      retrieval_mode: mode,
      top_k_dense: top_k,
      top_k_sparse: top_k,
      top_k_fused: Math.min(top_k * 2, 50),
      top_k_final: top_k_final,
    };
    if (source) payload.source = source;
    var d = await api('/v1/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderCitations(d);
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
});

$('refreshBtn').addEventListener('click', loadDocuments);

loadHealth();
loadDocuments();
setInterval(loadHealth, 15000);
setInterval(loadDocuments, 15000);
