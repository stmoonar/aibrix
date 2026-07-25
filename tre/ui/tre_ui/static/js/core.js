/* TRE Console -- shared state, helpers, view registry and the SSE feed.

   Live data arrives ONLY via /api/stream (SSE, 0.5s). /api/meta is fetched once
   for static params. The browser never polls upstream: the in-pod sampler owns
   all reads, so N open tabs still cost the control loop nothing. */
'use strict';

export const MODEL_HUES = ['var(--m1)', 'var(--m2)', 'var(--m3)', 'var(--m4)'];

export const S = {
  meta: null,
  colors: {},
  snap: null,
  mode: 'active',
  view: 'overview',
  audit: { ran_at_ms: null, result: null },
};

export const $ = (sel, root = document) => root.querySelector(sel);

export const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};

export const fmt = (v, d = 3) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
export const fmtInt = (v) => (v == null) ? '—' : String(v);

export const ageText = (ms) => ms == null ? '—'
  : (ms < 1500 ? 'now' : ms < 60000 ? Math.round(ms / 1000) + 's' : Math.round(ms / 60000) + 'm') + ' ago';

export const clockText = (ms) => {
  if (!ms) return '—';
  const d = new Date(ms);
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, '0')).join(':');
};

/* An idle model and a broken signal pipeline are different situations. The old
   console rendered both as an em dash, which reads as "the system is down". */
export function zText(z, state) {
  if (z != null && !Number.isNaN(z)) return Number(z).toFixed(3);
  return String(state || '').toLowerCase() === 'idle' ? 'idle · 无负载' : '信号缺失';
}

export function freshness(ageMs, staleMs = 15000) {
  if (ageMs == null) return 'unknown';
  return ageMs > staleMs ? 'stale' : 'fresh';
}

export function toast(msg, isErr) {
  let t = $('#toast');
  if (!t) { t = el('div'); t.id = 'toast'; t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = msg;
  t.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.className = 'toast'; }, 3200);
}

export async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const b = await r.text().catch(() => '');
    throw new Error(`${r.status} ${b.slice(0, 160)}`);
  }
  return r.status === 204 ? {} : r.json();
}

export async function confirmOp(question, fn, okMsg) {
  if (!window.confirm(question)) return;
  try { await fn(); toast(okMsg || 'done'); } catch (e) { toast('failed: ' + e.message, true); }
}

export function emptyState(txt) {
  return Object.assign(el('div', 'empty-state'), { textContent: txt });
}

/* ---------- view registry ---------- */

const VIEWS = new Map();
const TITLES = {
  overview: 'Overview', signals: 'Signals', fleet: 'GPU Fleet', ops: 'Ops & Control',
};

export function registerView(id, view) { VIEWS.set(id, view); }

export function renderAll() {
  VIEWS.forEach((view, id) => {
    try { view.render(); } catch (err) { console.error('view ' + id + ' failed', err); }
  });
}

export function switchView(id) {
  S.view = id;
  $('#view-title').textContent = TITLES[id] || id;
  document.querySelectorAll('.nav-item').forEach((n) => n.classList.toggle('active', n.dataset.view === id));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + id));
  const view = VIEWS.get(id);
  if (view && view.onEnter) view.onEnter();
  renderAll();
}

export function buildNav() {
  const items = [['overview', 'Overview', '1'], ['signals', 'Signals', '2'],
                 ['fleet', 'GPU Fleet', '3'], ['ops', 'Ops & Control', '4']];
  const rail = $('#nav');
  rail.innerHTML = '';
  items.forEach(([id, label, key]) => {
    const b = el('button', 'nav-item' + (id === S.view ? ' active' : ''));
    b.dataset.view = id;
    b.appendChild(el('span', 'k', key));
    b.appendChild(el('span', null, label));
    b.onclick = () => switchView(id);
    rail.appendChild(b);
  });
  document.addEventListener('keydown', (e) => {
    const next = { '1': 'overview', '2': 'signals', '3': 'fleet', '4': 'ops' }[e.key];
    if (next && !/input|textarea/i.test(document.activeElement.tagName)) switchView(next);
  });
}

/* ---------- SSE ---------- */

export function setConn(ok) {
  const p = $('#conn');
  if (!p) return;
  p.className = 'pill ' + (ok ? 'ok' : 'bad');
  $('#conn-txt').textContent = ok ? 'live' : 'reconnecting';
}

export function openStream() {
  const es = new EventSource('/api/stream');
  es.onmessage = (ev) => {
    try {
      S.snap = JSON.parse(ev.data);
      renderTopbar();
      renderAll();
      setConn(true);
    } catch (_) { /* a malformed frame must not kill the stream */ }
  };
  es.onerror = () => setConn(false);  // EventSource reconnects on its own
}

export function renderTopbar() {
  const snap = S.snap || {};
  $('#ver').textContent = 'v' + (snap.version || 0);
  $('#stamp').textContent = clockText(snap.sampled_at_ms);

  const badge = $('#mode-badge');
  badge.className = 'pill ' + (S.mode === 'observe' ? 'warn' : 'ok');
  $('#mode-badge-txt').innerHTML = S.mode === 'observe'
    ? 'controller <b>OBSERVE</b>' : 'controller <b>ACTIVE</b>';

  const sup = (snap.fleet || {}).supervisor || {};
  const supOk = sup.running && !sup.last_error && !(sup.drift_observations > 0);
  $('#sup-badge').className = 'pill ' + (sup.running == null ? '' : (supOk ? 'ok' : 'warn'));
  $('#sup-badge-txt').innerHTML = 'supervisor <b>' +
    (sup.running == null ? '—' : sup.running ? (supOk ? 'OK' : 'DRIFT') : 'STOPPED') + '</b>';

  const audit = S.audit || {};
  const issues = ((audit.result || {}).issues || []).length;
  $('#audit-badge').className = 'pill ' + (audit.ran_at_ms == null ? '' : (issues ? 'bad' : 'ok'));
  $('#audit-badge-txt').innerHTML = 'audit <b>' +
    (audit.ran_at_ms == null ? '未运行' : issues ? issues + ' issues' : 'healthy') + '</b>';

  const gpuAge = (snap.gpu_truth || {}).age_ms;
  const gpu = $('#gpufresh');
  gpu.className = 'pill ' + (gpuAge == null ? '' : (freshness(gpuAge) === 'stale' ? 'bad' : 'ok'));
  gpu.innerHTML = '<span class="dot"></span>GPU truth <b>' + ageText(gpuAge) + '</b>';
}
