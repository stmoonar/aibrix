/* TRE Console -- framework-free SPA.
   Live data arrives ONLY via /api/stream (SSE, 0.5s, delta-on-version). /api/meta is fetched
   once for static params. No polling of upstream: the in-pod sampler owns all reads, so the
   browser adds zero control-loop overhead no matter how many tabs are open. */
'use strict';

const MODEL_HUES = ['var(--m1)', 'var(--m2)', 'var(--m3)', 'var(--m4)'];
const HIST_CAP = 600;

const S = {
  meta: null,
  colors: {},          // model -> css color
  hist: {},            // model -> [{t, z}]
  seenHistTs: {},      // model -> Set of ts already ingested
  snap: null,
  mode: 'active',
  view: 'live',
};

/* ---------- helpers ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, txt) => { const n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const fmt = (v, d = 3) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const fmtInt = (v) => (v == null) ? '—' : String(v);
const ageText = (ms) => ms == null ? '—' : (ms < 1500 ? 'now' : ms < 60000 ? Math.round(ms / 1000) + 's' : Math.round(ms / 60000) + 'm') + ' ago';
const clockText = (ms) => { if (!ms) return '—'; const d = new Date(ms); return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0') + ':' + String(d.getSeconds()).padStart(2, '0'); };

function toast(msg, isErr) {
  let t = $('#toast'); if (!t) { t = el('div'); t.id = 'toast'; t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = msg; t.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(t._h); t._h = setTimeout(() => { t.className = 'toast'; }, 3200);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { const b = await r.text().catch(() => ''); throw new Error(`${r.status} ${b.slice(0, 160)}`); }
  return r.status === 204 ? {} : r.json();
}

/* ---------- init ---------- */
async function init() {
  buildNav();
  try {
    S.meta = await api('/api/meta');
  } catch (e) { toast('meta load failed: ' + e.message, true); S.meta = { models: [], topology: { nodes: [] } }; }
  S.meta.models.forEach((m, i) => { S.colors[m.name] = MODEL_HUES[i % MODEL_HUES.length]; S.hist[m.name] = []; S.seenHistTs[m.name] = new Set(); });
  try { S.mode = (await api('/api/ops/controller/mode')).mode; } catch (_) {}
  renderControl();
  openStream();
}

function buildNav() {
  const items = [['live', 'Live Signals', '1'], ['fleet', 'GPU Fleet', '2'], ['control', 'Control & Params', '3']];
  const rail = $('#nav');
  items.forEach(([id, label, key]) => {
    const b = el('button', 'nav-item' + (id === S.view ? ' active' : ''));
    b.dataset.view = id;
    b.appendChild(el('span', 'k', key));
    b.appendChild(el('span', null, label));
    b.onclick = () => switchView(id);
    rail.appendChild(b);
  });
  document.addEventListener('keydown', (e) => { const n = { '1': 'live', '2': 'fleet', '3': 'control' }[e.key]; if (n && !/input|textarea/i.test(document.activeElement.tagName)) switchView(n); });
}

const VIEW_TITLES = { live: 'Live Signals', fleet: 'GPU Fleet', control: 'Control & Params' };
function switchView(id) {
  S.view = id;
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === id));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + id));
  $('#view-title').textContent = VIEW_TITLES[id] || 'TRE Console';
  if (S.snap) render();
}

/* ---------- SSE ---------- */
function openStream() {
  const es = new EventSource('/api/stream');
  es.onmessage = (ev) => { try { S.snap = JSON.parse(ev.data); ingest(S.snap); render(); setConn(true); } catch (_) {} };
  es.onerror = () => { setConn(false); };  // EventSource auto-reconnects
}
function setConn(ok) { const p = $('#conn'); if (p) { p.className = 'pill ' + (ok ? 'ok' : 'bad'); $('#conn-dot').className = 'dot'; $('#conn-txt').textContent = ok ? 'live' : 'reconnecting'; } }

/* ---------- history accumulation (works with or without server-side decision hist) ---------- */
function ingest(snap) {
  const models = snap.models || {};
  for (const name of Object.keys(S.hist)) {
    const m = models[name]; if (!m) continue;
    const seen = S.seenHistTs[name];
    // backfill from server decision-hist tail (authoritative when present)
    (m.hist_tail || []).forEach(p => {
      const t = p.window_end_ms || p.ts; if (t == null || seen.has(t)) return;
      seen.add(t); S.hist[name].push({ t, z: p.z_m != null ? p.z_m : p.trs_z_m });
    });
    // else accumulate from the live decision so the line still moves under load
    const st = m.state || {};
    const dt = (snap.decision && snap.decision.latest && snap.decision.latest.ts_ms) || snap.sampled_at_ms;
    if (st.z_m != null && dt != null && !seen.has(dt)) { seen.add(dt); S.hist[name].push({ t: dt, z: st.z_m }); }
    S.hist[name].sort((a, b) => a.t - b.t);
    if (S.hist[name].length > HIST_CAP) S.hist[name] = S.hist[name].slice(-HIST_CAP);
  }
}

/* ---------- render root ---------- */
function render() {
  renderTopbar();
  if (S.view === 'live') renderLive();
  else if (S.view === 'fleet') renderFleet();
  else if (S.view === 'control') renderControlLive();
}

function renderTopbar() {
  const snap = S.snap, dec = (snap.decision && snap.decision.latest) || {};
  const dage = snap.decision && snap.decision.age_ms;
  const loopPill = $('#loop');
  loopPill.className = 'pill ' + (dage == null ? '' : dage < 15000 ? 'ok' : dage < 45000 ? 'warn' : 'bad');
  $('#loop-txt').innerHTML = `loop <b>${dec.loop || '—'}</b> · ${ageText(dage)}`;
  $('#stamp').textContent = 'decision @ ' + clockText(dec.ts_ms);
  const gage = snap.gpu_truth && snap.gpu_truth.age_ms;
  $('#gpufresh').innerHTML = `GPU truth <b>${ageText(gage)}</b>`;
  $('#ver').textContent = 'v' + (snap.version || 0);
  syncModeButtons();
}

/* ---------- LIVE ---------- */
function renderLive() {
  const grid = $('#live-grid'); grid.innerHTML = '';
  const models = (S.meta.models || []);
  if (!models.length) { grid.appendChild(emptyState('No models in registry.')); }
  for (const meta of models) {
    const name = meta.name;
    const st = ((S.snap.models[name] || {}).state) || {};
    grid.appendChild(modelCard(meta, st));
  }
  renderFeed();
}

function stateName(st) {
  const s = st.state; if (s) return String(s).toUpperCase();
  // no band from controller (reduced schema): mark idle vs active by traffic
  if (st.y_m != null && st.y_m <= 1e-9) return 'IDLE';
  return '—';
}

function modelCard(meta, st) {
  const name = meta.name, color = S.colors[name];
  const card = el('div', 'mcard'); card.style.setProperty('--accent', color);
  const head = el('header');
  const nm = el('div', 'name'); nm.innerHTML = `<b>●</b> ${name}`;
  head.appendChild(nm);
  const sn = stateName(st);
  head.appendChild(el('span', 'state-tag state-' + sn, sn));
  card.appendChild(head);

  const zrow = el('div', 'zrow');
  const zb = el('div', 'zbig'); zb.innerHTML = `${fmt(st.z_m, 3)}<small>Zₘ</small>`;
  zrow.appendChild(zb);
  card.appendChild(zrow);

  card.appendChild(sparkline(name, meta));

  const m = el('div', 'metrics');
  const rows = [
    ['TRS', fmt(st.trs, 3)], ['θₘ', fmt(meta.trs.theta_m, 3)], ['ηₘ', fmt(st.eta_m, 2)],
    ['Q_ctl', fmt(st.q_ctl, 2)], ['Yₘ', fmt(st.y_m, 1)], ['routable', fmtInt(st.routable_pods)],
  ];
  rows.forEach(([k, v]) => { const d = el('div'); d.appendChild(el('span', null, k)); d.appendChild(el('span', 'num', v)); m.appendChild(d); });
  card.appendChild(m);

  const chips = el('div', 'chips');
  const awake = smAwake(name), bound = smBound(name);
  chips.appendChild(el('span', 'chip', `awake ${awake}/${bound}`));
  if (st.signal_warm === true) chips.appendChild(el('span', 'chip warm', 'warm'));
  else if (st.signal_warm === false) chips.appendChild(el('span', 'chip cold', 'warming'));
  if (st.is_saturated) chips.appendChild(el('span', 'chip', 'saturated'));
  if (st.signal_unavailable_reason) chips.appendChild(el('span', 'chip cold', st.signal_unavailable_reason));
  card.appendChild(chips);
  return card;
}

function smAwake(name) { const s = ((S.snap.sm || {}).state || {}).models || {}; return (s[name] || {}).awake != null ? s[name].awake : '—'; }
function smBound(name) { const s = ((S.snap.sm || {}).state || {}).models || {}; return (s[name] || {}).bound != null ? s[name].bound : '—'; }

function sparkline(name, meta) {
  const W = 300, H = 62, pad = 4;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'spark'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('preserveAspectRatio', 'none');
  const pts = S.hist[name] || [];
  const taus = [meta.trs.tau_crit, meta.trs.tau_low, meta.trs.tau_high].filter(v => v != null);
  const zs = pts.map(p => p.z).filter(v => v != null);
  const hi = Math.max(0.001, ...zs, ...taus) * 1.1;
  const lo = 0;
  const x = (i) => pad + (pts.length <= 1 ? 0 : (i / (pts.length - 1)) * (W - 2 * pad));
  const y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
  // tau guide lines
  const bandColors = ['var(--crit)', 'var(--low)', 'var(--high)'];
  taus.forEach((tv, i) => {
    const ln = document.createElementNS(svg.namespaceURI, 'line');
    ln.setAttribute('x1', 0); ln.setAttribute('x2', W); ln.setAttribute('y1', y(tv)); ln.setAttribute('y2', y(tv));
    ln.setAttribute('stroke', bandColors[i]); ln.setAttribute('stroke-width', '1'); ln.setAttribute('stroke-dasharray', '3 3'); ln.setAttribute('opacity', '.35');
    svg.appendChild(ln);
  });
  if (pts.length > 1) {
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(p.z == null ? lo : p.z).toFixed(1)}`).join(' ');
    const path = document.createElementNS(svg.namespaceURI, 'path');
    path.setAttribute('d', d); path.setAttribute('fill', 'none'); path.setAttribute('stroke', S.colors[name]); path.setAttribute('stroke-width', '2'); path.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(path);
    const last = pts[pts.length - 1];
    const dot = document.createElementNS(svg.namespaceURI, 'circle');
    dot.setAttribute('cx', x(pts.length - 1)); dot.setAttribute('cy', y(last.z == null ? lo : last.z)); dot.setAttribute('r', '2.5'); dot.setAttribute('fill', S.colors[name]);
    svg.appendChild(dot);
  } else {
    const t = document.createElementNS(svg.namespaceURI, 'text');
    t.setAttribute('x', W / 2); t.setAttribute('y', H / 2); t.setAttribute('text-anchor', 'middle'); t.setAttribute('fill', 'var(--muted)'); t.setAttribute('font-size', '11');
    t.textContent = 'awaiting samples'; svg.appendChild(t);
  }
  return svg;
}

function renderFeed() {
  const feed = $('#feed'); feed.innerHTML = '';
  const evs = (S.snap.events_head || []);
  if (!evs.length) { feed.appendChild(emptyState('No recent scale/leak/safescale events.')); return; }
  evs.forEach(e => {
    const row = el('div', 'ev');
    row.appendChild(el('div', 't', clockText(e.ts_ms)));
    row.appendChild(el('div', 'kind kind-' + (e.kind || 'event'), e.kind || 'event'));
    row.appendChild(el('div', null, e.text || ''));
    feed.appendChild(row);
  });
}

/* ---------- FLEET ---------- */
function renderFleet() {
  const root = $('#fleet-root'); root.innerHTML = '';
  const truth = (S.snap.gpu_truth && S.snap.gpu_truth.nodes) || [];
  const bindings = ((S.snap.sm || {}).state || {}).bindings || [];
  const thr = (S.meta.thresholds || {});
  if (!truth.length) { root.appendChild(emptyState('No GPU-truth yet (agents write tre:gpu_truth:<node> every ~5s).')); return; }

  for (const node of truth) {
    const block = el('div', 'node-block');
    const head = el('div', 'node-head');
    head.appendChild(el('div', 'h', node.node));
    const nbinds = bindings.filter(b => b.node === node.node);
    head.appendChild(el('span', 'sub', `${node.gpus.length} GPUs · ${nbinds.length} bindings`));
    block.appendChild(head);

    const row = el('div', 'gpu-row');
    node.gpus.forEach((g, idx) => {
      const resident = nbinds.filter(b => (b.gpu_ids || []).includes(idx));
      row.appendChild(gpuCell(node, g, idx, resident, thr));
    });
    block.appendChild(row);
    root.appendChild(block);
  }
}

function gpuCell(node, g, idx, resident, thr) {
  const cell = el('div', 'gpu');
  const usedFrac = g.total_mib ? g.used_mib / g.total_mib : 0;
  const awakeCount = resident.filter(b => b.awake).length;
  const leak = resident.length === 0 && g.used_mib > (thr.sleep_leak_used_mib || 8192);
  if (awakeCount > 1 || leak) cell.classList.add('alarm');

  const gh = el('div', 'gh');
  gh.appendChild(el('span', 'idx', 'GPU ' + idx));
  gh.appendChild(el('span', 'uuid', (g.uuid || '').replace('GPU-', '').slice(0, 8)));
  cell.appendChild(gh);

  const bar = el('div', 'membar' + (usedFrac > 0.9 ? ' hot' : '')); const fill = el('i'); fill.style.width = Math.min(100, usedFrac * 100).toFixed(1) + '%'; bar.appendChild(fill); cell.appendChild(bar);
  cell.appendChild(el('div', 'memtext', `${(g.used_mib / 1024).toFixed(1)} / ${(g.total_mib / 1024).toFixed(0)} GiB`));

  const res = el('div', 'resident');
  if (!resident.length) {
    res.appendChild(el('div', 'empty', leak ? '' : 'free'));
    if (leak) res.appendChild(el('div', 'leakflag', `⚠ residual ${(g.used_mib / 1024).toFixed(1)} GiB, no binding`));
  } else {
    resident.forEach(b => {
      const line = el('div', 'bind');
      const sw = el('span', 'swatch'); sw.style.background = S.colors[b.model] || 'var(--idle)'; line.appendChild(sw);
      const nm = el('span', 'mname', b.model); if ((b.gpu_ids || []).length > 1) nm.textContent += ` (tp${b.gpu_ids.length})`; line.appendChild(nm);
      const stt = b.hidden ? 'hidden' : (b.awake ? 'awake' : 'asleep');
      line.appendChild(el('span', 'st ' + stt, stt));
      res.appendChild(line);
    });
    if (awakeCount > 1) res.appendChild(el('div', 'leakflag', `⚠ ${awakeCount} awake on one GPU`));
  }
  cell.appendChild(res);
  return cell;
}

/* ---------- CONTROL ---------- */
function renderControl() {
  const grid = $('#ctl-grid'); grid.innerHTML = '';
  (S.meta.models || []).forEach(meta => grid.appendChild(controlPanel(meta)));
}

function controlPanel(meta) {
  const name = meta.name, color = S.colors[name];
  const p = el('div', 'panel'); p.style.setProperty('--accent', color); p.dataset.model = name;
  p.appendChild(el('h3', null, name));
  p.appendChild(el('div', 'accent-bar'));

  const liveRow = el('div', 'row');
  liveRow.innerHTML = `<span class="muted">awake / bound</span><span class="num" data-fld="awb">—</span>`;
  p.appendChild(liveRow);

  const tRow = el('div', 'row');
  tRow.appendChild(Object.assign(el('span', 'muted'), { textContent: 'wake target' }));
  const stepper = el('div', 'stepper');
  const dec = el('button', null, '−'), val = el('span', 'val num', '0'), inc = el('button', null, '+');
  let target = 0; const setT = (v) => { target = Math.max(meta.min_replicas, Math.min(meta.max_replicas, v)); val.textContent = target; };
  dec.onclick = () => setT(target - 1); inc.onclick = () => setT(target + 1);
  stepper.append(dec, val, inc); tRow.appendChild(stepper);
  p.appendChild(tRow);
  p.dataset.max = meta.max_replicas;

  const actions = el('div', 'actions');
  const applyBtn = el('button', 'btn primary', 'Set target');
  applyBtn.onclick = () => confirmOp(`Set ${name} wake target to ${target}?`, () => api(`/api/ops/models/${name}/target`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ wake_replicas: target }) }), `${name} target → ${target}`);
  actions.appendChild(applyBtn);
  p.appendChild(actions);

  // static params table
  const tbl = el('table', 'params');
  const P = meta.trs, SLO = meta.slo;
  const params = [
    ['θₘ (theta)', fmt(P.theta_m, 3)], ['τ crit / low / high', `${fmt(P.tau_crit, 2)} / ${fmt(P.tau_low, 2)} / ${fmt(P.tau_high, 2)}`],
    ['q_sat', fmt(P.qsat, 2)], ['EMA τ (ms)', fmtInt(P.ema_tau_ms)],
    ['SLO ttft/tpot/e2e p95', `${fmtInt(SLO.ttft_p95_ms)} / ${fmtInt(SLO.tpot_p95_ms)} / ${fmtInt(SLO.e2e_p95_ms)}`],
    ['replicas min / max', `${fmtInt(meta.min_replicas)} / ${fmtInt(meta.max_replicas)}`],
  ];
  params.forEach(([k, v]) => { const tr = el('tr'); tr.appendChild(el('td', null, k)); tr.appendChild(el('td', null, v)); tbl.appendChild(tr); });
  p.appendChild(tbl);
  p.appendChild(Object.assign(el('div', 'sub'), { textContent: 'Parameter editing (restart-to-apply) ships in P0-4.', style: 'margin:10px 0 0;font-size:12px' }));
  return p;
}

function renderControlLive() {
  document.querySelectorAll('#ctl-grid .panel').forEach(p => {
    const name = p.dataset.model;
    const awb = p.querySelector('[data-fld="awb"]');
    if (awb) awb.textContent = `${smAwake(name)} / ${smBound(name)}`;
  });
}

/* ---------- controller mode + fleet ops ---------- */
function syncModeButtons() {
  const a = $('#mode-active'), o = $('#mode-observe');
  if (!a) return;
  a.className = S.mode === 'active' ? 'on-active' : '';
  o.className = S.mode === 'observe' ? 'on-observe' : '';
  const badge = $('#mode-badge');
  badge.className = 'pill ' + (S.mode === 'observe' ? 'warn' : 'ok');
  $('#mode-badge-txt').innerHTML = S.mode === 'observe' ? 'controller <b>OBSERVE</b>' : 'controller <b>ACTIVE</b>';
}

async function setMode(mode) {
  if (mode === S.mode) return;
  const msg = mode === 'observe' ? 'Pause controller actions? It will keep computing decisions but stop scaling/hiding.' : 'Resume controller actions?';
  confirmOp(msg, async () => { const r = await api('/api/ops/controller/mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }) }); S.mode = r.mode; syncModeButtons(); return r; }, `controller → ${mode}`);
}

function reconcile() { confirmOp('Run a reconcile now?', () => api('/api/ops/reconcile', { method: 'POST' }), 'reconcile requested'); }
function defrag() { confirmOp('Run defrag (tp_size=2)? This may migrate sleeping replicas.', () => api('/api/ops/defrag', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tp_size: 2 }) }), 'defrag requested'); }

async function confirmOp(question, fn, okMsg) {
  if (!window.confirm(question)) return;
  try { await fn(); toast(okMsg || 'done'); } catch (e) { toast('failed: ' + e.message, true); }
}

function emptyState(txt) { return Object.assign(el('div', 'empty-state'), { textContent: txt }); }

window.TRE = { setMode, reconcile, defrag };
document.addEventListener('DOMContentLoaded', init);
