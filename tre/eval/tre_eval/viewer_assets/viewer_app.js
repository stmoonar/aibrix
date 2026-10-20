/* TRE results viewer: page code. Data: <script id="data">, helpers: TRECore, charts: uPlot. */
(function () {
  'use strict';
  const C = window.TRECore;
  const D = JSON.parse(document.getElementById('data').textContent);
  const $ = (s, el) => (el || document).querySelector(s);

  function h(tag, attrs) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        const v = attrs[k];
        if (v == null || v === false) continue;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k === 'html') el.innerHTML = v;
        else if (k === 'style') el.style.cssText = v;
        else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? '' : v);
      }
    }
    for (let i = 2; i < arguments.length; i++) {
      const c = arguments[i];
      if (c == null || c === false) continue;
      (Array.isArray(c) ? c : [c]).forEach((x) => el.append(x instanceof Node ? x : document.createTextNode(String(x))));
    }
    return el;
  }
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ------------------------------------------------------------------ constants / state
  const MODELS = D.models.map((m) => m.name);
  const MODEL = Object.fromEntries(D.models.map((m) => [m.name, m]));
  const ARM = Object.fromEntries(D.arms.map((a) => [a.name, a]));
  const G = C.grid(D.arms);
  const FULL = [G.lo, G.hi];
  const PANELS = [
    { id: 'load', title: 'Offered load', def: true },
    { id: 'replicas', title: 'Awake replicas', def: true },
    { id: 'viol', title: 'SLO violations', def: true },
    { id: 'ttft', title: 'TTFT p95', def: true },
    { id: 'tpot', title: 'TPOT p95', def: true },
    { id: 'waiting', title: 'Queue: waiting', def: true },
    { id: 'running', title: 'Queue: running', def: false },
    { id: 'kv', title: 'KV cache', def: false },
    { id: 'z', title: 'Z signal', def: true },
    { id: 'reissues', title: 'Reissued requests', def: false },
  ];
  const KINDS = [
    { id: 'decision_up', label: 'Scale-up decision', glyph: '▲', def: true },
    { id: 'decision_down', label: 'Scale-down decision', glyph: '▼', def: true },
    { id: 'wake', label: 'Wake', glyph: '●', def: true },
    { id: 'sleep', label: 'Sleep', glyph: '○', def: true },
    { id: 'probe', label: 'SafeScale probe', glyph: '▬', def: true },
    { id: 'refusal', label: 'SM refusal', glyph: '✕', def: true },
    { id: 'counterfactual', label: 'TRE would act (observe)', glyph: '◇', def: false },
  ];
  const KIND = Object.fromEntries(KINDS.map((k) => [k.id, k]));
  const S = {
    v: 1, m: MODELS.length > 1 ? 'all' : MODELS[0], arms: D.arms.map((a) => a.name),
    p: PANELS.filter((p) => p.def).map((p) => p.id), x: null, log: true, th: 'auto',
    k: KINDS.filter((k) => k.def).map((k) => k.id), e: null, q: '', inRange: true, cursorT: null,
  };
  (function restore() {
    const st = C.parseHash(location.hash);
    if (st.v) S.v = st.v;
    if (st.m && (st.m === 'all' || MODEL[st.m])) S.m = st.m;
    if (st.a) { const a = st.a.filter((n) => ARM[n]); if (a.length) S.arms = a; }
    if (st.p) S.p = st.p.filter((p) => PANELS.some((q) => q.id === p));
    if (st.x) S.x = C.clampRange(st.x, FULL[0], FULL[1], 5);
    if (st.log != null) S.log = st.log;
    if (st.th === 'light' || st.th === 'dark') S.th = st.th;
    if (st.k) S.k = st.k.filter((k) => KIND[k]);
    if (st.e) S.e = st.e;
  })();

  let CH = [];          // uPlot instances in the current view
  let LANES = [];       // custom canvases in the current view
  let RENDER_AC = null; // aborts the window listeners of the previous render's canvases
  let ACTIVE = null;    // uPlot under the mouse
  let GEO = null;       // plot area of a uPlot (css px): {l, r}
  let MX = 0, MY = 0;
  const AXIS_W = 92;
  const PAD_R = 14;
  const TIP = $('#tip');

  const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  const isDark = () => S.th === 'dark' || (S.th === 'auto' && !!(mq && mq.matches));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const armColor = (a) => (isDark() ? a.color.dark : a.color.light);
  const modelColor = (m) => (MODEL[m] ? (isDark() ? MODEL[m].color.dark : MODEL[m].color.light) : css('--muted'));
  const visArms = () => D.arms.filter((a) => S.arms.indexOf(a.name) >= 0);
  const range = () => S.x || FULL;
  function alpha(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  function luminance(hex) {
    const n = parseInt(hex.slice(1), 16);
    const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    return 0.2126 * f((n >> 16) & 255) + 0.7152 * f((n >> 8) & 255) + 0.0722 * f(n & 255);
  }
  function swatch(a) {
    const cls = 'sw' + (a.dash && a.dash.length ? (a.dash[0] <= 2 ? ' dot' : ' dash') : '');
    return h('span', { class: cls, style: 'color:' + armColor(a), 'aria-hidden': 'true' });
  }
  function modelDot(m) { return h('span', { class: 'dotm', style: 'background:' + modelColor(m), 'aria-hidden': 'true' }); }
  const hatchCache = {};
  function hatch(ctx, color, bg) {
    const key = color + '|' + (bg || '');
    if (hatchCache[key]) return hatchCache[key];
    const c = document.createElement('canvas');
    const px = Math.round(6 * (window.devicePixelRatio || 1));
    c.width = c.height = px;
    const g = c.getContext('2d');
    if (bg) { g.fillStyle = bg; g.fillRect(0, 0, px, px); }
    g.strokeStyle = color;
    g.lineWidth = Math.max(1, px / 5);
    g.beginPath();
    g.moveTo(-1, px + 1); g.lineTo(px + 1, -1);
    g.moveTo(-1, 1); g.lineTo(1, -1);
    g.moveTo(px - 1, px + 1); g.lineTo(px + 1, px - 1);
    g.stroke();
    return (hatchCache[key] = ctx.createPattern(c, 'repeat'));
  }

  // ------------------------------------------------------------------ hash
  let hashTimer = null;
  function writeHash() {
    clearTimeout(hashTimer);
    hashTimer = setTimeout(() => {
      const st = { v: S.v, m: S.m, x: S.x, log: S.log ? null : false, th: S.th, e: S.e };
      if (S.arms.length !== D.arms.length) st.a = S.arms;
      const dp = PANELS.filter((p) => p.def).map((p) => p.id);
      if (S.p.join() !== dp.join()) st.p = S.p;
      const dk = KINDS.filter((k) => k.def).map((k) => k.id);
      if (S.k.join() !== dk.join()) st.k = S.k;
      const s = C.serializeHash(st);
      if (s !== location.hash) history.replaceState(null, '', s === '#' ? location.pathname + location.search : s);
    }, 120);
  }

  // ------------------------------------------------------------------ tooltip
  document.addEventListener('mousemove', (e) => { MX = e.clientX; MY = e.clientY; }, { passive: true, capture: true });
  function showTip(html) {
    TIP.innerHTML = html;
    TIP.hidden = false;
    const w = TIP.offsetWidth, ht = TIP.offsetHeight;
    let x = MX + 16, y = MY + 14;
    if (x + w > window.innerWidth - 8) x = MX - w - 16;
    if (y + ht > window.innerHeight - 8) y = Math.max(8, MY - ht - 14);
    TIP.style.left = Math.max(8, x) + 'px';
    TIP.style.top = y + 'px';
  }
  function hideTip() { TIP.hidden = true; }
  function tipHead(title, t) {
    return '<div class="th"><span>' + esc(title) + '</span><span>' + C.fmtT(t) + ' <span class="x">(' + Math.round(t) + ' s)</span></span></div>';
  }
  function tipRow(color, dash, label, value, extra) {
    const st = 'color:' + color;
    const cls = 'sw' + (dash && dash.length ? (dash[0] <= 2 ? ' dot' : ' dash') : '');
    return '<div class="tr"><span class="' + cls + '" style="' + st + '"></span>' + esc(label) +
      '<span class="v">' + value + '</span>' + (extra ? '<span class="x">' + extra + '</span>' : '') + '</div>';
  }

  // ------------------------------------------------------------------ range
  function setRange(r) {
    const c = C.clampRange(r, FULL[0], FULL[1], 5);
    S.x = c[0] <= FULL[0] && c[1] >= FULL[1] ? null : c;
    applyRange();
  }
  function resetZoom() { S.x = null; applyRange(); }
  function applyRange() {
    const r = range();
    CH.forEach((u) => u.setScale('x', { min: r[0], max: r[1] }));
    LANES.forEach((l) => l.draw());
    drawNav();
    $('#range-label').textContent = C.fmtT(r[0]) + ' – ' + C.fmtT(r[1]) + (S.x ? '  (' + Math.round(r[1] - r[0]) + ' s)' : '  whole run');
    writeHash();
  }
  function zoomAt(factor, center) { setRange(C.zoomRange(range(), factor, center, FULL[0], FULL[1])); }
  function setCursorT(t, from) {
    S.cursorT = t;
    LANES.forEach((l) => l.overlay());
    if (from !== 'plot') {
      CH.forEach((u) => {
        if (t == null) u.setCursor({ left: -10, top: -10 }, false);
        else {
          const x = u.valToPos(t, 'x');
          u.setCursor({ left: x, top: Math.max(1, u.over.clientHeight / 2) }, false);
        }
      });
    }
  }

  // ------------------------------------------------------------------ uPlot panel
  const T_INCRS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  function phasesFor(model) {
    if (!model || !D.onsets.length) return [];
    const ref = D.arms.length ? D.arms[0].name : null;
    return D.onsets.filter((o) => o.model === model && o.arm === ref && o.t_on != null && o.t_off != null);
  }
  function fmtY(v) {
    if (v == null) return '';
    const a = Math.abs(v);
    if (a >= 1e4) return (v / 1e3).toFixed(0) + 'k';
    if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1).replace(/\.0$/, '') + 'k';
    if (a >= 10 || Number.isInteger(v)) return String(Math.round(v * 10) / 10);
    return String(Math.round(v * 100) / 100);
  }

  function plot(el, cfg) {
    // cfg: {series: [{label, color, dash, values, step, width, fill, hidden, spanGaps}], height, showX, yRange, yLog,
    //       refs: [{y, label}], bands, model, tip(idx) -> html, yIncrs}
    const ink2 = css('--ink2'), grid = css('--grid'), muted = css('--muted');
    const width = Math.max(200, el.clientWidth);
    const font = '11px ' + css('--font');
    const series = [{}].concat(cfg.series.map((s) => ({
      label: s.label, stroke: s.hidden ? 'transparent' : s.color, width: s.hidden ? 0 : (s.width || 1.5),
      dash: s.dash && s.dash.length ? s.dash : undefined, fill: s.fill, spanGaps: !!s.spanGaps,
      paths: s.step ? uPlot.paths.stepped({ align: 1 }) : undefined, points: { show: false },
    })));
    const data = [G.x].concat(cfg.series.map((s) => s.values));
    const opts = {
      width: width, height: cfg.height,
      padding: [6, PAD_R, cfg.showX ? 0 : 4, 0],
      legend: { show: false },
      cursor: {
        sync: { key: 'tre' }, y: false, drag: { x: true, y: false, setScale: false },
        points: { size: 6, width: 1.5 }, focus: { prox: -1 },
        bind: { dblclick: () => () => { resetZoom(); return null; } },
      },
      scales: {
        x: { time: false, range: () => range() },
        y: cfg.yLog ? { distr: 3, log: 10, range: cfg.yRange } : { range: cfg.yRange },
      },
      axes: [
        {
          scale: 'x', stroke: ink2, font: font, space: 70, incrs: T_INCRS, size: cfg.showX ? 24 : 2,
          grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1, size: cfg.showX ? 3 : 0 },
          values: (u, sp) => (cfg.showX ? sp.map(C.fmtT) : sp.map(() => '')),
        },
        {
          scale: 'y', stroke: ink2, font: font, size: AXIS_W, space: 22, gap: 4,
          grid: { stroke: grid, width: 1 }, ticks: { show: false }, incrs: cfg.yIncrs,
          splits: cfg.yLog ? (u, ai, mn, mx) => { const o = []; for (let v = Math.pow(10, Math.ceil(Math.log10(Math.max(mn, 1e-9)) - 1e-9)); v <= mx * 1.0001; v *= 10) o.push(v); return o; } : undefined,
          values: (u, sp) => sp.map(fmtY),
        },
      ],
      series: series,
      bands: cfg.bands,
      hooks: {
        drawClear: [(u) => {
          const ph = phasesFor(cfg.model);
          if (!ph.length) return;
          const ctx = u.ctx;
          ctx.save();
          ctx.fillStyle = css('--phase');
          ph.forEach((o) => {
            const a = Math.max(u.bbox.left, u.valToPos(o.t_on, 'x', true));
            const b = Math.min(u.bbox.left + u.bbox.width, u.valToPos(o.t_off, 'x', true));
            if (b > a) ctx.fillRect(a, u.bbox.top, b - a, u.bbox.height);
          });
          ctx.restore();
        }],
        draw: [(u) => {
          if (!cfg.refs || !cfg.refs.length) return;
          const ctx = u.ctx, pr = uPlot.pxRatio;
          ctx.save();
          ctx.font = (10.5 * pr) + 'px ' + css('--font');
          cfg.refs.forEach((r) => {
            if (r.y == null) return;
            const y = Math.round(u.valToPos(r.y, 'y', true)) + 0.5;
            if (y < u.bbox.top || y > u.bbox.top + u.bbox.height) return;
            ctx.strokeStyle = r.color || ink2;
            ctx.lineWidth = pr;
            ctx.setLineDash([4 * pr, 3 * pr]);
            ctx.beginPath(); ctx.moveTo(u.bbox.left, y); ctx.lineTo(u.bbox.left + u.bbox.width, y); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = r.color || muted;
            ctx.textAlign = 'right';
            ctx.textBaseline = r.below ? 'top' : 'bottom';
            ctx.fillText(r.label, u.bbox.left + u.bbox.width - 4 * pr, r.below ? y + 2 * pr : y - 2 * pr);
          });
          ctx.restore();
        }],
        setSelect: [(u) => {
          if (u.select.width > 4) {
            const a = u.posToVal(u.select.left, 'x'), b = u.posToVal(u.select.left + u.select.width, 'x');
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
            setRange([a, b]);
          }
        }],
        setCursor: [(u) => {
          if (ACTIVE !== u) return;
          const i = u.cursor.idx;
          if (i == null) { hideTip(); setCursorT(null, 'plot'); return; }
          const t = u.data[0][i];
          setCursorT(t, 'plot');
          showTip(cfg.tip ? cfg.tip(i, t) : tipHead('', t));
        }],
        ready: [(u) => {
          u.over.addEventListener('mouseenter', () => { ACTIVE = u; });
          u.over.addEventListener('mouseleave', () => { if (ACTIVE === u) { ACTIVE = null; hideTip(); setCursorT(null, 'plot'); } });
          u.over.addEventListener('wheel', wheelZoom(u), { passive: false });
        }],
      },
    };
    const u = new uPlot(opts, data, el);
    if (!GEO) {
      const pr = uPlot.pxRatio;
      GEO = { l: u.bbox.left / pr, r: width - (u.bbox.left + u.bbox.width) / pr };
    }
    CH.push(u);
    return u;
  }
  function wheelZoom(u) {
    return (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const t = u.posToVal(e.offsetX, 'x');
      zoomAt(e.deltaY > 0 ? 1.25 : 0.8, t);
    };
  }
  function autoRange(lo0, pad, minHi) {
    return (u, mn, mx) => {
      if (mn == null || mx == null || !isFinite(mn) || !isFinite(mx)) return [lo0 == null ? 0 : lo0, minHi || 1];
      const lo = lo0 == null ? mn : Math.min(lo0, mn);
      const hi = Math.max(mx, minHi || 0);
      const p = (hi - lo || 1) * (pad == null ? 0.08 : pad);
      return [lo, hi + p];
    };
  }
  function logRange(minHi) {
    return (u, mn, mx) => {
      if (mn == null || !isFinite(mn) || mn <= 0) mn = 1;
      if (mx == null || !isFinite(mx)) mx = 10;
      mx = Math.max(mx, minHi || 0);
      return [Math.pow(10, Math.floor(Math.log10(mn))), Math.pow(10, Math.ceil(Math.log10(mx * 1.05)))];
    };
  }

  // ------------------------------------------------------------------ panels (view 1)
  const hasField = (m, f) => D.arms.some((a) => { const s = a.ts.models[m] && a.ts.models[m][f]; return s && s.some((v) => v != null); });
  function armSeries(m, field, opts) {
    opts = opts || {};
    return visArms().filter(opts.filter || (() => true)).map((a) => {
      let vals = C.aligned(a, m, field, G);
      if (opts.map) vals = vals.map((v) => (v == null ? null : opts.map(v)));
      return { arm: a, label: a.label, color: armColor(a), dash: a.dash, values: vals, step: !!opts.step, spanGaps: !!opts.spanGaps };
    });
  }
  function seriesTip(title, unit, list, fmt) {
    return (i, t) => {
      let s = tipHead(title, t);
      list.forEach((x) => {
        const v = x.values[i];
        s += tipRow(x.color, x.dash, x.label, v == null ? '–' : (fmt ? fmt(v) : C.fmtNum(v)) + (unit ? ' ' + unit : ''));
      });
      return s;
    };
  }
  function smooth(vals, w) {  // centred moving mean over w seconds (display only; nulls skipped)
    const out = new Array(vals.length).fill(null), k = Math.floor(w / 2);
    let s = 0, n = 0;
    for (let i = -k; i < vals.length; i++) {
      const add = i + k, drop = i - k - 1;
      if (add < vals.length && vals[add] != null) { s += vals[add]; n++; }
      if (drop >= 0 && vals[drop] != null) { s -= vals[drop]; n--; }
      if (i >= 0 && vals[i] != null && n) out[i] = s / n;
    }
    return out;
  }
  function bins10(vals) {  // per-second counts -> 10 s sums, drawn as a step over the bin
    const sums = {};
    vals.forEach((v, i) => { if (v != null) { const b = Math.floor(G.x[i] / 10); sums[b] = (sums[b] || 0) + v; } });
    return vals.map((v, i) => { const b = Math.floor(G.x[i] / 10); return b in sums ? sums[b] : null; });
  }

  function buildPanel(id, m, last, height) {
    const mm = MODEL[m] || {};
    const slo = mm.slo || {};
    const hasRho = hasField(m, 'rho');
    const spec = { id: id, unit: '', note: '', empty: null, cfg: null };
    const base = { height: height, showX: last, model: m };
    if (id === 'load') {
      const f = hasRho ? 'rho' : 'offered_rps';
      const ref = visArms().find((a) => a.ts.models[m]) || D.arms[0];
      const vals = C.aligned(ref, m, f, G);
      const sm = smooth(vals, 10);
      const col = modelColor(m);
      const comp = visArms().map((a) => ({ a: a, v: C.aligned(a, m, 'completed_rps', G), off: C.aligned(a, m, 'offered_rps', G) }));
      spec.title = 'Offered load';
      spec.unit = (hasRho ? 'ρ = offered req/s ÷ per-replica capacity (replica equivalents)' : 'offered requests / s') + '; line = 10 s mean';
      spec.cfg = Object.assign(base, {
        series: [{ label: 'offered (1 s)', color: alpha(col, 0.35), values: vals, fill: alpha(col, 0.12), width: 0.75 },
          { label: 'offered (10 s mean)', color: col, values: sm, width: 1.75 }],
        yRange: autoRange(0, 0.1),
        tip: (i, t) => {
          let s = tipHead('Offered load', t);
          s += tipRow(col, null, hasRho ? 'ρ, 10 s mean' : 'offered, 10 s mean', C.fmtNum(sm[i]), hasRho ? '1 s: ' + C.fmtNum(vals[i]) : '');
          if (hasRho && comp[0]) s += tipRow(col, null, 'offered', C.fmtNum(comp[0].off[i]), 'req/s');
          comp.forEach((c) => { s += tipRow(armColor(c.a), c.a.dash, c.a.label + ' completed', C.fmtNum(c.v[i]), 'req/s'); });
          return s;
        },
      });
    } else if (id === 'replicas') {
      const arms = visArms();
      const series = [];
      const bands = [];
      let rho = null;
      if (hasRho) {
        const ref = arms.find((a) => a.ts.models[m]) || D.arms[0];
        rho = smooth(C.aligned(ref, m, 'rho', G), 10);
        series.push({ label: 'demand ρ', color: css('--axis'), values: rho, fill: css('--demand'), width: 0.75 });
      }
      const rows = [];
      const off = (i) => (arms.length > 1 ? (i - (arms.length - 1) / 2) * 0.07 : 0);  // de-overlap equal steps
      arms.forEach((a, ai) => {
        const awake = C.aligned(a, m, 'awake', G), rout = C.aligned(a, m, 'routable', G);
        const sh = (v) => (v == null ? null : v + off(ai));
        const hid = C.aligned(a, m, 'hidden', G), tgt = C.aligned(a, m, 'target', G);
        series.push({ label: a.label + ' routable', color: armColor(a), values: rout.map(sh), step: true, hidden: true });
        const ri = series.length;
        series.push({ label: a.label, color: armColor(a), dash: a.dash, values: awake.map(sh), step: true, width: 1.75 });
        bands.push({ series: [ri + 1, ri], fill: (u) => hatch(u.ctx, alpha(armColor(a), 0.75)) });
        rows.push({ a: a, awake: awake, hid: hid, tgt: tgt });
      });
      let mx = Math.max(1, mm.max_awake || 0);
      rows.forEach((r) => r.awake.forEach((v) => { if (v != null && v > mx) mx = v; }));
      if (rho) rho.forEach((v) => { if (v != null && v > mx) mx = v; });
      spec.title = 'Awake replicas';
      spec.unit = 'replicas; hatched = hidden (awake, not routable)' + (hasRho ? '; grey = demand ρ (10 s mean)' : '') + (arms.length > 1 ? '; arms offset slightly so equal steps stay visible' : '');
      spec.cfg = Object.assign(base, {
        series: series, bands: bands, yIncrs: [1, 2, 4, 5, 10],
        yRange: () => [0, Math.ceil(mx) + 0.6],
        refs: mm.max_awake ? [{ y: mm.max_awake, label: 'max awake ' + mm.max_awake }] : [],
        tip: (i, t) => {
          let s = tipHead('Awake replicas', t);
          if (rho) s += tipRow(css('--axis'), null, 'demand ρ', C.fmtNum(rho[i]));
          rows.forEach((r) => {
            const ex = [];
            if (r.hid[i]) ex.push(r.hid[i] + ' hidden');
            if (r.tgt[i] != null && r.tgt[i] !== r.awake[i]) ex.push('target ' + r.tgt[i] + (r.a.target_source ? ' (' + r.a.target_source + ')' : ''));
            s += tipRow(armColor(r.a), r.a.dash, r.a.label, r.awake[i] == null ? '–' : r.awake[i], ex.join(', '));
          });
          return s;
        },
      });
    } else if (id === 'viol') {
      const list = armSeries(m, 'viol_pct', { step: true });
      spec.title = 'SLO violations';
      spec.unit = '% of scored requests, 10 s bins by send time';
      spec.cfg = Object.assign(base, { series: list, yRange: autoRange(0, 0.08, 5), tip: seriesTip('SLO violations', '%', list) });
    } else if (id === 'ttft' || id === 'tpot') {
      const f = id === 'ttft' ? 'ttft_p95_ms' : 'tpot_p95_ms';
      const list = armSeries(m, f, { step: true });
      const thr = id === 'ttft' ? slo.ttft_p95_ms : slo.tpot_p95_ms;
      spec.title = id === 'ttft' ? 'TTFT p95' : 'TPOT p95';
      spec.unit = 'ms, 10 s bins by send time' + (S.log ? ', log scale' : '');
      if (!list.some((x) => x.values.some((v) => v != null))) spec.empty = id === 'tpot' ? 'No TPOT series (requests.csv missing).' : 'No TTFT series.';
      spec.cfg = Object.assign(base, {
        series: list, yLog: S.log, yRange: S.log ? logRange(thr) : autoRange(0, 0.08, thr ? thr * 1.2 : 0),
        refs: thr ? [{ y: thr, label: (id === 'ttft' ? 'SLO floor ' : 'SLO ') + thr + ' ms', color: css('--critical') }] : [],
        tip: seriesTip(spec.title, 'ms', list, (v) => C.fmtNum(v, 0)),
      });
    } else if (id === 'waiting' || id === 'running') {
      const list = armSeries(m, id);
      spec.title = id === 'waiting' ? 'Queue: waiting' : 'Queue: running';
      spec.unit = 'requests, sum over routable pods (vLLM gauge, 1 s)';
      spec.cfg = Object.assign(base, { series: list, yRange: autoRange(0, 0.08, 1), tip: seriesTip(spec.title, '', list, (v) => C.fmtNum(v, 0)) });
    } else if (id === 'kv') {
      const list = armSeries(m, 'kv_mean', { map: (v) => v * 100 });
      spec.title = 'KV cache';
      spec.unit = '% used, mean over routable pods';
      spec.cfg = Object.assign(base, { series: list, yRange: () => [0, 100], tip: seriesTip('KV cache', '%', list, (v) => C.fmtNum(v, 1)) });
    } else if (id === 'z') {
      const list = armSeries(m, 'z', { filter: (a) => a.family === 'tre', spanGaps: true });
      const tau = mm.tau || {};
      spec.title = 'Z signal';
      spec.unit = 'Z = TSS ÷ θ per 10 s window (TRE arms)';
      if (!list.length) spec.empty = 'No TRE arm shown: Z is drawn for TRE arms only.';
      const refs = [];
      if (tau.crit != null) refs.push({ y: tau.crit, label: 'τ_crit ' + tau.crit, color: css('--critical'), below: true });
      if (tau.high != null) refs.push({ y: tau.high, label: 'τ_high ' + tau.high });
      const zs = [];
      list.forEach((x) => x.values.forEach((v) => { if (v != null) zs.push(v); }));
      zs.sort((p, q) => p - q);
      const p98 = zs.length ? zs[Math.floor(0.98 * (zs.length - 1))] : 1;
      const zmax = Math.max((tau.high || 1) * 1.3, p98 * 1.25);
      if (zs.length && zs[zs.length - 1] > zmax) spec.unit += '; clipped at ' + C.fmtNum(zmax, 1) + ' (max ' + C.fmtNum(zs[zs.length - 1], 1) + ')';
      spec.cfg = Object.assign(base, { series: list, refs: refs, yRange: () => [0, zmax], tip: seriesTip('Z signal', '', list) });
    } else if (id === 'reissues') {
      const list = armSeries(m, 'reissues').map((x) => Object.assign(x, { values: bins10(x.values), step: true }));
      spec.title = 'Reissued requests';
      spec.unit = 'sidecar continuations per 10 s (in-flight requests cut by a sleep)';
      spec.cfg = Object.assign(base, { series: list, yRange: autoRange(0, 0.08, 1), tip: seriesTip('Reissued requests', '/ 10 s', list, (v) => C.fmtNum(v, 0)) });
    }
    return spec;
  }

  function panelEl(spec) {
    const el = h('div', { class: 'panel', 'data-panel': spec.id },
      h('div', { class: 'panel-head' }, h('span', { class: 'panel-title', text: spec.title }), h('span', { class: 'panel-unit', text: spec.unit, title: spec.unit })));
    return el;
  }

  // ------------------------------------------------------------------ custom lanes (event strip, swimlanes, gantt)
  function Lane(parent, o) {
    // o: {height, draw(ctx, L), hit(x, y) -> item, tip(item, t) -> html, click(item), axis}
    const wrap = h('div', { class: 'lane-wrap' });
    const cv = h('canvas');
    const ov = h('canvas', { class: 'ov' });
    wrap.append(cv, ov);
    parent.append(wrap);
    const L = { o: o, cv: cv, ov: ov, items: [], drag: null };
    L.width = () => wrap.clientWidth;
    L.left = () => (GEO ? GEO.l : AXIS_W);
    L.right = () => L.width() - (GEO ? GEO.r : PAD_R);
    L.tx = (t) => { const r = range(); return L.left() + (t - r[0]) / (r[1] - r[0]) * (L.right() - L.left()); };
    L.xt = (x) => { const r = range(); return r[0] + (x - L.left()) / (L.right() - L.left()) * (r[1] - r[0]); };
    L.size = () => {
      const pr = window.devicePixelRatio || 1;
      const w = L.width(), ht = o.height + (o.axis ? 18 : 0);
      [cv, ov].forEach((c) => { c.width = Math.round(w * pr); c.height = Math.round(ht * pr); c.style.width = w + 'px'; c.style.height = ht + 'px'; });
      L.h = ht;
      return pr;
    };
    L.draw = () => {
      const pr = L.size();
      const ctx = cv.getContext('2d');
      ctx.setTransform(pr, 0, 0, pr, 0, 0);
      ctx.clearRect(0, 0, L.width(), L.h);
      const r = range();
      const ticks = C.timeTicks(r[0], r[1], L.right() - L.left(), 70);
      ctx.strokeStyle = css('--grid');
      ctx.lineWidth = 1;
      ctx.beginPath();
      ticks.forEach((t) => { const x = Math.round(L.tx(t)) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, o.height); });
      ctx.stroke();
      L.items = [];
      ctx.save();
      ctx.beginPath(); ctx.rect(L.left(), 0, L.right() - L.left(), L.h); ctx.clip();
      o.draw(ctx, L);
      ctx.restore();
      if (o.gutter) o.gutter(ctx, L);
      if (o.axis) {
        ctx.fillStyle = css('--ink2');
        ctx.font = '11px ' + css('--font');
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ticks.forEach((t) => ctx.fillText(C.fmtT(t), L.tx(t), o.height + 4));
      }
      L.overlay();
    };
    L.overlay = () => {
      const pr = window.devicePixelRatio || 1;
      const ctx = ov.getContext('2d');
      ctx.setTransform(pr, 0, 0, pr, 0, 0);
      ctx.clearRect(0, 0, L.width(), L.h);
      if (L.drag && Math.abs(L.drag.x1 - L.drag.x0) > 2) {
        ctx.fillStyle = css('--brush');
        const a = Math.min(L.drag.x0, L.drag.x1), b = Math.max(L.drag.x0, L.drag.x1);
        ctx.fillRect(a, 0, b - a, o.height);
      }
      if (S.cursorT != null) {
        const x = Math.round(L.tx(S.cursorT)) + 0.5;
        if (x >= L.left() && x <= L.right()) {
          ctx.strokeStyle = css('--ink2');
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, o.height); ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      if (o.overlay) o.overlay(ctx, L);
    };
    L.add = (item, x0, y0, x1, y1) => { if (x1 >= L.left() && x0 <= L.right()) L.items.push({ item: item, x0: x0, y0: y0, x1: x1, y1: y1 }); };
    L.hit = (x, y) => {
      let best = null, bd = 1e9;
      L.items.forEach((it) => {
        const dx = x < it.x0 ? it.x0 - x : x > it.x1 ? x - it.x1 : 0;
        const dy = y < it.y0 ? it.y0 - y : y > it.y1 ? y - it.y1 : 0;
        const d = Math.hypot(dx, dy);
        if (d <= 5 && d < bd) { bd = d; best = it.item; }
      });
      return best;
    };
    const pos = (e) => { const b = cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
    cv.addEventListener('mousedown', (e) => {
      const p = pos(e);
      if (p[0] < L.left() || p[0] > L.right()) return;
      L.drag = { x0: p[0], x1: p[0] };
      e.preventDefault();
    });
    cv.addEventListener('mousemove', (e) => {
      const p = pos(e);
      const inside = p[0] >= L.left() && p[0] <= L.right() && p[1] <= o.height;
      if (L.drag) { L.drag.x1 = Math.max(L.left(), Math.min(L.right(), p[0])); }
      const t = inside ? L.xt(p[0]) : null;
      setCursorT(t, 'lane');
      const it = inside ? L.hit(p[0], p[1]) : null;
      cv.style.cursor = it ? 'pointer' : 'crosshair';
      if (inside && o.tip) {
        const html = o.tip(it, t, p[1]);
        if (html) showTip(html); else hideTip();
      } else hideTip();
    });
    cv.addEventListener('mouseleave', () => { hideTip(); setCursorT(null, 'lane'); if (L.drag) { L.drag = null; L.overlay(); } });
    window.addEventListener('mouseup', (e) => {
      if (!L.drag) return;
      const d = L.drag;
      L.drag = null;
      if (Math.abs(d.x1 - d.x0) > 4) { setRange([L.xt(Math.min(d.x0, d.x1)), L.xt(Math.max(d.x0, d.x1))]); return; }
      L.overlay();
      if (e.target !== cv) return;
      const p = pos(e);
      const it = L.hit(p[0], p[1]);
      if (it && o.click) o.click(it);
    }, { signal: RENDER_AC.signal });
    cv.addEventListener('dblclick', () => resetZoom());
    cv.addEventListener('wheel', (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      zoomAt(e.deltaY > 0 ? 1.25 : 0.8, L.xt(pos(e)[0]));
    }, { passive: false });
    LANES.push(L);
    return L;
  }

  function truncate(ctx, s, w) {
    if (ctx.measureText(s).width <= w) return s;
    while (s.length > 1 && ctx.measureText(s + '…').width > w) s = s.slice(0, -1);
    return s + '…';
  }
  function glyph(ctx, kind, x, y, color, r, sel) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (kind === 'decision_up') { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y + r * 0.8); ctx.lineTo(x - r, y + r * 0.8); ctx.closePath(); ctx.fill(); }
    else if (kind === 'decision_down') { ctx.moveTo(x, y + r); ctx.lineTo(x + r, y - r * 0.8); ctx.lineTo(x - r, y - r * 0.8); ctx.closePath(); ctx.fill(); }
    else if (kind === 'wake') { ctx.arc(x, y, r * 0.85, 0, 7); ctx.fill(); }
    else if (kind === 'sleep') { ctx.arc(x, y, r * 0.8, 0, 7); ctx.fillStyle = css('--surface'); ctx.fill(); ctx.stroke(); }
    else if (kind === 'refusal') { ctx.lineWidth = 2; ctx.moveTo(x - r * 0.8, y - r * 0.8); ctx.lineTo(x + r * 0.8, y + r * 0.8); ctx.moveTo(x + r * 0.8, y - r * 0.8); ctx.lineTo(x - r * 0.8, y + r * 0.8); ctx.stroke(); }
    else if (kind === 'counterfactual') { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); ctx.stroke(); }
    if (sel) { ctx.beginPath(); ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1.5; ctx.arc(x, y, r + 3.5, 0, 7); ctx.stroke(); }
    ctx.restore();
  }
  const selKey = (a, type, id) => a.name + '|' + type + '|' + id;
  function probeEnd(p) { return p.t1 != null ? p.t1 : G.hi; }
  function drawProbe(ctx, L, a, p, y, hgt, color) {
    const x0 = L.tx(p.t0), x1 = L.tx(probeEnd(p));
    const xh = p.t_hide != null ? L.tx(p.t_hide) : x0;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(xh, y); ctx.stroke();
    ctx.fillStyle = hatch(ctx, alpha(color, 0.9));
    ctx.fillRect(xh, y - hgt / 2, Math.max(1, x1 - xh), hgt);
    ctx.strokeRect(xh, y - hgt / 2, Math.max(1, x1 - xh), hgt);
    const r = hgt / 2 + 1;
    if (p.outcome === 'commit') { ctx.fillStyle = color; ctx.fillRect(x1 - r, y - r, 2 * r, 2 * r); }
    else if (p.outcome === 'rollback') {
      ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1.5; ctx.beginPath();
      ctx.moveTo(x1 - r, y - r); ctx.lineTo(x1 + r, y + r); ctx.moveTo(x1 + r, y - r); ctx.lineTo(x1 - r, y + r); ctx.stroke();
    }
    if (S.e === selKey(a, 'probe', p.id)) { ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1.5; ctx.strokeRect(x0 - 3, y - hgt / 2 - 3, x1 - x0 + 6, hgt + 6); }
    ctx.restore();
    L.add({ arm: a, type: 'probe', obj: p }, Math.min(x0, x1), y - hgt / 2 - 2, Math.max(x0, x1), y + hgt / 2 + 2);
  }
  function drawEvent(ctx, L, a, e, y, r) {
    const x = L.tx(e.t);
    const col = modelColor(e.model);
    if (e.kind === 'wake' && e.t_end != null) {
      const x1 = L.tx(e.t_end);
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.restore();
    }
    glyph(ctx, e.kind, x, y, e.kind === 'counterfactual' ? css('--ink2') : col, r, S.e === selKey(a, 'ev', e.id));
    L.add({ arm: a, type: 'ev', obj: e }, x - r, y - r, (e.kind === 'wake' && e.t_end != null ? L.tx(e.t_end) : x) + r, y + r);
  }
  function itemTip(it) {
    if (!it) return null;
    const a = it.arm;
    if (it.type === 'probe') {
      const p = it.obj;
      return tipHead('SafeScale probe · ' + a.label, p.t0) +
        tipRow(modelColor(p.model), null, p.model, esc(p.outcome), esc(p.reason || '')) +
        '<div class="tr">' + C.fmtT(p.t0) + ' → ' + (p.t1 != null ? C.fmtT(p.t1) : 'end') + '<span class="x">' +
        (p.t1 != null ? C.fmtNum(p.t1 - p.t0, 1) + ' s' : '') + '</span></div>';
    }
    if (it.type === 'ev') {
      const e = it.obj;
      const k = KIND[e.kind] || { label: e.kind };
      return tipHead(k.label + ' · ' + a.label, e.t) +
        tipRow(modelColor(e.model), null, e.model || '', esc(e.reason == null ? '' : e.reason), e.binding ? esc(shortBinding(e.binding)) : '') +
        (e.t_end != null ? '<div class="tr">done after<span class="v">' + C.fmtNum(e.t_end - e.t, 2) + ' s</span></div>' : '');
    }
    if (it.type === 'bar') {
      const b = it.obj;
      const hid = b.hidden.reduce((s, x) => s + (x[1] - x[0]), 0);
      return tipHead('Awake binding · ' + a.label, b.t0) +
        tipRow(modelColor(b.model), null, b.model, esc(shortBinding(b.binding))) +
        '<div class="tr">' + C.fmtT(b.t0) + ' → ' + C.fmtT(b.t1) + (b.open ? ' (still awake)' : '') + '<span class="v">' + C.fmtNum(b.t1 - b.t0, 0) + ' s</span></div>' +
        (hid ? '<div class="tr">hidden<span class="v">' + C.fmtNum(hid, 0) + ' s</span><span class="x">' + b.hidden.length + ' span(s)</span></div>' : '');
    }
    if (it.type === 'frag') {
      const f = it.obj;
      return tipHead('Stranded TP ' + f.tp + ' slot · ' + a.label, f.t0) +
        tipRow(css('--serious'), null, 'up to ' + f.free + ' GPUs free', '', 'no aligned free ' + f.tp + '-GPU slot') +
        '<div class="tr">' + C.fmtT(f.t0) + ' → ' + C.fmtT(f.t1) + '<span class="v">' + (f.t1 - f.t0) + ' s</span></div>';
    }
    return null;
  }
  function shortBinding(b) {
    if (!b) return '';
    const p = b.split('/');
    if (p.length !== 3) return b;
    const n = D.nodes.find((x) => x.name === p[1]);
    return (n ? n.short : p[1]) + ' / GPU ' + p[2];
  }
  function eventsOf(a, m, kinds) {
    return a.events.filter((e) => !e.uncertain && kinds.indexOf(e.kind) >= 0 && (m === 'all' || e.model === m));  // 'time uncertain': table only
  }
  function probesOf(a, m) { return S.k.indexOf('probe') >= 0 ? a.probes.filter((p) => m === 'all' || p.model === m) : []; }

  // event strip on top of each timeline column: one row per arm
  function eventStrip(parent, m) {
    const arms = visArms();
    const rowH = 15;
    const panel = h('div', { class: 'panel' },
      h('div', { class: 'panel-head' }, h('span', { class: 'panel-title', text: 'Scale events' }),
        h('span', { class: 'panel-unit', text: KINDS.filter((k) => S.k.indexOf(k.id) >= 0).map((k) => k.glyph + ' ' + k.label.toLowerCase()).join('  ') })));
    parent.append(panel);
    Lane(panel, {
      height: arms.length * rowH + 4,
      draw: (ctx, L) => {
        arms.forEach((a, i) => {
          const y = 2 + i * rowH + rowH / 2;
          probesOf(a, m).forEach((p) => drawProbe(ctx, L, a, p, y, 5, modelColor(p.model)));
          eventsOf(a, m, S.k).forEach((e) => drawEvent(ctx, L, a, e, y, 4));
        });
      },
      gutter: (ctx, L) => {
        ctx.font = '11px ' + css('--font');
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        arms.forEach((a, i) => {
          ctx.fillStyle = css('--ink2');
          ctx.fillText(truncate(ctx, a.label, L.left() - 10), L.left() - 8, 2 + i * rowH + rowH / 2);
        });
      },
      tip: (it) => itemTip(it),
      click: (it) => select(it),
    });
  }

  // ------------------------------------------------------------------ view 1
  function renderTimeline(sec) {
    const cols = S.m === 'all' ? MODELS : [S.m];
    const grid = h('div', { class: 'cols', style: 'grid-template-columns:repeat(' + cols.length + ',minmax(0,1fr))' });
    sec.append(grid);
    const height = cols.length > 1 ? 112 : 132;
    const panels = PANELS.filter((p) => S.p.indexOf(p.id) >= 0);
    cols.forEach((m) => {
      const col = h('div', { class: 'col' });
      grid.append(col);
      const mm = MODEL[m] || {};
      col.append(h('div', { class: 'col-head' }, modelDot(m), m,
        h('span', { class: 'panel-unit', text: (mm.tp > 1 ? 'TP ' + mm.tp + ' \u00B7 ' : '') + (mm.slo && mm.slo.ttft_p95_ms ? 'SLO TTFT ' + mm.slo.ttft_p95_ms + ' ms, TPOT ' + (mm.slo.tpot_p95_ms || '\u2013') + ' ms' : '') })));
      eventStrip(col, m);  // a Lane: drawn by render() once the plots fixed the plot-area geometry
      panels.forEach((p, i) => {
        const spec = buildPanel(p.id, m, i === panels.length - 1, height);
        const el = panelEl(spec);
        col.append(el);
        if (spec.empty) { el.append(h('div', { class: 'panel-empty', text: spec.empty })); return; }
        const box = h('div');
        el.append(box);
        plot(box, spec.cfg);
      });
    });
    if (!panels.length) sec.append(h('p', { class: 'panel-empty', text: 'No panel selected. Open "Panels" to choose what to draw.' }));
  }

  // ------------------------------------------------------------------ view 2
  function renderEvents(sec) {
    const tools = h('div', { class: 'ev-tools' });
    sec.append(tools);
    KINDS.forEach((k) => {
      const on = S.k.indexOf(k.id) >= 0;
      tools.append(h('button', {
        type: 'button', class: 'chip', 'aria-pressed': String(on), title: k.label,
        onclick: () => { S.k = on ? S.k.filter((x) => x !== k.id) : S.k.concat([k.id]); render(); },
      }, h('span', { class: 'mk', text: k.glyph }), k.label));
    });
    const q = h('input', { type: 'search', name: 'q', placeholder: 'Filter: reason, binding, model…', value: S.q, 'aria-label': 'Filter events' });
    tools.append(q);
    const inr = h('label', { class: 'check' }, h('input', { type: 'checkbox', name: 'in-range' }), 'Only the zoomed range');
    inr.firstChild.checked = S.inRange;
    tools.append(inr);
    const cnt = h('span', { class: 'ev-count' });
    tools.append(cnt);

    // context: replicas of the selected model(s)
    const models = S.m === 'all' ? MODELS : [S.m];
    models.forEach((m) => {
      const spec = buildPanel('replicas', m, false, models.length > 1 ? 84 : 110);
      spec.title = 'Awake replicas · ' + m;
      const el = panelEl(spec);
      sec.append(el);
      const box = h('div');
      el.append(box);
      plot(box, spec.cfg);
    });

    const arms = visArms();
    const lanesHost = [];
    arms.forEach((a) => {
      const rows = [
        { id: 'dec', label: 'Decisions', kinds: ['decision_up', 'decision_down'] },
        { id: 'ws', label: 'Wake / sleep', kinds: ['wake', 'sleep'] },
      ];
      if (a.probes.length) rows.push({ id: 'probe', label: 'Probes', kinds: ['probe'] });
      if (a.events.some((e) => e.kind === 'refusal' && !e.uncertain)) rows.push({ id: 'ref', label: 'Refusals', kinds: ['refusal'] });
      if (a.events.some((e) => e.kind === 'counterfactual') && S.k.indexOf('counterfactual') >= 0) rows.push({ id: 'cf', label: 'TRE (observe)', kinds: ['counterfactual'] });
      const n = (k) => eventsOf(a, S.m, [k]).length;
      const stats = [n('decision_up') + ' up', n('decision_down') + ' down', n('wake') + ' wakes', n('sleep') + ' sleeps'];
      if (a.probes.length) {
        const pr = probesOf(a, S.m);
        stats.push(pr.filter((p) => p.outcome === 'commit').length + ' commits / ' + pr.filter((p) => p.outcome === 'rollback').length + ' rollbacks');
      }
      if (rows.some((r) => r.id === 'ref')) stats.push(n('refusal') + ' refusals');
      const el = h('div', { class: 'panel' },
        h('div', { class: 'panel-head' }, swatch(a), h('span', { class: 'panel-title', text: a.label }), h('span', { class: 'panel-unit', text: stats.join(' · ') })));
      sec.append(el);
      lanesHost.push({ a: a, el: el, rows: rows });
    });
    lanesHost.forEach((x) => {
      const rowH = 20;
      Lane(x.el, {
        height: x.rows.length * rowH + 4, axis: true,
        draw: (ctx, L) => {
          x.rows.forEach((r, i) => {
            const y = 2 + i * rowH + rowH / 2;
            ctx.strokeStyle = css('--grid'); ctx.beginPath(); ctx.moveTo(L.left(), Math.round(y) + 0.5); ctx.lineTo(L.right(), Math.round(y) + 0.5); ctx.stroke();
            if (r.id === 'probe') probesOf(x.a, S.m).forEach((p) => drawProbe(ctx, L, x.a, p, y, 7, modelColor(p.model)));
            else eventsOf(x.a, S.m, r.kinds.filter((k) => S.k.indexOf(k) >= 0)).forEach((e) => drawEvent(ctx, L, x.a, e, y, 5));
          });
        },
        gutter: (ctx, L) => {
          ctx.font = '11px ' + css('--font'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = css('--ink2');
          x.rows.forEach((r, i) => ctx.fillText(truncate(ctx, r.label, L.left() - 10), L.left() - 8, 2 + i * rowH + rowH / 2));
        },
        tip: (it) => itemTip(it),
        click: (it) => select(it),
      });
    });

    // table
    const wrap = h('div', { class: 'tbl-wrap' });
    sec.append(wrap);
    const fillTable = () => {
      const r = range();
      const rows = [];
      arms.forEach((a) => {
        a.events.forEach((e) => {
          if (!C.eventMatches(e, { kinds: S.k, model: S.m, q: S.q })) return;
          if (S.inRange && (e.t < r[0] || e.t > r[1])) return;
          rows.push({ a: a, type: 'ev', o: e, t: e.t });
        });
        probesOf(a, S.m).forEach((p) => {
          if (S.q && [p.model, p.reason, p.outcome, p.binding, 'probe'].join(' ').toLowerCase().indexOf(S.q.toLowerCase()) < 0) return;
          if (S.inRange && (probeEnd(p) < r[0] || p.t0 > r[1])) return;
          rows.push({ a: a, type: 'probe', o: p, t: p.t0 });
        });
      });
      rows.sort((x, y) => x.t - y.t);
      cnt.textContent = rows.length + ' event' + (rows.length === 1 ? '' : 's') + (rows.length > 2000 ? ' (first 2000 listed)' : '');
      const tb = h('tbody');
      rows.slice(0, 2000).forEach((x) => {
        const o = x.o;
        const kind = x.type === 'probe' ? 'probe' : o.kind;
        const k = KIND[kind] || { glyph: '?', label: kind };
        const dur = x.type === 'probe' ? (o.t1 != null ? C.fmtNum(o.t1 - o.t0, 1) + ' s' : 'open') : (o.t_end != null ? C.fmtNum(o.t_end - o.t, 2) + ' s' : (o.value != null ? C.fmtNum(o.value) : ''));
        const reason = (o.uncertain ? 'time uncertain · ' : '') + (x.type === 'probe' ? o.outcome + (o.reason && o.reason !== o.outcome ? ': ' + o.reason : '') : (o.reason == null ? '' : o.reason));
        const key = selKey(x.a, x.type, o.id);
        const tr = h('tr', { class: S.e === key ? 'sel' : null, onclick: () => select({ arm: x.a, type: x.type, obj: o }) },
          h('td', o.uncertain ? { text: '~' + C.fmtT(x.t), class: 'warn', title: 'time uncertain: the clock of this log is unknown' } : { text: C.fmtT(x.t) }),
          h('td', null, swatch(x.a), x.a.label),
          h('td', null, o.model ? modelDot(o.model) : null, o.model || ''),
          h('td', null, h('span', { class: 'mk', style: 'color:' + modelColor(o.model), text: k.glyph }), ' ', k.label),
          h('td', { text: reason, title: reason }),
          h('td', { text: shortBinding(o.binding), title: o.binding || '' }),
          h('td', { text: dur }));
        tb.append(tr);
      });
      wrap.innerHTML = '';
      wrap.append(h('table', { class: 'ev' },
        h('thead', null, h('tr', null, ['Time', 'Arm', 'Model', 'Kind', 'Reason / path', 'Binding', 'Duration / value'].map((c) => h('th', { text: c })))), tb));
    };
    fillTable();
    REFILL = fillTable;
    let qt = null;
    q.addEventListener('input', () => { clearTimeout(qt); qt = setTimeout(() => { S.q = q.value; fillTable(); }, 150); });
    inr.firstChild.addEventListener('change', (e) => { S.inRange = e.target.checked; fillTable(); });
  }
  let REFILL = null;

  // ------------------------------------------------------------------ view 3
  function renderGpus(sec) {
    const lg = h('div', { class: 'legend-row' });
    MODELS.forEach((m) => lg.append(h('span', { class: 'lg' }, h('i', { style: 'background:' + modelColor(m) }), m + (MODEL[m].tp > 1 ? ' (TP ' + MODEL[m].tp + ')' : ''))));
    lg.append(h('span', { class: 'lg' }, h('i', { class: 'hatch' }), 'hidden (not routable)'));
    lg.append(h('span', { class: 'lg' }, h('i', { class: 'free' }), 'free GPU'));
    const tps = MODELS.map((m) => MODEL[m].tp).filter((k) => k >= 2);
    if (tps.length) lg.append(h('span', { class: 'lg', title: 'From the GPU layout only, the same rule for every arm. It counts idle GPUs only: a placement blocked by GPUs that are awake (occupied) is not stranded; see the SM 409 marks and the probe outcomes for that (TRE shows 0 s because it never leaves a GPU idle).' }, h('i', { class: 'frag' }),
      'stranded TP slot: ≥ ' + Math.min.apply(null, tps) + ' GPUs free but no aligned free slot'));
    lg.append(h('span', { class: 'lg' }, h('i', { class: 'refusal' }), 'SM 409 refusal'));
    sec.append(lg);

    // total GPUs awake per arm (shares the cursor and zoom)
    const list = visArms().map((a) => {
      const vals = new Array(G.x.length).fill(null);
      MODELS.forEach((m) => C.aligned(a, m, 'gpus_awake', G).forEach((v, i) => { if (v != null) vals[i] = (vals[i] || 0) + v; }));
      return { arm: a, label: a.label, color: armColor(a), dash: a.dash, values: vals, step: true };
    });
    const nG = D.nodes.reduce((s, n) => s + n.gpus, 0);
    const el = panelEl({ id: 'gpus', title: 'GPUs awake', unit: 'all models; ' + nG + ' GPUs in the pool' });
    sec.append(el);
    const box = h('div');
    el.append(box);
    plot(box, { height: 110, showX: false, series: list, yRange: () => [0, Math.max(nG, 1) + 0.5], yIncrs: [1, 2, 4],
      refs: nG ? [{ y: nG, label: 'pool ' + nG }] : [], tip: seriesTip('GPUs awake', '', list, (v) => C.fmtNum(v, 0)) });

    const lanes = [];
    D.nodes.forEach((n) => { for (let i = 0; i < n.gpus; i++) lanes.push({ id: n.name + '/' + i, node: n, i: i }); });
    const laneH = 16, nodeGap = 8, fragH = 8, top = fragH + 6;
    const laneY = {};
    let y = top;
    lanes.forEach((l, k) => {
      if (k && lanes[k - 1].node !== l.node) y += nodeGap;
      laneY[l.id] = y;
      y += laneH + 1;
    });
    const height = y + 2;
    visArms().forEach((a) => {
      const bars = C.gpuBars(a.gpus);
      const s = (a.summary || {}).ALL || {};
      const fragS = a.frag.reduce((x, f) => x + (f.t1 - f.t0), 0);
      const refs = a.events.filter((e) => e.kind === 'refusal' && !e.uncertain);
      const panel = h('div', { class: 'panel' },
        h('div', { class: 'gantt-head' }, swatch(a), h('span', { class: 'panel-title', text: a.label }),
          h('span', { class: 'stat', text: (s.mean_gpus != null ? 'mean ' + C.fmtNum(s.mean_gpus, 2) + ' GPUs' : '') + ' · ' + bars.length + ' awake intervals · stranded TP slot ' + fragS + ' s' + (refs.length ? ' · ' + refs.length + ' SM 409' : '') })));
      sec.append(panel);
      Lane(panel, {
        height: height, axis: true,
        draw: (ctx, L) => {
          // free-lane background
          ctx.strokeStyle = css('--grid');
          lanes.forEach((l) => { ctx.strokeRect(L.left() + 0.5, laneY[l.id] + 0.5, L.right() - L.left() - 1, laneH - 1); });
          a.frag.forEach((f) => {
            const x0 = L.tx(f.t0), x1 = L.tx(f.t1);
            ctx.fillStyle = css('--serious');
            ctx.fillRect(x0, 2, Math.max(2, x1 - x0), fragH);
            ctx.fillStyle = alpha(css('--serious'), 0.12);
            ctx.fillRect(x0, top, Math.max(2, x1 - x0), height - top - 2);
            L.add({ arm: a, type: 'frag', obj: f }, x0, 0, Math.max(x0 + 2, x1), fragH + 4);
          });
          ctx.font = '10.5px ' + css('--font');
          ctx.textBaseline = 'middle';
          bars.forEach((b, bi) => {
            const ys = b.gpus.map((g) => laneY[g]).filter((v) => v != null);
            if (!ys.length) return;
            const x0 = L.tx(b.t0), x1 = L.tx(b.t1);
            if (x1 < L.left() || x0 > L.right()) return;
            ys.sort((p, q) => p - q);
            // contiguous lanes -> one bar; otherwise one bar per lane
            const groups = [];
            ys.forEach((yy) => { const g = groups[groups.length - 1]; if (g && yy - g[1] <= laneH + 1) g[1] = yy; else groups.push([yy, yy]); });
            const col = modelColor(b.model);
            const sel = S.e === selKey(a, 'bar', bi);
            groups.forEach((g) => {
              const y0 = g[0] + 1, hh = g[1] - g[0] + laneH - 2;
              ctx.fillStyle = col;
              ctx.fillRect(x0, y0, Math.max(1, x1 - x0), hh);
              b.hidden.forEach((hs) => {
                const a0 = L.tx(hs[0]), a1 = L.tx(hs[1]);
                ctx.fillStyle = hatch(ctx, alpha(isDark() ? '#1a1a19' : '#ffffff', 0.85));
                ctx.fillRect(a0, y0, Math.max(1.5, a1 - a0), hh);
              });
              ctx.strokeStyle = css('--surface');
              ctx.lineWidth = 1;
              ctx.strokeRect(x0 + 0.5, y0 + 0.5, Math.max(1, x1 - x0) - 1, hh - 1);
              if (sel) { ctx.strokeStyle = css('--ink'); ctx.lineWidth = 2; ctx.strokeRect(x0 - 1, y0 - 1, x1 - x0 + 2, hh + 2); ctx.lineWidth = 1; }
              const vis0 = Math.max(x0, L.left()), vis1 = Math.min(x1, L.right());
              if (vis1 - vis0 > 46) {
                ctx.fillStyle = luminance(col) > 0.3 ? '#1a1a18' : '#ffffff';
                ctx.textAlign = 'left';
                ctx.fillText(truncate(ctx, b.model, vis1 - vis0 - 8), vis0 + 4, y0 + hh / 2 + 0.5);
              }
              L.add({ arm: a, type: 'bar', obj: b, idx: bi }, x0, y0, x1, y0 + hh);
            });
          });
          // SM 409 refusals: a red cross on the lanes of the refused binding (in the strip when it has none)
          ctx.strokeStyle = css('--serious');
          ctx.lineWidth = 1.6;
          refs.forEach((e) => {
            const x = L.tx(e.t);
            if (x < L.left() - 4 || x > L.right() + 4) return;
            const pb = e.binding ? e.binding.split('/') : [];
            let ys = pb.length === 3 ? pb[2].split(',').map((i) => laneY[pb[1] + '/' + i]).filter((v) => v != null) : [];
            ys = ys.length ? ys.map((v) => v + laneH / 2) : [2 + fragH / 2];
            ys.forEach((yc) => {
              ctx.beginPath(); ctx.moveTo(x - 3.5, yc - 3.5); ctx.lineTo(x + 3.5, yc + 3.5); ctx.moveTo(x + 3.5, yc - 3.5); ctx.lineTo(x - 3.5, yc + 3.5); ctx.stroke();
              L.add({ arm: a, type: 'ev', obj: e }, x - 5, yc - 6, x + 5, yc + 6);
            });
          });
          ctx.lineWidth = 1;
        },
        gutter: (ctx, L) => {
          ctx.font = '10.5px ' + css('--font'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
          ctx.fillStyle = css('--muted');
          ctx.fillText('frag.', L.left() - 8, 2 + fragH / 2);
          lanes.forEach((l) => {
            ctx.fillStyle = l.i === 0 ? css('--ink2') : css('--muted');
            ctx.fillText((l.i === 0 ? l.node.short + '  ' : '') + l.i, L.left() - 8, laneY[l.id] + laneH / 2);
          });
          ctx.strokeStyle = css('--axis');
          D.nodes.forEach((n) => n.pairs.forEach((p) => {
            const y0 = laneY[n.name + '/' + p[0]], y1 = laneY[n.name + '/' + p[1]];
            if (y0 == null || y1 == null) return;
            ctx.beginPath(); ctx.moveTo(L.left() - 3.5, y0 + 2); ctx.lineTo(L.left() - 3.5, y1 + laneH - 2); ctx.stroke();
          }));
        },
        tip: (it) => itemTip(it),
        click: (it) => select(it),
      });
    });
    if (!D.nodes.length) sec.append(h('div', { class: 'warnbox', text: 'No GPU topology and no GPU intervals in this report.' }));
  }

  // ------------------------------------------------------------------ detail panel
  function select(it) {
    if (!it) return;
    hideTip();
    if (it.type === 'bar') S.e = selKey(it.arm, 'bar', it.idx != null ? it.idx : 0);
    else if (it.type === 'frag') S.e = null;
    else S.e = selKey(it.arm, it.type, it.obj.id);
    openDetail(it);
    LANES.forEach((l) => l.draw());
    if (REFILL) REFILL();
    writeHash();
  }
  function kv(dl, k, v) { if (v == null || v === '') return; dl.append(h('dt', { text: k }), h('dd', null, v)); }
  function openDetail(it) {
    const a = it.arm, o = it.obj;
    const body = $('#detail-body');
    body.innerHTML = '';
    const dl = h('dl');
    let t0, t1, title, model = o.model;
    if (it.type === 'probe') {
      title = 'SafeScale probe';
      t0 = o.t0; t1 = probeEnd(o);
      kv(dl, 'Outcome', o.outcome);
      kv(dl, 'Reason', o.reason);
      kv(dl, 'Started', C.fmtT(o.t0) + '  (' + o.t0 + ' s)');
      kv(dl, 'Hidden at', o.t_hide != null ? C.fmtT(o.t_hide) : null);
      kv(dl, 'Resolved', o.t1 != null ? C.fmtT(o.t1) + '  (after ' + C.fmtNum(o.t1 - o.t0, 1) + ' s)' : 'still probing at the end of the run');
      kv(dl, 'Binding', o.binding);
    } else if (it.type === 'bar') {
      title = 'Awake binding';
      t0 = o.t0; t1 = o.t1;
      kv(dl, 'Awake', C.fmtT(o.t0) + ' → ' + C.fmtT(o.t1) + (o.open ? ' (still awake at the end)' : ''));
      kv(dl, 'Duration', C.fmtNum(o.t1 - o.t0, 1) + ' s');
      kv(dl, 'Binding', o.binding);
      kv(dl, 'GPUs', o.gpus.join(', '));
      kv(dl, 'Hidden', o.hidden.length ? o.hidden.map((x) => C.fmtT(x[0]) + '–' + C.fmtT(x[1])).join(', ') : 'never');
    } else if (it.type === 'frag') {
      title = 'Stranded TP ' + o.tp + ' slot';
      t0 = o.t0; t1 = o.t1;
      kv(dl, 'Interval', C.fmtT(o.t0) + ' → ' + C.fmtT(o.t1) + '  (' + (o.t1 - o.t0) + ' s)');
      kv(dl, 'Free GPUs', 'up to ' + o.free);
      kv(dl, 'Meaning', 'At least ' + o.tp + ' GPUs had no awake binding, but no aligned ' + o.tp + '-GPU slot was completely free, so a TP ' + o.tp + ' replica could not be placed. Computed from the GPU layout only, the same way for every arm.');
    } else {
      const k = KIND[o.kind] || { label: o.kind };
      title = k.label;
      t0 = o.t; t1 = o.t_end != null ? o.t_end : o.t;
      kv(dl, 'Time', o.uncertain ? 'time uncertain (clock of the log unknown); log stamp ~' + C.fmtT(o.t) : C.fmtT(o.t) + '  (' + o.t + ' s)');
      if (o.t_end != null) kv(dl, 'Done', C.fmtT(o.t_end) + '  (after ' + C.fmtNum(o.t_end - o.t, 2) + ' s)');
      kv(dl, 'Source', o.src);
      kv(dl, o.kind === 'sleep' ? 'Path' : o.kind === 'wake' ? 'Status' : 'Reason', o.reason == null ? null : String(o.reason));
      kv(dl, 'Binding', o.binding);
      if (o.value != null && o.kind !== 'wake') kv(dl, o.kind === 'sleep' ? 'Waited' : 'Value', C.fmtNum(o.value) + (o.kind === 'sleep' ? ' s' : ''));
    }
    $('#detail-title').textContent = title;
    const head = h('dl');
    kv(head, 'Arm', h('span', null, swatch(a), a.label));
    if (model) kv(head, 'Model', h('span', null, modelDot(model), model));
    body.append(head, dl);
    const zoomBtn = h('button', { type: 'button', class: 'btn', text: 'Zoom here (±30 s)', onclick: () => setRange([t0 - 30, t1 + 30]) });
    const otherView = S.v === 3 ? h('button', { type: 'button', class: 'btn', text: 'Show in events', onclick: () => { S.v = 2; render(); } })
      : h('button', { type: 'button', class: 'btn', text: 'Show GPUs', onclick: () => { S.v = 3; render(); } });
    body.append(h('div', { class: 'detail-actions' }, zoomBtn, otherView));
    // nearby events on the same arm (causal chain: decision -> wake -> sleep ...)
    const near = [];
    a.events.forEach((e) => { if (e !== o && !e.uncertain && e.t >= t0 - 15 && e.t <= t1 + 15 && (!model || !e.model || e.model === model || (o.binding && e.binding === o.binding))) near.push({ type: 'ev', o: e, t: e.t }); });
    a.probes.forEach((p) => { if (p !== o && p.t0 <= t1 + 15 && probeEnd(p) >= t0 - 15) near.push({ type: 'probe', o: p, t: p.t0 }); });
    near.sort((x, y) => x.t - y.t);
    if (near.length) {
      body.append(h('h3', { text: 'Same arm, ±15 s' }));
      const ul = h('ul', { class: 'near' });
      near.slice(0, 40).forEach((x) => {
        const kind = x.type === 'probe' ? 'probe' : x.o.kind;
        const k = KIND[kind] || { glyph: '?', label: kind };
        ul.append(h('li', { onclick: () => select({ arm: a, type: x.type, obj: x.o }) },
          C.fmtT(x.t) + '  ', h('span', { class: 'mk', style: 'color:' + modelColor(x.o.model), text: k.glyph }), ' ' + k.label + ' · ' + (x.o.model || '') +
          (x.type === 'probe' ? ' · ' + x.o.outcome : x.o.reason ? ' · ' + x.o.reason : '')));
      });
      body.append(ul);
    }
    if (o.detail && Object.keys(o.detail).length) {
      body.append(h('h3', { text: 'Detail' }), h('pre', { text: JSON.stringify(o.detail, null, 1) }));
    }
    if (o.log) body.append(h('h3', { text: 'Raw log line' }), h('pre', { text: o.log }));
    const d = $('#detail');
    d.classList.add('open');
    d.setAttribute('aria-hidden', 'false');
  }
  function closeDetail() {
    const d = $('#detail');
    d.classList.remove('open');
    d.setAttribute('aria-hidden', 'true');
    if (S.e) { S.e = null; LANES.forEach((l) => l.draw()); if (REFILL) REFILL(); writeHash(); }
  }
  function findSelected() {
    if (!S.e) return null;
    const p = S.e.split('|');
    const a = ARM[p[0]];
    if (!a) return null;
    const id = Number(p[2]);
    if (p[1] === 'ev') { const o = a.events.find((e) => e.id === id); return o ? { arm: a, type: 'ev', obj: o } : null; }
    if (p[1] === 'probe') { const o = a.probes.find((e) => e.id === id); return o ? { arm: a, type: 'probe', obj: o } : null; }
    if (p[1] === 'bar') { const o = C.gpuBars(a.gpus)[id]; return o ? { arm: a, type: 'bar', obj: o, idx: id } : null; }
    return null;
  }

  // ------------------------------------------------------------------ overview navigator
  function drawNav() {
    const cv = $('#nav');
    const pr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, ht = 34;
    cv.width = Math.round(w * pr); cv.height = Math.round(ht * pr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(pr, 0, 0, pr, 0, 0);
    ctx.clearRect(0, 0, w, ht);
    const l = GEO ? GEO.l : AXIS_W, r = w - (GEO ? GEO.r : PAD_R);
    const tx = (t) => l + (t - FULL[0]) / (FULL[1] - FULL[0]) * (r - l);
    NAV.tx = tx; NAV.xt = (x) => FULL[0] + (x - l) / (r - l) * (FULL[1] - FULL[0]); NAV.l = l; NAV.r = r;
    ctx.fillStyle = css('--surface');
    ctx.fillRect(l, 0, r - l, ht - 12);
    // offered load per model, common scale
    const ref = D.arms[0];
    let mx = 0;
    const field = MODELS.some((m) => hasField(m, 'rho')) ? 'rho' : 'offered_rps';
    const ser = MODELS.map((m) => { const v = ref ? smooth(C.aligned(ref, m, field, G), 10) : []; v.forEach((x) => { if (x != null && x > mx) mx = x; }); return [m, v]; });
    ser.forEach(([m, v]) => {
      ctx.strokeStyle = modelColor(m);
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      let pen = false;
      v.forEach((x, i) => {
        if (x == null) { pen = false; return; }
        const X = tx(G.x[i]), Y = (ht - 13) - (x / (mx || 1)) * (ht - 16);
        if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
      });
      ctx.stroke();
    });
    const rr = range();
    ctx.fillStyle = css('--wash');
    ctx.fillRect(l, 0, tx(rr[0]) - l, ht - 12);
    ctx.fillRect(tx(rr[1]), 0, r - tx(rr[1]), ht - 12);
    if (S.x) {
      ctx.strokeStyle = css('--ink');
      ctx.lineWidth = 1.5;
      ctx.strokeRect(tx(rr[0]), 0.75, tx(rr[1]) - tx(rr[0]), ht - 13.5);
    }
    if (NAV.drag) {
      ctx.fillStyle = css('--brush');
      const a = Math.min(NAV.drag[0], NAV.drag[1]), b = Math.max(NAV.drag[0], NAV.drag[1]);
      ctx.fillRect(a, 0, b - a, ht - 12);
    }
    ctx.fillStyle = css('--muted');
    ctx.font = '10.5px ' + css('--font');
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    C.timeTicks(FULL[0], FULL[1], r - l, 80).forEach((t) => ctx.fillText(C.fmtT(t), tx(t), ht - 11));
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(field === 'rho' ? 'load ρ' : 'load', l - 8, (ht - 12) / 2);
  }
  const NAV = { drag: null };
  (function navEvents() {
    const cv = $('#nav');
    const px = (e) => e.clientX - cv.getBoundingClientRect().left;
    cv.addEventListener('mousedown', (e) => { const x = px(e); if (x < NAV.l || x > NAV.r) return; NAV.drag = [x, x]; e.preventDefault(); });
    cv.addEventListener('mousemove', (e) => {
      if (NAV.drag) { NAV.drag[1] = Math.max(NAV.l, Math.min(NAV.r, px(e))); drawNav(); }
    });
    window.addEventListener('mouseup', (e) => {
      if (!NAV.drag) return;
      const d = NAV.drag;
      NAV.drag = null;
      if (Math.abs(d[1] - d[0]) > 4) setRange([NAV.xt(Math.min(d[0], d[1])), NAV.xt(Math.max(d[0], d[1]))]);
      else if (e.target === cv) {
        const r = range(), span = r[1] - r[0], c = NAV.xt(d[0]);
        if (S.x) setRange([c - span / 2, c + span / 2]); else drawNav();
      } else drawNav();
    });
    cv.addEventListener('dblclick', () => resetZoom());
  })();

  // ------------------------------------------------------------------ header
  function renderHeader() {
    $('#trace-name').textContent = D.trace;
    const meta = $('#trace-meta');
    meta.innerHTML = '';
    [D.seed != null ? 'seed ' + D.seed : null, D.arms.length + ' arms', 'load ' + C.fmtT(0) + '–' + C.fmtT(Math.max.apply(null, D.arms.map((a) => a.load_s || 0))),
      'generated ' + D.generated].filter(Boolean).forEach((s) => meta.append(h('span', { text: s })));
    const key = S.m === 'all' ? 'ALL' : S.m;
    const sc = $('#scorecard');
    sc.innerHTML = '';
    const tbl = h('table');
    tbl.append(h('caption', { text: 'Whole-run scores for ' + (key === 'ALL' ? 'all models' : key) + ' (comparison.csv; V_req = share of scored requests that miss the SLO)' }));
    const cols = ['Arm', 'Requests', 'V_req (95% CI)', 'TTFT p95', 'TPOT p95', 'Mean GPUs', 'GPU·s / good req', 'Wakes / sleeps', 'Probes c / r', 'Valid', 'Checks'];
    tbl.append(h('thead', null, h('tr', null, cols.map((c) => h('th', { text: c })))));
    const tb = h('tbody');
    D.arms.forEach((a) => {
      const s = (a.summary || {})[key] || {};
      const inLoad = (t) => t != null && t >= 0 && (a.load_s == null || t <= a.load_s);  // the scored load window only
      const evs = (k) => a.events.filter((e) => e.kind === k && inLoad(e.t) && (key === 'ALL' || e.model === key)).length;
      const pr = a.probes.filter((p) => inLoad(p.t0) && (key === 'ALL' || p.model === key));
      const v = a.validation || {};
      const mism = (v.mismatches || []).length;
      const checked = ((v.score_json || {}).checked || 0) + ((v.compare_json || {}).checked || 0);
      const valid = s.valid == null || s.valid === '' ? '–' : String(s.valid).toLowerCase() === 'true' ? 'valid' : 'invalid';
      const nw = (a.warnings || []).length;
      tb.append(h('tr', null,
        h('td', { class: 'arm' }, swatch(a), a.label, a.label.toLowerCase() !== a.name.toLowerCase() ? h('span', { class: 'sub', text: ' ' + a.name }) : null),
        h('td', { text: C.fmtNum(s.n, 0) }),
        h('td', null, h('b', { text: s.V_req_pct != null ? C.fmtNum(s.V_req_pct, 2) + ' %' : '–' }),
          h('span', { class: 'sub', text: s.V_req_ci_lo != null ? ' ' + C.fmtNum(s.V_req_ci_lo, 1) + '–' + C.fmtNum(s.V_req_ci_hi, 1) : '' })),
        h('td', { text: s.ttft_p95_ms != null ? C.fmtNum(s.ttft_p95_ms, 0) + ' ms' : '–' }),
        h('td', { text: s.tpot_p95_ms != null ? C.fmtNum(s.tpot_p95_ms, 1) + ' ms' : '–' }),
        h('td', { text: C.fmtNum(s.mean_gpus, 2) }),
        h('td', { text: C.fmtNum(s.gpu_s_per_good_req, 3) }),
        h('td', { text: evs('wake') + ' / ' + evs('sleep'), class: 'num-tip',
          title: 'SM wakes / sleeps in [0, load end]\nDecisions (' + (a.decision_source || 'unknown source') + '): ' + evs('decision_up') + ' up / ' + evs('decision_down') + ' down' }),
        h('td', { text: pr.length ? pr.filter((p) => p.outcome === 'commit').length + ' / ' + pr.filter((p) => p.outcome === 'rollback').length : '–', title: 'SafeScale probes started in [0, load end]: commit / rollback' }),
        h('td', { class: valid === 'valid' ? 'ok' : valid === 'invalid' ? 'bad' : '', text: valid }),
        h('td', { class: mism ? 'bad' : nw ? 'warn' : 'ok', title: (a.warnings || []).join('\n'),
          text: (mism ? mism + ' mismatch' + (mism > 1 ? 'es' : '') : checked ? checked + ' ok' : 'not checked') + (nw ? ', ' + nw + ' warn.' : '') })));
    });
    tbl.append(tb);
    sc.append(tbl);

    const side = $('#hdr-side');
    side.innerHTML = '';
    const ref = D.arms[0];
    const ph = D.onsets.filter((o) => ref && o.arm === ref.name && o.t_on != null);
    if (ph.length) {
      const det = h('details', { open: true }, h('summary', { text: 'Hot phases (' + ph.length + '), click to zoom' }));
      const ul = h('ul', { class: 'inline' });
      ph.forEach((o) => ul.append(h('li', null, h('a', { href: '#', onclick: (e) => { e.preventDefault(); setRange([o.t_on - 30, (o.t_off || o.t_on) + 30]); } },
        modelDot(o.model), o.model + '  ' + C.fmtT(o.t_on) + '–' + C.fmtT(o.t_off)))));
      det.append(ul);
      side.append(det);
    }
    if (D.decision_points.length) {
      const det = h('details', null, h('summary', { text: 'Decision points (' + D.decision_points.filter((d) => String(d.pass).toLowerCase() === 'true').length + ' / ' + D.decision_points.length + ' pass)' }));
      const ul = h('ul');
      D.decision_points.forEach((d) => {
        const ok = String(d.pass).toLowerCase() === 'true';
        ul.append(h('li', null, h('span', { class: ok ? 'ok' : 'bad', text: ok ? 'pass ' : 'fail ' }), (ARM[d.arm] ? ARM[d.arm].label : d.arm) + ': ' + d.name + ' (observed ' + d.observed + ', expected ' + d.expected + ')'));
      });
      det.append(ul);
      side.append(det);
    }
    const warns = D.warnings.map((w) => ['export', w]).concat([].concat.apply([], D.arms.map((a) => (a.warnings || []).map((w) => [a.label, w]))));
    if (warns.length) {
      const det = h('details', null, h('summary', { class: 'warn', text: warns.length + ' warning' + (warns.length > 1 ? 's' : '') }));
      const ul = h('ul');
      warns.forEach((w) => ul.append(h('li', null, h('b', { text: w[0] + ': ' }), w[1])));
      det.append(ul);
      side.append(det);
    }
  }

  // ------------------------------------------------------------------ toolbar
  function renderBar() {
    document.querySelectorAll('#views button').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.v) === S.v)));
    const ms = $('#models');
    ms.innerHTML = '';
    (MODELS.length > 1 ? ['all'] : []).concat(MODELS).forEach((m) => {
      ms.append(h('button', { type: 'button', 'aria-pressed': String(S.m === m), onclick: () => { S.m = m; renderHeader(); render(); } },
        m === 'all' ? null : modelDot(m), m === 'all' ? 'All' : m));
    });
    const ar = $('#arms');
    ar.innerHTML = '';
    D.arms.forEach((a) => {
      const on = S.arms.indexOf(a.name) >= 0;
      ar.append(h('button', {
        type: 'button', class: 'chip', 'aria-pressed': String(on), title: 'Click: show / hide. Shift-click: only this arm.',
        onclick: (e) => {
          if (e.shiftKey) S.arms = [a.name];
          else if (on) S.arms = S.arms.length > 1 ? S.arms.filter((x) => x !== a.name) : S.arms;
          else S.arms = D.arms.map((x) => x.name).filter((x) => x === a.name || S.arms.indexOf(x) >= 0);
          render();
        },
      }, swatch(a), a.label));
    });
    const pn = $('#panels');
    pn.innerHTML = '';
    PANELS.forEach((p) => {
      const lab = h('label', { class: 'check' }, h('input', { type: 'checkbox', name: 'panel-' + p.id }), p.title);
      lab.firstChild.checked = S.p.indexOf(p.id) >= 0;
      lab.firstChild.addEventListener('change', (e) => {
        S.p = PANELS.map((q) => q.id).filter((id) => (id === p.id ? e.target.checked : S.p.indexOf(id) >= 0));
        render();
      });
      pn.append(lab);
    });
    $('#panels-ctl').style.display = S.v === 1 ? '' : 'none';
    $('#chk-log').checked = S.log;
    $('#btn-theme').textContent = 'Theme: ' + S.th;
  }

  // ------------------------------------------------------------------ render
  function render() {
    hideTip();
    CH.forEach((u) => u.destroy());
    CH = []; LANES = []; ACTIVE = null; GEO = null; REFILL = null;
    if (RENDER_AC) RENDER_AC.abort();
    RENDER_AC = new AbortController();
    renderBar();
    [1, 2, 3].forEach((v) => { const s = $('#view-' + v); s.innerHTML = ''; s.classList.toggle('on', v === S.v); });
    const sec = $('#view-' + S.v);
    if (S.v === 1) renderTimeline(sec);
    else if (S.v === 2) renderEvents(sec);
    else renderGpus(sec);
    LANES.forEach((l) => l.draw());
    applyRange();
  }

  function setTheme(th) {
    S.th = th;
    if (th === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', th);
    for (const k in hatchCache) delete hatchCache[k];
    renderHeader();
    render();
  }

  // ------------------------------------------------------------------ wiring
  document.querySelectorAll('#views button').forEach((b) => b.addEventListener('click', () => { S.v = Number(b.dataset.v); render(); }));
  $('#btn-reset').addEventListener('click', resetZoom);
  $('#chk-log').addEventListener('change', (e) => { S.log = e.target.checked; render(); });
  $('#btn-theme').addEventListener('click', () => setTheme(S.th === 'auto' ? (isDark() ? 'light' : 'dark') : S.th === 'dark' ? 'light' : 'auto'));
  $('#btn-help').addEventListener('click', (e) => { const hp = $('#help'); hp.hidden = !hp.hidden; e.currentTarget.setAttribute('aria-expanded', String(!hp.hidden)); });
  $('#detail-close').addEventListener('click', closeDetail);
  document.addEventListener('click', (e) => { const m = $('#panels-menu'); if (m.open && !m.contains(e.target)) m.open = false; });
  if (mq && mq.addEventListener) mq.addEventListener('change', () => { if (S.th === 'auto') setTheme('auto'); });
  document.addEventListener('keydown', (e) => {
    const tg = e.target;
    if (tg && (tg.tagName === 'INPUT' || tg.tagName === 'TEXTAREA' || tg.isContentEditable)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (k === 'ArrowLeft' || k === 'ArrowRight') { setRange(C.panRange(range(), k === 'ArrowLeft' ? -0.2 : 0.2, FULL[0], FULL[1])); e.preventDefault(); }
    else if (k === '+' || k === '=') zoomAt(1 / 1.5, S.cursorT);
    else if (k === '-' || k === '_') zoomAt(1.5, S.cursorT);
    else if (k === 'r' || k === 'R') resetZoom();
    else if (k === '1' || k === '2' || k === '3') { S.v = Number(k); render(); }
    else if (k === 'l' || k === 'L') { S.log = !S.log; render(); }
    else if (k === 't' || k === 'T') $('#btn-theme').click();
    else if (k === 'Escape') { closeDetail(); $('#help').hidden = true; }
    else if (k === '?') $('#btn-help').click();
  });
  let rt = null;
  let lastW = 0;
  new ResizeObserver(() => {
    const w = $('#main').clientWidth;
    if (w === lastW) return;
    lastW = w;
    clearTimeout(rt);
    rt = setTimeout(render, 150);
  }).observe($('#main'));

  if (S.th !== 'auto') document.documentElement.setAttribute('data-theme', S.th);
  renderHeader();
  render();
  lastW = $('#main').clientWidth;
  const pre = findSelected();
  if (pre) openDetail(pre);
  window.TREViewer = { state: S, data: D, setRange: setRange, render: render, charts: () => CH, lanes: () => LANES };
})();
