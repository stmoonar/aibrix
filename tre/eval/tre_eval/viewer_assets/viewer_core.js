/* TRE results viewer: DOM-free helpers (time format, hash state, series alignment, ranges).
   Loaded as a classic script in the page (global TRECore) and under node by the eval tests. */
(function (root) {
  'use strict';

  function fmtT(s) {
    if (s == null || !isFinite(s)) return '';
    const neg = s < 0;
    let a = Math.abs(s);
    const m = Math.floor(a / 60);
    const sec = a - m * 60;
    const ss = Number.isInteger(sec) ? String(sec).padStart(2, '0') : sec.toFixed(1).padStart(4, '0');
    return (neg ? '−' : '') + m + ':' + ss;
  }

  function fmtNum(v, digits) {
    if (v == null || !isFinite(v)) return '–';
    const a = Math.abs(v);
    const d = digits != null ? digits : a >= 100 ? 0 : a >= 10 ? 1 : 2;
    return v.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: d });
  }

  // ---- URL hash state: v=1&m=model|all&a=arm1,arm2&p=panel,..&x=lo,hi&log=1&th=dark&k=kind,..
  const HASH_KEYS = ['v', 'm', 'a', 'p', 'x', 'log', 'th', 'k', 'e'];

  function parseHash(h) {
    const out = {};
    String(h || '').replace(/^#/, '').split('&').forEach(function (kv) {
      if (!kv) return;
      const i = kv.indexOf('=');
      const k = i < 0 ? kv : kv.slice(0, i);
      const v = i < 0 ? '' : decodeURIComponent(kv.slice(i + 1));
      if (HASH_KEYS.indexOf(k) < 0) return;
      if (k === 'a' || k === 'p' || k === 'k') out[k] = v ? v.split(',') : [];
      else if (k === 'x') {
        const p = v.split(',').map(Number);
        if (p.length === 2 && p.every(isFinite) && p[1] > p[0]) out.x = p;
      } else if (k === 'v') {
        const n = Number(v);
        if (n === 1 || n === 2 || n === 3) out.v = n;
      } else if (k === 'log') out.log = v !== '0';
      else out[k] = v;
    });
    return out;
  }

  function serializeHash(st) {
    const parts = [];
    const enc = function (s) { return encodeURIComponent(s).replace(/%2C/g, ','); };
    if (st.v) parts.push('v=' + st.v);
    if (st.m) parts.push('m=' + enc(st.m));
    if (st.a) parts.push('a=' + enc(st.a.join(',')));
    if (st.p) parts.push('p=' + enc(st.p.join(',')));
    if (st.x) parts.push('x=' + st.x.map(function (v) { return Math.round(v * 10) / 10; }).join(','));
    if (st.log != null) parts.push('log=' + (st.log ? 1 : 0));
    if (st.th && st.th !== 'auto') parts.push('th=' + st.th);
    if (st.k) parts.push('k=' + enc(st.k.join(',')));
    if (st.e) parts.push('e=' + enc(st.e));
    return '#' + parts.join('&');
  }

  // ---- one x grid for all arms (1 s), arm series padded with null
  function grid(arms) {
    let lo = Infinity, hi = -Infinity;
    arms.forEach(function (a) {
      if (!a.ts || !a.ts.n) return;
      lo = Math.min(lo, a.ts.t0);
      hi = Math.max(hi, a.ts.t0 + a.ts.n - 1);
    });
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const x = new Array(hi - lo + 1);
    for (let i = 0; i < x.length; i++) x[i] = lo + i;
    return { lo: lo, hi: hi, x: x };
  }

  function aligned(arm, model, field, g) {
    const out = new Array(g.x.length).fill(null);
    const f = arm.ts && arm.ts.models && arm.ts.models[model];
    const src = f && f[field];
    if (!src) return out;
    const off = arm.ts.t0 - g.lo;
    for (let i = 0; i < src.length; i++) {
      const j = i + off;
      if (j >= 0 && j < out.length) out[j] = src[i];
    }
    return out;
  }

  function valueAt(arm, model, field, t) {
    const f = arm.ts && arm.ts.models && arm.ts.models[model];
    const src = f && f[field];
    if (!src) return null;
    const i = Math.floor(t) - arm.ts.t0;
    return i >= 0 && i < src.length ? src[i] : null;
  }

  // ---- ranges
  function clampRange(r, lo, hi, minSpan) {
    let a = r[0], b = r[1];
    const span = Math.max(b - a, minSpan || 5);
    if (span >= hi - lo) return [lo, hi];
    if (a < lo) { a = lo; b = lo + span; }
    if (b > hi) { b = hi; a = hi - span; }
    return [a, b];
  }

  function zoomRange(r, factor, center, lo, hi) {
    const c = center != null ? center : (r[0] + r[1]) / 2;
    const a = c - (c - r[0]) * factor;
    const b = c + (r[1] - c) * factor;
    return clampRange([a, b], lo, hi, 5);
  }

  function panRange(r, frac, lo, hi) {
    const d = (r[1] - r[0]) * frac;
    return clampRange([r[0] + d, r[1] + d], lo, hi, 5);
  }

  // nice time ticks (seconds) for the custom canvases; same steps as the uPlot x axis
  const T_INCRS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  function timeTicks(a, b, px, minGap) {
    const span = Math.max(b - a, 1e-9);
    const want = Math.max(1, Math.floor(px / (minGap || 70)));
    let step = T_INCRS[T_INCRS.length - 1];
    for (let i = 0; i < T_INCRS.length; i++) {
      if (span / T_INCRS[i] <= want) { step = T_INCRS[i]; break; }
    }
    const out = [];
    for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(t);
    return out;
  }

  // ---- events
  function eventMatches(e, f) {
    if (f.kinds && f.kinds.indexOf(e.kind) < 0) return false;
    if (f.model && f.model !== 'all' && e.model !== f.model) return false;
    if (f.q) {
      const q = f.q.toLowerCase();
      const hay = [e.kind, e.model, e.binding, e.reason, e.src].join(' ').toLowerCase();
      if (hay.indexOf(q) < 0) return false;
    }
    return true;
  }

  // TP>=2 intervals arrive once per GPU: one bar per (binding, t0)
  function gpuBars(gpus) {
    const seen = {};
    const out = [];
    Object.keys(gpus || {}).forEach(function (g) {
      gpus[g].forEach(function (iv) {
        const key = iv[3] + '@' + iv[0];
        if (seen[key]) { seen[key].gpus.push(g); return; }
        const bar = { t0: iv[0], t1: iv[1], model: iv[2], binding: iv[3], hidden: iv[4] || [], open: !!iv[5], gpus: [g] };
        seen[key] = bar;
        out.push(bar);
      });
    });
    return out;
  }

  const api = { fmtT: fmtT, fmtNum: fmtNum, parseHash: parseHash, serializeHash: serializeHash, grid: grid,
    aligned: aligned, valueAt: valueAt, clampRange: clampRange, zoomRange: zoomRange, panRange: panRange,
    timeTicks: timeTicks, eventMatches: eventMatches, gpuBars: gpuBars };
  root.TRECore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
