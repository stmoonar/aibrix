/* Full-width linked time-series chart, hand-drawn as inline SVG.

   No chart library: the UI pod has no route to any CDN, and the shapes we need
   (threshold bands, replica step lines, action markers, a cursor shared across
   charts) are a few dozen lines of path maths. */
'use strict';

const NS = 'http://www.w3.org/2000/svg';

const H = 340;          // total svg height
const PAD_L = 58;
const PAD_R = 58;
const TOP = 14;
const MAIN_BOTTOM = 240;   // signal band ends here -- ~300px of real plotting area
const REP_TOP = 254;       // replica band
const REP_BOTTOM = 312;
const AXIS_Y = 332;

/* ---------- pure helpers (DOM-free, so the scale maths stays checkable) ---------- */

export function makeScale(lo, hi, px0, px1) {
  const span = (hi - lo) || 1;
  return (v) => px0 + ((v - lo) / span) * (px1 - px0);
}

export function niceBounds(values, extra = []) {
  const all = values.concat(extra).filter((v) => v != null && !Number.isNaN(v));
  if (!all.length) return { lo: 0, hi: 1 };
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.12;
  return { lo: lo - pad, hi: hi + pad };
}

/** Path data for a series that may contain gaps (null = no signal). */
export function linePath(points, xOf, yOf, key) {
  let d = '';
  let pen = false;
  points.forEach((p) => {
    const v = p[key];
    if (v == null || Number.isNaN(v)) { pen = false; return; }
    d += (pen ? 'L' : 'M') + xOf(p.ts_ms).toFixed(1) + ' ' + yOf(v).toFixed(1) + ' ';
    pen = true;
  });
  return d.trim();
}

/** Step path: replica counts hold their value until the next sample. */
export function stepPath(points, xOf, yOf, key) {
  let d = '';
  let prevY = null;
  points.forEach((p) => {
    const v = p[key];
    if (v == null || Number.isNaN(v)) return;
    const x = xOf(p.ts_ms);
    const y = yOf(v);
    if (prevY == null) d += `M${x.toFixed(1)} ${y.toFixed(1)} `;
    else d += `L${x.toFixed(1)} ${prevY.toFixed(1)} L${x.toFixed(1)} ${y.toFixed(1)} `;
    prevY = y;
  });
  return d.trim();
}

/* ---------- rendering ---------- */

function svgEl(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, String(v)));
  if (parent) parent.appendChild(n);
  return n;
}

export function createChart(container, options) {
  const opts = options || {};
  const chart = {
    container,
    model: opts.model || '',
    color: opts.color || 'var(--m1)',
    points: [],
    taus: {},
    windowMs: opts.windowMs || 15 * 60 * 1000,
    showTps: false,
    onHover: null,
    _cursorTs: null,
  };

  const card = document.createElement('div');
  card.className = 'chart-card';
  const head = document.createElement('div');
  head.className = 'chart-head';
  const title = document.createElement('h3');
  title.textContent = chart.model;
  const legend = document.createElement('div');
  legend.className = 'chart-legend';
  head.appendChild(title);
  head.appendChild(legend);
  const readout = document.createElement('div');
  readout.className = 'chart-readout';
  head.appendChild(readout);
  card.appendChild(head);
  const holder = document.createElement('div');
  holder.className = 'chart-holder';
  card.appendChild(holder);
  container.appendChild(card);

  legend.innerHTML =
    '<span class="lg zm">Zₘ</span>' +
    '<span class="lg q">queue_len</span>' +
    '<span class="lg rep">replicas awake</span>' +
    '<span class="lg rept">target</span>' +
    '<span class="lg act">动作</span>';

  chart.update = function update(points, taus) {
    if (points) chart.points = points;
    if (taus) chart.taus = taus;
    draw();
  };

  chart.showCursorAt = function showCursorAt(tsMs) {
    chart._cursorTs = tsMs;
    drawCursor();
  };

  function windowRange() {
    const pts = chart.points;
    const end = pts.length ? pts[pts.length - 1].ts_ms : Date.now();
    return { start: end - chart.windowMs, end };
  }

  function visible() {
    const { start } = windowRange();
    return chart.points.filter((p) => p.ts_ms >= start);
  }

  let svg = null;
  let cursorLayer = null;
  let geom = null;

  function draw() {
    holder.innerHTML = '';
    const W = Math.max(640, holder.clientWidth || container.clientWidth || 900);
    svg = svgEl('svg', { width: W, height: H, class: 'chart-svg' }, holder);

    const pts = visible();
    const { start, end } = windowRange();
    const xOf = makeScale(start, end, PAD_L, W - PAD_R);

    const taus = chart.taus || {};
    const tauVals = [taus.tau_high, taus.tau_crit, taus.tau_low].filter((v) => v != null);
    const zb = niceBounds(pts.map((p) => p.z_m), tauVals);
    const yZ = makeScale(zb.lo, zb.hi, MAIN_BOTTOM, TOP);

    const qb = niceBounds(pts.map((p) => p.queue_len), [0]);
    const yQ = makeScale(qb.lo, qb.hi, MAIN_BOTTOM, TOP);

    geom = { W, xOf, yZ, yQ, start, end, pts };

    // plot frame
    svgEl('rect', { x: PAD_L, y: TOP, width: W - PAD_L - PAD_R, height: MAIN_BOTTOM - TOP,
                    class: 'plot-bg' }, svg);

    // threshold bands: above tau_high is the "needs capacity" region
    if (taus.tau_high != null) {
      const yTop = Math.max(TOP, yZ(zb.hi));
      const yHigh = yZ(taus.tau_high);
      svgEl('rect', { x: PAD_L, y: yTop, width: W - PAD_L - PAD_R,
                      height: Math.max(0, yHigh - yTop), class: 'band-high' }, svg);
      svgEl('line', { x1: PAD_L, x2: W - PAD_R, y1: yHigh, y2: yHigh, class: 'tau tau-high' }, svg);
      svgEl('text', { x: W - PAD_R + 4, y: yHigh + 4, class: 'tau-label' }, svg)
        .textContent = 'τ_high';
    }
    if (taus.tau_crit != null) {
      const yCrit = yZ(taus.tau_crit);
      svgEl('rect', { x: PAD_L, y: yCrit, width: W - PAD_L - PAD_R,
                      height: Math.max(0, MAIN_BOTTOM - yCrit), class: 'band-crit' }, svg);
      svgEl('line', { x1: PAD_L, x2: W - PAD_R, y1: yCrit, y2: yCrit, class: 'tau tau-crit' }, svg);
      svgEl('text', { x: W - PAD_R + 4, y: yCrit + 4, class: 'tau-label' }, svg)
        .textContent = 'τ_crit';
    }

    // left / right axis ticks
    [zb.lo, (zb.lo + zb.hi) / 2, zb.hi].forEach((v) => {
      svgEl('text', { x: PAD_L - 8, y: yZ(v) + 4, class: 'axis-l' }, svg).textContent = v.toFixed(2);
    });
    [qb.lo, qb.hi].forEach((v) => {
      svgEl('text', { x: W - PAD_R + 6, y: yQ(v) + 4, class: 'axis-r' }, svg).textContent = v.toFixed(0);
    });

    // action markers first, so the signal lines stay readable on top
    pts.filter((p) => p.action && p.action !== 'none').forEach((p) => {
      const x = xOf(p.ts_ms);
      const line = svgEl('line', { x1: x, x2: x, y1: TOP, y2: REP_BOTTOM, class: 'action-mark' }, svg);
      svgEl('title', {}, line).textContent =
        `${new Date(p.ts_ms).toLocaleTimeString()} · ${p.action}` +
        (p.tier ? ` · tier=${p.tier}` : '');
    });

    if (chart.showTps) {
      const tb = niceBounds(pts.map((p) => p.decode_tps).concat(pts.map((p) => p.prefill_tps)), [0]);
      const yT = makeScale(tb.lo, tb.hi, MAIN_BOTTOM, TOP);
      svgEl('path', { d: linePath(pts, xOf, yT, 'decode_tps'), class: 'line decode' }, svg);
      svgEl('path', { d: linePath(pts, xOf, yT, 'prefill_tps'), class: 'line prefill' }, svg);
    }

    svgEl('path', { d: linePath(pts, xOf, yQ, 'queue_len'), class: 'line queue' }, svg);
    const zPath = svgEl('path', { d: linePath(pts, xOf, yZ, 'z_m'), class: 'line zm' }, svg);
    zPath.style.stroke = chart.color;

    // replica band: awake vs target. A persistent gap is "wants to scale but cannot".
    const repMax = Math.max(1, ...pts.map((p) => Math.max(p.replicas_target || 0, p.replicas_awake || 0)));
    const yR = makeScale(0, repMax, REP_BOTTOM, REP_TOP);
    svgEl('rect', { x: PAD_L, y: REP_TOP, width: W - PAD_L - PAD_R,
                    height: REP_BOTTOM - REP_TOP, class: 'plot-bg' }, svg);
    svgEl('text', { x: PAD_L - 8, y: REP_TOP + 10, class: 'axis-l' }, svg).textContent = String(repMax);
    svgEl('text', { x: PAD_L - 8, y: REP_BOTTOM, class: 'axis-l' }, svg).textContent = '0';
    svgEl('path', { d: stepPath(pts, xOf, yR, 'replicas_target'), class: 'line rep-target' }, svg);
    svgEl('path', { d: stepPath(pts, xOf, yR, 'replicas_awake'), class: 'line rep-awake' }, svg);

    // x axis
    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      const t = start + (end - start) * f;
      const x = PAD_L + (W - PAD_L - PAD_R) * f;
      svgEl('text', { x, y: AXIS_Y, class: 'axis-x' }, svg)
        .textContent = new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    });

    cursorLayer = svgEl('g', { class: 'cursor-layer' }, svg);
    drawCursor();

    svg.addEventListener('mousemove', (ev) => {
      const rect = svg.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const frac = (x - PAD_L) / Math.max(1, (geom.W - PAD_L - PAD_R));
      const ts = geom.start + (geom.end - geom.start) * Math.min(1, Math.max(0, frac));
      if (chart.onHover) chart.onHover(ts); else chart.showCursorAt(ts);
    });
    svg.addEventListener('mouseleave', () => {
      if (chart.onHover) chart.onHover(null); else chart.showCursorAt(null);
    });
  }

  function nearest(tsMs) {
    let best = null;
    let bestGap = Infinity;
    geom.pts.forEach((p) => {
      const gap = Math.abs(p.ts_ms - tsMs);
      if (gap < bestGap) { bestGap = gap; best = p; }
    });
    return best;
  }

  function drawCursor() {
    if (!cursorLayer || !geom) return;
    cursorLayer.innerHTML = '';
    if (chart._cursorTs == null || !geom.pts.length) { readout.textContent = ''; return; }
    const x = geom.xOf(chart._cursorTs);
    svgEl('line', { x1: x, x2: x, y1: TOP, y2: REP_BOTTOM, class: 'cursor' }, cursorLayer);
    const p = nearest(chart._cursorTs);
    if (!p) { readout.textContent = ''; return; }
    if (p.z_m != null) svgEl('circle', { cx: geom.xOf(p.ts_ms), cy: geom.yZ(p.z_m), r: 3.5, class: 'cursor-dot' }, cursorLayer);
    readout.textContent =
      `${new Date(p.ts_ms).toLocaleTimeString()}  ` +
      `Zₘ ${p.z_m == null ? 'idle' : p.z_m.toFixed(3)}  ` +
      `queue ${p.queue_len == null ? '—' : p.queue_len.toFixed(1)}  ` +
      `awake ${p.replicas_awake ?? '—'}/${p.replicas_target ?? '—'}  ` +
      `${p.tier || ''}${p.action && p.action !== 'none' ? ' · ' + p.action : ''}`;
  }

  chart.redraw = draw;
  return chart;
}

/** Hovering any chart moves the cursor on all of them -- cross-model comparison
    is the whole point of Zₘ, so the charts must share one time cursor. */
export function linkCrosshair(charts) {
  charts.forEach((chart) => {
    chart.onHover = (tsMs) => charts.forEach((other) => other.showCursorAt(tsMs));
  });
}
