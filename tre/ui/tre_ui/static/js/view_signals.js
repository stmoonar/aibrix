/* Signals -- one full-width chart per model, sharing a time cursor. */
'use strict';

import { $, S, api, el, registerView, toast } from './core.js';
import { createChart, linkCrosshair } from './chart.js';

const WINDOWS = [['5m', 5 * 60 * 1000], ['15m', 15 * 60 * 1000], ['1h', 60 * 60 * 1000]];

const state = {
  charts: [],
  windowMs: 15 * 60 * 1000,
  paused: false,
  showTps: false,
  built: false,
  fetching: false,
  lastFetch: 0,
};

function buildBar() {
  const seg = $('#sig-window');
  seg.innerHTML = '';
  WINDOWS.forEach(([label, ms]) => {
    const b = el('button', state.windowMs === ms ? 'on' : '', label);
    b.onclick = () => {
      state.windowMs = ms;
      state.charts.forEach((c) => { c.windowMs = ms; c.redraw(); });
      buildBar();
    };
    seg.appendChild(b);
  });

  const pause = $('#sig-pause');
  pause.textContent = state.paused ? '继续' : '暂停';
  pause.classList.toggle('primary', state.paused);
  pause.onclick = () => { state.paused = !state.paused; buildBar(); };

  const tps = $('#sig-tps');
  tps.checked = state.showTps;
  tps.onchange = () => {
    state.showTps = tps.checked;
    state.charts.forEach((c) => { c.showTps = state.showTps; c.redraw(); });
  };
}

function buildCharts() {
  const root = $('#sig-charts');
  root.innerHTML = '';
  state.charts = (S.meta && S.meta.models ? S.meta.models : []).map((meta) => {
    const chart = createChart(root, {
      model: meta.name,
      color: S.colors[meta.name],
      windowMs: state.windowMs,
    });
    chart.taus = meta.trs || {};
    return chart;
  });
  linkCrosshair(state.charts);
  state.built = state.charts.length > 0;
}

async function refresh() {
  if (state.paused || state.fetching || !state.built) return;
  state.fetching = true;
  const since = Date.now() - state.windowMs - 60000;
  try {
    await Promise.all(state.charts.map(async (chart) => {
      const payload = await api(`/api/signal/timeline?model=${encodeURIComponent(chart.model)}&since_ms=${since}`);
      chart.update(payload.points || []);
    }));
    const n = state.charts.reduce((acc, c) => acc + c.points.length, 0);
    $('#sig-status').textContent = `${n} 个窗口点 · ${state.paused ? '已暂停' : '刷新中'}`;
  } catch (e) {
    $('#sig-status').textContent = 'timeline 拉取失败: ' + e.message;
  } finally {
    state.fetching = false;
    state.lastFetch = Date.now();
  }
}

window.addEventListener('resize', () => {
  if (S.view === 'signals') state.charts.forEach((c) => c.redraw());
});

registerView('signals', {
  onEnter() {
    if (!state.built) { buildBar(); buildCharts(); }
    refresh();
  },
  render() {
    if (S.view !== 'signals') return;
    if (!state.built) { buildBar(); buildCharts(); }
    // The SSE frame arrives every 0.5s; the timeline is a separate, bounded
    // pull, so throttle it to roughly the sampler's own 2s cadence.
    if (Date.now() - state.lastFetch > 2000) refresh();
  },
});
