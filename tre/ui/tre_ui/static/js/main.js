/* Console entry point: load static meta once, then let the SSE feed drive. */
'use strict';

import {
  MODEL_HUES, S, api, buildNav, openStream, renderTopbar, switchView, toast,
} from './core.js';
import './view_overview.js';
import './view_signals.js';
import './view_fleet.js';
import { initOps } from './view_ops.js';

async function init() {
  buildNav();
  try {
    S.meta = await api('/api/meta');
  } catch (e) {
    toast('meta 加载失败: ' + e.message, true);
    S.meta = { models: [], topology: { nodes: [] } };
  }
  (S.meta.models || []).forEach((m, i) => {
    S.colors[m.name] = MODEL_HUES[i % MODEL_HUES.length];
  });
  try { S.mode = (await api('/api/ops/controller/mode')).mode; } catch (_) { /* keep default */ }

  initOps();
  renderTopbar();
  switchView('overview');
  openStream();
}

document.addEventListener('DOMContentLoaded', init);
