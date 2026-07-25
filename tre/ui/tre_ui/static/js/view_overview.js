/* Overview -- answers one question: do I need to look deeper right now? */
'use strict';

import {
  $, S, ageText, clockText, el, emptyState, fmtInt, freshness, registerView, zText,
} from './core.js';

function tile(label, value, tone, detail) {
  const t = el('div', 'tile ' + (tone || ''));
  t.appendChild(el('div', 'tile-label', label));
  t.appendChild(el('div', 'tile-value', value));
  if (detail) t.appendChild(el('div', 'tile-detail', detail));
  return t;
}

function renderHealth() {
  const root = $('#ov-health');
  root.innerHTML = '';
  const snap = S.snap || {};
  const sup = (snap.fleet || {}).supervisor || {};
  const audit = S.audit || {};

  const observeReason = S.mode !== 'observe' ? ''
    : (sup.last_recovery_operation_id
        ? 'supervisor 自动切换（repair ' + String(sup.last_recovery_operation_id).slice(0, 8) + '），需人工确认后恢复 active'
        : '人工切换');
  root.appendChild(tile('controller', S.mode === 'observe' ? 'OBSERVE' : 'ACTIVE',
    S.mode === 'observe' ? 'warn' : 'ok', observeReason || '正在执行决策'));

  const supTone = sup.running == null ? '' : (sup.running && !sup.last_error ? 'ok' : 'bad');
  root.appendChild(tile('supervisor', sup.running == null ? '—' : (sup.running ? 'running' : 'stopped'),
    supTone, 'drift ' + fmtInt(sup.drift_observations) + (sup.last_error ? ' · ' + sup.last_error : '')));

  const issues = ((audit.result || {}).issues || []).length;
  root.appendChild(tile('一致性审计',
    audit.ran_at_ms == null ? '未运行' : (issues ? issues + ' issues' : 'healthy'),
    audit.ran_at_ms == null ? '' : (issues ? 'bad' : 'ok'),
    audit.ran_at_ms == null ? '在 Ops 页手动运行' : '于 ' + clockText(audit.ran_at_ms)));

  const decisionAge = (snap.decision || {}).age_ms;
  root.appendChild(tile('决策循环', ageText(decisionAge),
    freshness(decisionAge, 10000) === 'stale' ? 'warn' : 'ok',
    'loop ' + (((snap.decision || {}).latest || {}).loop || '—')));

  ((snap.gpu_truth || {}).nodes || []).forEach((node) => {
    const stale = freshness((snap.gpu_truth || {}).age_ms) === 'stale';
    root.appendChild(tile('GPU truth · ' + node.node, stale ? 'stale' : 'fresh',
      stale ? 'bad' : 'ok', (node.gpus || []).length + ' GPU'));
  });
}

function renderFleet() {
  const root = $('#ov-fleet');
  root.innerHTML = '';
  const state = ((S.snap || {}).fleet || {}).state || {};
  if (state.error) { root.appendChild(emptyState('service-manager 不可达: ' + state.error)); return; }

  const desired = (state.desired || []).length;
  const observed = (state.observed || []).length;
  const mismatches = state.mismatches || [];
  const row = el('div', 'fleet-counts');
  row.appendChild(tile('desired', String(desired), '', 'v' + fmtInt(state.desired_version)));
  row.appendChild(tile('observed', String(observed), '', 'v' + fmtInt(state.observed_version)));
  row.appendChild(tile('mismatch', String(mismatches.length),
    mismatches.length ? 'bad' : 'ok', mismatches.length ? '需要收敛' : '已收敛'));
  root.appendChild(row);

  if (mismatches.length) {
    const list = el('div', 'issues');
    mismatches.slice(0, 20).forEach((m) => {
      list.appendChild(el('div', 'issue', typeof m === 'string' ? m : JSON.stringify(m)));
    });
    root.appendChild(list);
  }
}

function renderModels() {
  const root = $('#ov-models');
  root.innerHTML = '';
  const snap = S.snap || {};
  const states = ((snap.decision || {}).latest || {}).model_states || {};
  const smModels = ((snap.sm || {}).state || {}).models || {};

  (S.meta && S.meta.models ? S.meta.models : []).forEach((meta) => {
    const st = states[meta.name] || {};
    const sm = smModels[meta.name] || {};
    const row = el('div', 'ov-row');
    row.style.setProperty('--accent', S.colors[meta.name] || 'var(--m1)');
    row.appendChild(el('span', 'accent-dot'));
    row.appendChild(el('span', 'ov-name', meta.name));

    const tier = String(st.state || '—');
    row.appendChild(el('span', 'ov-tier tier-' + tier.toLowerCase(), tier));

    const z = el('span', 'ov-z num', zText(st.z_m, st.state));
    if (st.z_m == null) z.classList.add('muted');
    row.appendChild(z);

    const taus = (meta.trs || {});
    row.appendChild(el('span', 'ov-tau sub',
      'τ ' + (taus.tau_crit ?? '—') + ' / ' + (taus.tau_high ?? '—')));

    row.appendChild(el('span', 'ov-rep num',
      'awake ' + fmtInt(sm.awake) + ' / bound ' + fmtInt(sm.bound)));
    row.appendChild(el('span', 'ov-routable num',
      'routable ' + fmtInt(st.routable_pods)));
    root.appendChild(row);
  });
  if (!root.children.length) root.appendChild(emptyState('等待 /api/meta'));
}

function renderFeed() {
  const root = $('#ov-feed');
  root.innerHTML = '';
  const events = ((S.snap || {}).events_head || []).slice(0, 5);
  if (!events.length) { root.appendChild(emptyState('暂无动作')); return; }
  events.forEach((e) => {
    const row = el('div', 'ev');
    row.appendChild(el('span', 'ev-t', clockText(e.ts_ms)));
    row.appendChild(el('span', 'ev-k ' + (e.kind || ''), e.kind || ''));
    row.appendChild(el('span', 'ev-x', e.text || ''));
    root.appendChild(row);
  });
}

registerView('overview', {
  render() {
    if (S.view !== 'overview') return;
    renderHealth();
    renderFleet();
    renderModels();
    renderFeed();
  },
});
