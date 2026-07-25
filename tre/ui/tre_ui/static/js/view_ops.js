/* Ops & Control -- operation journal, supervisor, manual audit, mode and params.

   The parameter editor and the mode/reconcile/defrag operations are carried
   over from the previous console unchanged: they work, and rewriting them would
   only add risk. What is new is everything above them -- the journal, the
   supervisor panel and the audit trigger, which the old console never showed. */
'use strict';

import {
  $, S, ageText, api, clockText, confirmOp, el, emptyState, fmt, fmtInt,
  registerView, renderTopbar, toast,
} from './core.js';

/* ---------- operation journal ---------- */

function duration(op) {
  const started = op.started_at_ms || op.created_at_ms;
  const ended = op.finished_at_ms || op.completed_at_ms;
  if (!started || !ended) return '';
  return ((ended - started) / 1000).toFixed(1) + 's';
}

function renderJournal() {
  const root = $('#ops-journal');
  root.innerHTML = '';
  const items = ((S.snap || {}).operations || {}).items || [];
  if (!items.length) { root.appendChild(emptyState('journal 为空')); return; }
  items.slice(0, 40).forEach((op) => {
    const status = String(op.status || '').toLowerCase();
    const row = el('div', 'jrow ' + status);
    row.appendChild(el('span', 'j-id', String(op.id || '').slice(0, 8)));
    row.appendChild(el('span', 'j-kind', op.kind || op.type || op.name || '—'));
    row.appendChild(el('span', 'j-status ' + status, status || '—'));
    row.appendChild(el('span', 'j-dur sub', duration(op)));
    const when = op.finished_at_ms || op.started_at_ms || op.created_at_ms;
    row.appendChild(el('span', 'j-when sub', when ? clockText(when) : ''));
    const err = op.error || op.failure_reason || op.message;
    if (err) row.appendChild(el('span', 'j-err', String(err)));
    root.appendChild(row);
  });
}

function renderSupervisor() {
  const root = $('#ops-supervisor');
  root.innerHTML = '';
  const sup = ((S.snap || {}).fleet || {}).supervisor || {};
  const rows = [
    ['enabled', String(sup.enabled ?? '—')],
    ['running', String(sup.running ?? '—')],
    ['drift observations', fmtInt(sup.drift_observations)],
    ['last error', sup.last_error || '—'],
    ['last recovery op', sup.last_recovery_operation_id || '—'],
    ['last drift', (sup.last_drift || []).length ? JSON.stringify(sup.last_drift).slice(0, 300) : '—'],
    ['age', ageText(((S.snap || {}).fleet || {}).age_ms)],
  ];
  rows.forEach(([k, v]) => {
    const row = el('div', 'kv');
    row.appendChild(el('span', 'k', k));
    row.appendChild(el('span', 'v', v));
    root.appendChild(row);
  });
}

/* ---------- manual audit ---------- */

async function loadAudit() {
  try { S.audit = await api('/api/ops/audit'); } catch (_) { /* leave the last value */ }
  renderAudit();
  renderTopbar();
}

async function runAudit() {
  const btn = $('#ops-audit-run');
  btn.disabled = true;
  btn.textContent = '审计中…';
  try {
    S.audit = await api('/api/ops/audit', { method: 'POST' });
    toast('审计完成');
  } catch (e) {
    toast('审计失败: ' + e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = '运行审计';
    renderAudit();
    renderTopbar();
  }
}

function renderAudit() {
  const root = $('#ops-audit');
  root.innerHTML = '';
  const audit = S.audit || {};
  $('#ops-audit-when').textContent = audit.ran_at_ms
    ? '上次运行 ' + clockText(audit.ran_at_ms) : '本会话尚未运行';
  const result = audit.result;
  if (!result) { root.appendChild(emptyState('点击上方按钮运行一次审计')); return; }
  const issues = result.issues || [];
  if (!issues.length) {
    root.appendChild(Object.assign(el('div', 'issue ok'),
      { textContent: `healthy · state version ${fmtInt(result.version)}` }));
    return;
  }
  issues.forEach((issue) => {
    root.appendChild(el('div', 'issue bad',
      typeof issue === 'string' ? issue : JSON.stringify(issue)));
  });
}

/* ---------- controller mode + fleet ops (carried over) ---------- */

function syncModeButtons() {
  const a = $('#mode-active');
  const o = $('#mode-observe');
  if (!a) return;
  a.className = S.mode === 'active' ? 'on-active' : '';
  o.className = S.mode === 'observe' ? 'on-observe' : '';
  renderTopbar();
}

async function setMode(mode) {
  if (mode === S.mode) return;
  const msg = mode === 'observe'
    ? '暂停 controller 执行？它会继续计算决策，但停止扩缩和 hide。'
    : '恢复 controller 执行？';
  confirmOp(msg, async () => {
    const r = await api('/api/ops/controller/mode', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    S.mode = r.mode;
    syncModeButtons();
    return r;
  }, `controller → ${mode}`);
}

/* ---------- params (carried over verbatim) ---------- */

const FIELD_LABEL = {
  'trs.theta_m': 'θₘ (theta)', 'trs.tau_crit': 'τ crit', 'trs.tau_low': 'τ low', 'trs.tau_high': 'τ high',
  'trs.w_p': 'w_p', 'trs.w_d': 'w_d', 'trs.lambda_wait': 'λ wait', 'trs.qmin': 'q_min', 'trs.qsat': 'q_sat',
  'trs.ema_alpha': 'EMA α', 'trs.epsat': 'ε_sat', 'trs.hsat': 'h_sat',
  'slo.ttft_p95_ms': 'SLO ttft p95', 'slo.tpot_p95_ms': 'SLO tpot p95', 'slo.e2e_p95_ms': 'SLO e2e p95',
  'min_replicas': 'replicas min', 'max_replicas': 'replicas max',
};
const FIELD_ORDER = ['trs.theta_m', 'trs.tau_crit', 'trs.tau_low', 'trs.tau_high', 'trs.qmin', 'trs.qsat',
  'trs.w_p', 'trs.w_d', 'trs.lambda_wait', 'trs.ema_alpha', 'trs.epsat', 'trs.hsat',
  'slo.ttft_p95_ms', 'slo.tpot_p95_ms', 'slo.e2e_p95_ms', 'min_replicas', 'max_replicas'];

function smField(name, field) {
  const models = ((S.snap || {}).sm || {}).state || {};
  const entry = (models.models || {})[name] || {};
  return entry[field] != null ? entry[field] : '—';
}

export async function loadParams() {
  try {
    S.params = await api('/api/params');
    S.paramsAvailable = true;
  } catch (e) {
    S.paramsAvailable = false;
    S.paramsError = /503/.test(e.message)
      ? '参数编辑需要集群内访问（kubernetes）。当前只读显示。' : e.message;
  }
  rebuildPanels();
  renderParamsBar();
}

function rebuildPanels() {
  const grid = $('#ctl-grid');
  grid.innerHTML = '';
  ((S.meta || {}).models || []).forEach((meta) => grid.appendChild(controlPanel(meta)));
}

function renderParamsBar() {
  const bar = $('#params-bar');
  if (!bar) return;
  bar.innerHTML = '';
  if (S.paramsAvailable === false) {
    bar.appendChild(Object.assign(el('div', 'sub'), { textContent: S.paramsError || '' }));
    return;
  }
  const pending = S.params && S.params.pending_restart;
  const wrap = el('div', 'restart-strip' + (pending ? ' pending' : ''));
  wrap.appendChild(el('span', null, pending
    ? '⚠ 参数已保存但尚未生效 —— 需重启 controller 才会加载。'
    : '参数与运行中的 controller 一致。'));
  const btn = el('button', 'btn ' + (pending ? 'primary' : ''), '重启 controller');
  btn.onclick = restartController;
  wrap.appendChild(btn);
  const status = el('span', 'sub', '');
  status.id = 'rollout-status';
  wrap.appendChild(status);
  if (S.params) {
    wrap.appendChild(el('span', 'sub',
      `hash ${String(S.params.params_hash || '').slice(0, 12)} · applied ${String(S.params.applied_hash || '').slice(0, 12)}`));
  }
  bar.appendChild(wrap);
}

function controlPanel(meta) {
  const name = meta.name;
  const view = (S.params && S.params.models && S.params.models[name]) || null;
  const p = el('div', 'panel');
  p.style.setProperty('--accent', S.colors[name]);
  p.dataset.model = name;
  p.appendChild(el('h3', null, name));
  p.appendChild(el('div', 'accent-bar'));

  const liveRow = el('div', 'row');
  liveRow.innerHTML = '<span class="muted">awake / bound</span><span class="num" data-fld="awb">—</span>';
  p.appendChild(liveRow);

  const tRow = el('div', 'row');
  tRow.appendChild(Object.assign(el('span', 'muted'), { textContent: 'wake target' }));
  const stepper = el('div', 'stepper');
  const dec = el('button', null, '−');
  const val = el('span', 'val num', '0');
  const inc = el('button', null, '+');
  let target = 0;
  const setT = (v) => {
    target = Math.max(meta.min_replicas, Math.min(meta.max_replicas, v));
    val.textContent = target;
  };
  dec.onclick = () => setT(target - 1);
  inc.onclick = () => setT(target + 1);
  stepper.append(dec, val, inc);
  tRow.appendChild(stepper);
  const setBtn = el('button', 'btn', 'Set');
  setBtn.onclick = () => confirmOp(`把 ${name} 的 wake target 设为 ${target}？`,
    () => api(`/api/ops/models/${name}/target`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wake_replicas: target }),
    }), `${name} target → ${target}`);
  tRow.appendChild(setBtn);
  p.appendChild(tRow);

  if (!view) {
    const P = meta.trs;
    const SLO = meta.slo;
    const tbl = el('table', 'params');
    [['θₘ', fmt(P.theta_m, 2)],
     ['τ c/l/h', `${fmt(P.tau_crit, 2)}/${fmt(P.tau_low, 2)}/${fmt(P.tau_high, 2)}`],
     ['SLO ttft/tpot/e2e', `${fmtInt(SLO.ttft_p95_ms)}/${fmtInt(SLO.tpot_p95_ms)}/${fmtInt(SLO.e2e_p95_ms)}`]]
      .forEach(([k, v]) => {
        const tr = el('tr');
        tr.appendChild(el('td', null, k));
        tr.appendChild(el('td', null, v));
        tbl.appendChild(tr);
      });
    p.appendChild(tbl);
    return p;
  }

  const form = el('div', 'pform');
  form.dataset.model = name;
  FIELD_ORDER.forEach((fld) => {
    const spec = view.editable[fld];
    if (!spec) return;
    const rowE = el('label', 'prow');
    rowE.appendChild(el('span', 'plabel', FIELD_LABEL[fld] || fld));
    const input = el('input', 'pinput num');
    input.type = 'number';
    input.value = spec.value;
    input.step = spec.type === 'int' ? '1' : 'any';
    input.min = spec.min;
    input.max = spec.max;
    input.dataset.field = fld;
    input.dataset.orig = String(spec.value);
    input.oninput = () => {
      input.classList.toggle('dirty', input.value !== input.dataset.orig);
      markPanelDirty(p);
    };
    rowE.appendChild(input);
    rowE.appendChild(el('span', 'phint', `${spec.min}–${spec.max}`));
    form.appendChild(rowE);
  });
  p.appendChild(form);

  const locked = el('div', 'locked-note');
  locked.appendChild(el('span', 'muted', 'locked: '));
  Object.entries(view.locked).forEach(([k, info]) => {
    const chip = el('span', 'lchip', k.replace('trs.', ''));
    chip.title = info.reason + ' = ' + info.value;
    locked.appendChild(chip);
  });
  p.appendChild(locked);

  const err = el('div', 'perr');
  err.dataset.err = name;
  p.appendChild(err);
  const actions = el('div', 'actions');
  const save = el('button', 'btn primary', '保存参数');
  save.dataset.save = name;
  save.disabled = true;
  save.onclick = () => saveModelParams(name, p);
  const reset = el('button', 'btn', '重置');
  reset.onclick = () => {
    p.querySelectorAll('.pinput').forEach((i) => {
      i.value = i.dataset.orig;
      i.classList.remove('dirty');
    });
    markPanelDirty(p);
  };
  actions.append(save, reset);
  p.appendChild(actions);
  return p;
}

function markPanelDirty(panel) {
  const dirty = panel.querySelectorAll('.pinput.dirty').length > 0;
  const save = panel.querySelector('[data-save]');
  if (save) save.disabled = !dirty;
}

async function saveModelParams(name, panel) {
  const changes = { trs: {}, slo: {} };
  panel.querySelectorAll('.pinput.dirty').forEach((i) => {
    const [sec, key] = i.dataset.field.includes('.')
      ? i.dataset.field.split('.') : [null, i.dataset.field];
    const v = i.step === '1' ? parseInt(i.value, 10) : parseFloat(i.value);
    if (sec) changes[sec][key] = v; else changes[key] = v;
  });
  const errBox = panel.querySelector('[data-err]');
  errBox.textContent = '';
  if (!window.confirm(`保存 ${name} 的参数改动？需重启 controller 后生效。`)) return;
  try {
    S.params = await api('/api/params', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        expected_resource_version: S.params.resource_version,
        models: { [name]: changes },
      }),
    });
    toast(`${name} 参数已保存 —— 重启后生效`);
    rebuildPanels();
    renderParamsBar();
    renderPanelsLive();
  } catch (e) {
    const m = e.message.match(/\{.*\}/s);
    if (m) {
      try {
        const d = JSON.parse(m[0]);
        errBox.textContent = (d.errors || [])
          .map((x) => `${x.field || ''}: ${x.error}${x.detail ? ' (' + x.detail + ')' : ''}`).join('; ');
      } catch (_) { errBox.textContent = e.message; }
    } else errBox.textContent = e.message;
    toast('保存被拒绝', true);
  }
}

async function restartController() {
  if (!window.confirm('重启 controller 以应用已保存的参数？控制会短暂（数秒）暂停，随后自动恢复。')) return;
  try {
    await api('/api/ops/controller/restart', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: 'apply params from console' }),
    });
    toast('已请求重启 controller');
    pollRollout(0);
  } catch (e) { toast('重启失败: ' + e.message, true); }
}

async function pollRollout(n) {
  const status = $('#rollout-status');
  if (!status) return;
  try {
    const r = await api('/api/ops/controller/rollout');
    status.textContent = `rollout: ${r.state} (${r.ready_replicas}/${r.desired})`;
    if (r.state === 'ready') { toast('controller 已重启 —— 参数生效'); loadParams(); return; }
    if (r.state === 'failed') {
      status.textContent = 'rollout FAILED: ' + (r.message || '');
      toast('rollout 失败', true);
      return;
    }
  } catch (_) { /* keep polling */ }
  if (n < 40) setTimeout(() => pollRollout(n + 1), 2000);
}

function renderPanelsLive() {
  document.querySelectorAll('#ctl-grid .panel').forEach((p) => {
    const name = p.dataset.model;
    const awb = p.querySelector('[data-fld="awb"]');
    if (awb) awb.textContent = `${smField(name, 'awake')} / ${smField(name, 'bound')}`;
  });
}

export function initOps() {
  $('#ops-audit-run').onclick = runAudit;
  $('#mode-active').onclick = () => setMode('active');
  $('#mode-observe').onclick = () => setMode('observe');
  $('#op-reconcile').onclick = () => confirmOp('现在执行一次 reconcile？',
    () => api('/api/ops/reconcile', { method: 'POST' }), 'reconcile 已请求');
  $('#op-defrag').onclick = () => confirmOp('执行 defrag (tp_size=2)？可能迁移睡眠中的副本。',
    () => api('/api/ops/defrag', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tp_size: 2 }),
    }), 'defrag 已请求');
  syncModeButtons();
  loadAudit();
  loadParams();
}

registerView('ops', {
  render() {
    if (S.view !== 'ops') return;
    renderJournal();
    renderSupervisor();
    renderPanelsLive();
  },
});
