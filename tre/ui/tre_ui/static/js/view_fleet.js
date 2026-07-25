/* GPU Fleet -- node x GPU matrix joining bindings, GPU truth and leases. */
'use strict';

import { $, S, el, emptyState, fmtInt, registerView } from './core.js';

function leaseFor(node, idx) {
  const leases = ((S.snap || {}).leases || {}).gpus || {};
  return leases[node + '/' + idx] || null;
}

function observedIds() {
  const state = ((S.snap || {}).fleet || {}).state || {};
  const ids = new Set();
  (state.observed || []).forEach((o) => {
    const id = typeof o === 'string' ? o : (o.binding_id || o.serve_id);
    if (id) ids.add(id);
  });
  return ids;
}

function desiredIds() {
  const state = ((S.snap || {}).fleet || {}).state || {};
  const ids = new Set();
  (state.desired || []).forEach((o) => {
    const id = typeof o === 'string' ? o : (o.binding_id || o.serve_id);
    if (id) ids.add(id);
  });
  return ids;
}

function gpuCell(node, gpu, idx, resident, desired, observed) {
  const cell = el('div', 'gpu');
  const usedFrac = gpu.total_mib ? gpu.used_mib / gpu.total_mib : 0;
  const awakeCount = resident.filter((b) => b.awake).length;
  const lease = leaseFor(node.node, idx);
  if (awakeCount > 1) cell.classList.add('alarm');

  const head = el('div', 'gh');
  head.appendChild(el('span', 'idx', 'GPU ' + idx));
  head.appendChild(el('span', 'uuid', (gpu.uuid || '').replace('GPU-', '').slice(0, 8)));
  cell.appendChild(head);

  const bar = el('div', 'membar' + (usedFrac > 0.9 ? ' hot' : ''));
  const fill = el('i');
  fill.style.width = Math.min(100, usedFrac * 100).toFixed(1) + '%';
  bar.appendChild(fill);
  cell.appendChild(bar);
  cell.appendChild(el('div', 'memtext',
    `${(gpu.used_mib / 1024).toFixed(1)} / ${(gpu.total_mib / 1024).toFixed(0)} GiB · ${(usedFrac * 100).toFixed(0)}%`));

  const res = el('div', 'resident');
  if (!resident.length) {
    res.appendChild(el('div', 'empty', 'free'));
  } else {
    resident.forEach((b) => {
      const line = el('div', 'bind');
      const sw = el('span', 'swatch');
      sw.style.background = S.colors[b.model] || 'var(--idle)';
      line.appendChild(sw);
      const nm = el('span', 'mname', b.model + ((b.gpu_ids || []).length > 1 ? ` (tp${b.gpu_ids.length})` : ''));
      line.appendChild(nm);
      const stt = b.hidden ? 'hidden' : (b.awake ? 'awake' : 'asleep');
      line.appendChild(el('span', 'st ' + stt, stt));
      const id = b.binding_id || b.serve_id;
      if (id && desired.size && !observed.has(id)) {
        line.classList.add('mismatch');
        line.appendChild(el('span', 'st mismatch', 'desired≠observed'));
      }
      if (b.serve_id) line.appendChild(el('span', 'pod sub', b.serve_id));
      res.appendChild(line);
    });
    if (awakeCount > 1) res.appendChild(el('div', 'leakflag', `⚠ ${awakeCount} awake on one GPU`));
  }
  cell.appendChild(res);

  const lz = el('div', 'lease');
  if (lease) {
    lz.appendChild(el('span', 'lease-phase ' + (lease.phase || ''), lease.phase || '—'));
    lz.appendChild(el('span', 'sub', 'fence ' + fmtInt(lease.fencing_token)));
    if (lease.owner) {
      const owner = el('span', 'sub owner', lease.owner);
      owner.title = lease.owner;
      lz.appendChild(owner);
    }
  } else {
    lz.appendChild(el('span', 'sub', 'no lease'));
  }
  cell.appendChild(lz);
  return cell;
}

registerView('fleet', {
  render() {
    if (S.view !== 'fleet') return;
    const root = $('#fleet-root');
    root.innerHTML = '';
    const snap = S.snap || {};
    const nodes = (snap.gpu_truth || {}).nodes || [];
    const bindings = ((snap.sm || {}).state || {}).bindings || [];
    if (!nodes.length) { root.appendChild(emptyState('等待 GPU truth')); return; }

    const observed = observedIds();
    const desired = desiredIds();

    nodes.forEach((node) => {
      const block = el('div', 'node-block');
      const head = el('div', 'node-head');
      head.appendChild(el('h3', null, node.node));
      const nodeBinds = bindings.filter((b) => b.node === node.node);
      head.appendChild(el('span', 'sub',
        `${nodeBinds.length} resident · ${nodeBinds.filter((b) => b.awake).length} awake`));
      block.appendChild(head);

      const grid = el('div', 'gpu-row');
      (node.gpus || []).forEach((gpu, idx) => {
        const resident = nodeBinds.filter((b) => (b.gpu_ids || []).includes(idx));
        grid.appendChild(gpuCell(node, gpu, idx, resident, desired, observed));
      });
      block.appendChild(grid);
      root.appendChild(block);
    });
  },
});
