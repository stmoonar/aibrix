/* Assertions for the chart module's pure scale maths. Driven by
   ui/tests/test_chart_js.py, which passes the module path as argv[2]. */
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const target = process.argv[2];
if (!target) {
  console.error('usage: node chart_check.mjs <path to chart.js>');
  process.exit(2);
}
const { makeScale, niceBounds, linePath, stepPath } = await import(pathToFileURL(target).href);

// makeScale maps a value range onto a pixel range, inverted for SVG y.
const y = makeScale(0, 10, 100, 0);
assert.equal(y(0), 100);
assert.equal(y(10), 0);
assert.equal(y(5), 50);

// a degenerate range must not divide by zero
assert.equal(Number.isFinite(makeScale(3, 3, 0, 100)(3)), true);

// niceBounds pads, takes the tau lines into account, and survives an all-null series
const b = niceBounds([1, 2], [5]);
assert.ok(b.lo < 1 && b.hi > 5);
assert.deepEqual(niceBounds([null, NaN]), { lo: 0, hi: 1 });
const flat = niceBounds([2, 2]);
assert.ok(flat.lo < 2 && flat.hi > 2);

// An idle model reports z_m = null. The stroke must break there rather than
// bridge the gap, which would draw a line through data that does not exist.
const d = linePath(
  [{ ts_ms: 0, z_m: 1 }, { ts_ms: 1, z_m: null }, { ts_ms: 2, z_m: 3 }],
  (t) => t, (v) => v, 'z_m',
);
assert.equal((d.match(/M/g) || []).length, 2, 'a gap must start a new subpath: ' + d);
assert.equal((d.match(/L/g) || []).length, 0);

// stepPath holds the previous value until the next sample (replica counts).
assert.equal(
  stepPath([{ ts_ms: 0, r: 1 }, { ts_ms: 10, r: 3 }], (t) => t, (v) => v, 'r'),
  'M0.0 1.0 L10.0 1.0 L10.0 3.0',
);

console.log('chart maths OK');
