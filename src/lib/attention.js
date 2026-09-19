const WQ = [
  [0.10, 0.05, 1.00, 0.42],
  [0.00, 0.12, 0.35, 0.95],
  [0.10, 0.25, 0.18, 0.35],
  [0.25, 0.10, 0.08, 0.10],
];

const WK = [
  [0.08, 0.05, 0.95, 0.40],
  [0.00, 0.10, 0.30, 0.92],
  [0.08, 0.24, 0.20, 0.32],
  [0.24, 0.08, 0.05, 0.12],
];

function normalize([x, xDot, theta, thetaDot]) {
  return [x / 2.4, xDot / 3.0, theta / 0.38, thetaDot / 3.5];
}
function matVec(M, v) {
  return M.map(row => row.reduce((sum, w, i) => sum + w * v[i], 0));
}
function dot(a, b) {
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}
function softmax(xs) {
  const max = Math.max(...xs);
  const e = xs.map(x => Math.exp(x - max));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / z);
}

export function runAttention(history) {
  const tokens = history.map(normalize);
  const q = tokens.map(v => matVec(WQ, v));
  const k = tokens.map(v => matVec(WK, v));
  const v = tokens.map(t => [...t]);

  const raw = q.map((qi, i) =>
    k.map((kj, j) => j > i ? -Infinity : dot(qi, kj) / Math.sqrt(qi.length) + 0.11 * j)
  );
  const weights = raw.map(row => {
    const finite = row.map(x => Number.isFinite(x) ? x : -1e9);
    return softmax(finite);
  });
  const last = weights.length - 1;
  const context = [0, 1, 2, 3].map(dim =>
    v.reduce((sum, vi, i) => sum + weights[last][i] * vi[dim], 0)
  );

  // An intentionally small, transparent attention-weighted state-feedback head.
  // Positive means push right; negative means push left.
  const actionScore =
    0.35 * context[0] +
    0.85 * context[1] +
    8.2 * context[2] +
    1.75 * context[3];

  return { tokens, q, k, v, raw, weights, context, actionScore };
}

export function forceFromScore(score) {
  return 10 * Math.tanh(score);
}
