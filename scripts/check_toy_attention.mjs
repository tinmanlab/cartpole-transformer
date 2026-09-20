import { runAttention, forceFromScore, RECENCY_BIAS_PER_STEP, ACTION_HEAD_WEIGHTS } from '../src/lib/attention.js';
import { runLearnedAttention } from '../src/lib/learned_attention.js';

const assert = (ok, message) => {
  if (!ok) throw new Error(message);
};
const close = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// Independent re-implementation of the PRE-FIX toy fallback exactly as it
// shipped (git 76c0b84, src/lib/attention.js before this branch): a single
// opaque `scores = dot(Q,K)/sqrt(d) + 1.20*j`, no preBiasScores/scoreBias
// decomposition existed at all. Kept deliberately separate from
// src/lib/attention.js (own WQ/WK/normalize copies) so this is a true
// external baseline, not a reflection of the code under test.
const NATIVE_WQ = [
  [0.10, 0.05, 1.00, 0.42],
  [0.00, 0.12, 0.35, 0.95],
  [0.10, 0.25, 0.18, 0.35],
  [0.25, 0.10, 0.08, 0.10],
];
const NATIVE_WK = [
  [0.08, 0.05, 0.95, 0.40],
  [0.00, 0.10, 0.30, 0.92],
  [0.08, 0.24, 0.20, 0.32],
  [0.24, 0.08, 0.05, 0.12],
];
const NATIVE_SCALE = [2.4, 3.0, 0.38, 3.5];
const NATIVE_ACTION_WEIGHTS = [1.50, 0.50, 8.00, 3.00];

function nativeMatVec(M, v) {
  return M.map(row => row.reduce((sum, w, i) => sum + w * v[i], 0));
}
function nativeSoftmaxRow(xs) {
  const finite = xs.map(x => Number.isFinite(x) ? x : -1e9);
  const max = Math.max(...finite);
  const exp = finite.map(x => Math.exp(x - max));
  const denom = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / denom);
}

// This function has NO decomposition: it returns only `scores` (already
// biased) and `weights`/`actionScore`, exactly the shape the pre-fix code
// exposed to the UI -- there is no separate pre-bias/bias field to read,
// which is precisely the fabrication bug F2 fixed (AttentionCellTrace had
// no source of truth for "dot/scale" alone).
function nativePreFixToyAttention(history) {
  const tokens = history.map(state => state.map((x, i) => x / NATIVE_SCALE[i]));
  const q = tokens.map(v => nativeMatVec(NATIVE_WQ, v));
  const k = tokens.map(v => nativeMatVec(NATIVE_WK, v));
  const v = tokens.map(t => [...t]);
  const scores = q.map((qi, i) =>
    k.map((kj, j) => {
      const dotSum = qi.reduce((sum, x, d) => sum + x * kj[d], 0);
      return dotSum / Math.sqrt(qi.length) + 1.20 * j;
    })
  );
  const raw = scores.map((row, i) => row.map((val, j) => (j > i ? -Infinity : val)));
  const weights = raw.map(nativeSoftmaxRow);
  const last = weights.length - 1;
  const context = Array.from({ length: v[0].length }, (_, d) =>
    weights[last].reduce((sum, w, j) => sum + w * v[j][d], 0)
  );
  const actionScore = context.reduce((sum, x, d) => sum + x * NATIVE_ACTION_WEIGHTS[d], 0);
  return { scores, weights, actionScore, force: forceFromScore(actionScore) };
}

function assertMatrixClose(a, b, label, eps = 1e-9) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      assert(close(a[i][j], b[i][j], eps), `${label} mismatch at [${i}][${j}]: ${a[i][j]} vs ${b[i][j]}`);
    }
  }
}

// Compare src/lib/attention.js against the independent pre-fix native
// baseline across several varied histories (not just one fixture): full
// scores matrix, full weights matrix, actionScore and force must all match
// exactly. This is what "byte-identical" actually means -- not one scalar.
const varietyHistories = [
  Array.from({ length: 8 }, () => [0, 0, 0.045, 0]),
  Array.from({ length: 8 }, (_, i) => [0.01 * i, -0.02 + 0.005 * i, 0.03 - 0.003 * i, 0.04 - 0.002 * i]),
  Array.from({ length: 8 }, (_, i) => [-0.5 + 0.1 * i, 0.2 * Math.sin(i), -0.1 * i * i * 0.01, 0.05 - 0.01 * i]),
];
for (const [idx, hist] of varietyHistories.entries()) {
  const fixed = runAttention(hist);
  const native = nativePreFixToyAttention(hist);
  assertMatrixClose(fixed.scores, native.scores, `variety[${idx}].scores`);
  assertMatrixClose(fixed.weights, native.weights, `variety[${idx}].weights`, 1e-8);
  assert(close(fixed.actionScore, native.actionScore, 1e-8), `variety[${idx}] actionScore mismatch: ${fixed.actionScore} vs ${native.actionScore}`);
  assert(close(forceFromScore(fixed.actionScore), native.force, 1e-8), `variety[${idx}] force mismatch`);

  // Demonstrate the actual bug F2 fixed: the pre-fix baseline structurally
  // has no way to decompose `scores` back into dot/scale vs. the fixed
  // bias -- there was no preBiasScores/scoreBias field to read, so a UI
  // built against it (AttentionCellTrace) could only ever display the
  // already-biased number labeled as if it were the raw dot/scale score.
  assert(native.preBiasScores === undefined, `variety[${idx}]: native baseline unexpectedly exposes preBiasScores (should not)`);
  assert(native.scoreBias === undefined, `variety[${idx}]: native baseline unexpectedly exposes scoreBias (should not)`);
  assert(Array.isArray(fixed.preBiasScores) && Array.isArray(fixed.scoreBias), `variety[${idx}]: current code must expose the decomposition the native baseline never had`);
}

// Fixture reproduced from the audited native repro: 8 identical tokens
// [x=0, xDot=0, theta=0.045, thetaDot=0], j=7 gives
// dot=0.015355782548476452, scale=2, pre-bias=0.007677891274238226,
// bias=8.4, score=8.407677891274238.
const history = varietyHistories[0];
const out = runAttention(history);

assert(out.modelType === 'transparent-toy', 'wrong model type');
assert(out.recencyBiasPerStep === RECENCY_BIAS_PER_STEP, 'recency bias metadata not exposed on result');
assert(Array.isArray(out.actionHeadWeights) && out.actionHeadWeights.length === 4, 'action head weights metadata not exposed on result');

// F2: the toy fallback must keep the explicit recency-prior bias, and the
// AttentionCellTrace-consumed preBiasScores/scoreBias must decompose exactly
// into the actual scores used for masking/softmax/action -- no silent
// zero-bias or fabricated "learned" score.
for (let i = 0; i < 8; i++) {
  for (let j = 0; j <= i; j++) {
    const dotSum = out.qkProducts[i][j].reduce((a, b) => a + b, 0);
    const scale = Math.sqrt(out.q[0].length);
    const expectedPreBias = dotSum / scale;
    assert(close(expectedPreBias, out.preBiasScores[i][j]), `preBiasScore mismatch at [${i}][${j}]`);
    assert(close(RECENCY_BIAS_PER_STEP * j, out.scoreBias[i][j]), `scoreBias mismatch at [${i}][${j}]`);
    assert(close(out.preBiasScores[i][j] + out.scoreBias[i][j], out.scores[i][j]), `score does not equal preBias+bias at [${i}][${j}]`);
  }
}

// Exact values from the audited repro (row 7, col 7): dot, pre-bias score and
// full biased score must match to the reported precision.
{
  const r = 7, c = 7;
  const dotSum = out.qkProducts[r][c].reduce((a, b) => a + b, 0);
  assert(close(dotSum, 0.015355782548476452, 1e-15), 'dot product regressed: ' + dotSum);
  assert(close(out.preBiasScores[r][c], 0.007677891274238226, 1e-15), 'pre-bias score regressed: ' + out.preBiasScores[r][c]);
  assert(close(out.scoreBias[r][c], 8.4, 1e-12), 'explicit bias regressed: ' + out.scoreBias[r][c]);
  assert(close(out.scores[r][c], 8.407677891274238, 1e-12), 'biased score regressed: ' + out.scores[r][c]);
}

// Golden fixture captured from the pre-refactor implementation (same formula,
// dot(Q,K)/sqrt(d) + 1.20*j) -- must be byte-identical after the F2 change,
// since only metadata/exposition changed, not the arithmetic.
{
  const last = 7;
  assert(close(out.actionScore, 0.9473684210526314, 1e-12), 'actionScore regressed: ' + out.actionScore);
  assert(close(forceFromScore(out.actionScore), 7.385893590022653, 1e-9), 'force regressed');
  const expectedContext = [0, 0, 0.11842105263157893, 0];
  out.context.forEach((v, d) => assert(close(v, expectedContext[d], 1e-12), 'context regressed at dim ' + d));
  assert(close(out.weights[last][0], 0.00015714923120686448, 1e-12), 'weights row 7 col 0 regressed');
  assert(close(out.weights[last][last], 0.6988531205266438, 1e-12), 'weights row 7 col 7 regressed');
}

// actionScore must still be the fixed-gain dot product of context, per
// ACTION_HEAD_WEIGHTS (not a learned matrix).
{
  const expected = out.context.reduce((sum, x, i) => sum + x * ACTION_HEAD_WEIGHTS[i], 0);
  assert(close(expected, out.actionScore, 1e-12), 'actionScore does not match fixed-gain dot(context)');
}

// Causal mask / softmax invariants (unchanged contract from the learned model).
for (let r = 0; r < 8; r++) {
  for (let c = r + 1; c < 8; c++) {
    assert(out.raw[r][c] === -Infinity, 'causal mask violation');
    assert(out.weights[r][c] === 0, 'future attention weight is not zero');
  }
  const sum = out.weights[r].reduce((a, b) => a + b, 0);
  assert(Math.abs(sum - 1) < 1e-6, 'softmax row does not sum to one');
}

// The learned model must show zero-bias behavior: no scoreBias/preBiasScores
// fields at all, so the shared AttentionCellTrace UI takes the "no fixed
// bias" branch and never fabricates a bias term for a real learned model.
{
  const model = {
    format: 'cartpole-tiny-transformer-v1',
    d_model: 8,
    sequence_length: 4,
    normalization_scale: [2.4, 3.0, 0.38, 3.5],
    weights: {
      embed: { weight: Array.from({length:8},()=>[0,0,0,0]), bias: Array(8).fill(0) },
      pos_embedding: Array.from({length:4},()=>Array(8).fill(0)),
      ln1: { weight: Array(8).fill(1), bias: Array(8).fill(0) },
      q: { weight: Array.from({length:8},()=>Array(8).fill(0)), bias: Array(8).fill(0) },
      k: { weight: Array.from({length:8},()=>Array(8).fill(0)), bias: Array(8).fill(0) },
      v: { weight: Array.from({length:8},()=>Array(8).fill(0)), bias: Array(8).fill(0) },
      o: { weight: Array.from({length:8},()=>Array(8).fill(0)), bias: Array(8).fill(0) },
      ln2: { weight: Array(8).fill(1), bias: Array(8).fill(0) },
      ff1: { weight: Array.from({length:16},()=>Array(8).fill(0)), bias: Array(16).fill(0) },
      ff2: { weight: Array.from({length:8},()=>Array(16).fill(0)), bias: Array(8).fill(0) },
      action: { weight: [Array(8).fill(0)], bias: [0] },
    },
  };
  const learnedHistory = Array.from({ length: 4 }, () => [0, 0, 0.045, 0]);
  const learnedOut = runLearnedAttention(learnedHistory, model);
  assert(learnedOut.scoreBias === undefined, 'learned model result must not carry a toy scoreBias field');
  assert(learnedOut.preBiasScores === undefined, 'learned model result must not carry a toy preBiasScores field');
}

console.log('toy attention (F2) regression check passed', {
  actionScore: out.actionScore,
  recencyBiasPerStep: out.recencyBiasPerStep,
  sampleBias: out.scoreBias[7][7],
});
