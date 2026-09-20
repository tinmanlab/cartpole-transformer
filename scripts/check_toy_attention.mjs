import { runAttention, forceFromScore, RECENCY_BIAS_PER_STEP, ACTION_HEAD_WEIGHTS } from '../src/lib/attention.js';
import { runLearnedAttention } from '../src/lib/learned_attention.js';

const assert = (ok, message) => {
  if (!ok) throw new Error(message);
};
const close = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// Fixture reproduced from the audited native repro: 8 identical tokens
// [x=0, xDot=0, theta=0.045, thetaDot=0], j=7 gives
// dot=0.015355782548476452, scale=2, pre-bias=0.007677891274238226,
// bias=8.4, score=8.407677891274238.
const history = Array.from({ length: 8 }, () => [0, 0, 0.045, 0]);
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
