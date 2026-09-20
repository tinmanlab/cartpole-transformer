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

export const STATE_FIELDS = ['x', 'xDot', 'theta', 'thetaDot'];
export const NORMALIZATION_SCALE = [2.4, 3.0, 0.38, 3.5];

// This toy fallback has no learned weights: Q/K are fixed 4x4 matrices, V is
// the identity (the normalized token itself), and this constant recency
// prior is added to every score so later keys are favored deterministically.
// It is NOT a learned attention bias -- keep it, don't relabel it as one.
export const RECENCY_BIAS_PER_STEP = 1.20;

// Fixed (not learned) feedback gains applied to the attention context to
// produce the action score: [x, xDot, theta, thetaDot].
export const ACTION_HEAD_WEIGHTS = [1.50, 0.50, 8.00, 3.00];

function normalize(state) {
  return state.map((value, i) => value / NORMALIZATION_SCALE[i]);
}
function matVec(M, v) {
  return M.map(row => row.reduce((sum, w, i) => sum + w * v[i], 0));
}
function dot(a, b) {
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}
function softmaxDetails(xs) {
  const finite = xs.map(x => Number.isFinite(x) ? x : -1e9);
  const max = Math.max(...finite);
  const shifted = finite.map(x => x - max);
  const exp = shifted.map(x => Math.exp(x));
  const denominator = exp.reduce((a, b) => a + b, 0);
  const weights = exp.map(x => x / denominator);
  return { max, shifted, exp, denominator, weights };
}

export function runAttention(history) {
  const rawTokens = history.map(state => [...state]);
  const normalizedTokens = rawTokens.map(normalize);
  const tokens = normalizedTokens;
  const q = tokens.map(v => matVec(WQ, v));
  const k = tokens.map(v => matVec(WK, v));
  const v = tokens.map(t => [...t]);

  const qkProducts = q.map(qi =>
    k.map(kj => qi.map((x, j) => x * kj[j]))
  );
  const preBiasScores = qkProducts.map(row =>
    row.map(products => products.reduce((sum, x) => sum + x, 0) / Math.sqrt(q[0].length))
  );
  const scoreBias = preBiasScores.map(row =>
    row.map((_, j) => RECENCY_BIAS_PER_STEP * j)
  );
  const scores = preBiasScores.map((row, i) =>
    row.map((value, j) => value + scoreBias[i][j])
  );
  const raw = scores.map((row, i) =>
    row.map((value, j) => j > i ? -Infinity : value)
  );
  const softmaxDetail = raw.map(softmaxDetails);
  const weights = softmaxDetail.map(detail => detail.weights);
  const weightedValueContributions = weights.map(row =>
    row.map((weight, i) => v[i].map(value => weight * value))
  );
  const perTokenContext = weightedValueContributions.map(contributions =>
    Array.from({ length: v[0].length }, (_, dim) =>
      contributions.reduce((sum, vector) => sum + vector[dim], 0)
    )
  );
  const last = weights.length - 1;
  const context = perTokenContext[last];

  // An intentionally small, transparent attention-weighted state-feedback head.
  // Positive means push right; negative means push left.
  const actionScore = dot(ACTION_HEAD_WEIGHTS, context);

  return {
    modelType: 'transparent-toy',
    encoderType: 'scale-only',
    recencyBiasPerStep: RECENCY_BIAS_PER_STEP,
    actionHeadWeights: ACTION_HEAD_WEIGHTS,
    rawTokens,
    normalizedTokens,
    tokens,
    q,
    k,
    v,
    qkProducts,
    preBiasScores,
    scoreBias,
    scores,
    raw,
    softmaxMax: softmaxDetail.map(detail => detail.max),
    softmaxShifted: softmaxDetail.map(detail => detail.shifted),
    softmaxExp: softmaxDetail.map(detail => detail.exp),
    softmaxDenominators: softmaxDetail.map(detail => detail.denominator),
    weights,
    weightedValueContributions,
    perTokenContext,
    context,
    actionScore
  };
}

export function forceFromScore(score) {
  return 10 * Math.tanh(score);
}
