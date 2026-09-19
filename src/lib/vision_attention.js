const EPS = 1e-5;

function matVec(weight, vector, bias = null) {
  return weight.map((row, i) => row.reduce((sum, w, j) => sum + w * vector[j], bias ? bias[i] : 0));
}
function add(a, b) {
  return a.map((x, i) => x + b[i]);
}
function layerNorm(vector, weight, bias) {
  const mean = vector.reduce((a, b) => a + b, 0) / vector.length;
  const variance = vector.reduce((s, x) => s + (x - mean) ** 2, 0) / vector.length;
  const inv = 1 / Math.sqrt(variance + EPS);
  return vector.map((x, i) => ((x - mean) * inv) * weight[i] + bias[i]);
}
function geluTanh(x) {
  const k = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(k * (x + 0.044715 * x * x * x)));
}
function softmaxDetails(row) {
  const finite = row.map(x => Number.isFinite(x) ? x : -1e9);
  const max = Math.max(...finite);
  const shifted = finite.map(x => x - max);
  const exp = shifted.map(x => Math.exp(x));
  const denominator = exp.reduce((a, b) => a + b, 0);
  const weights = exp.map(x => x / denominator);
  return { max, shifted, exp, denominator, weights };
}

export async function loadVisionModel() {
  const url = import.meta.env.BASE_URL + 'model/vision-transformer.json';
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error('vision model artifact unavailable');
  const model = await response.json();
  if (model.format !== 'cartpole-vision-transformer-v1') throw new Error('unsupported vision model format');
  return model;
}

export function runVisionAttention(patchHistory, model) {
  const w = model.weights;
  const frameFeatures = patchHistory.map(row => [...row]);
  const deltaFeatures = frameFeatures.map((row, i) =>
    i === 0 ? row.map(() => 0) : row.map((value, j) => value - frameFeatures[i - 1][j])
  );
  const tokenInputs = frameFeatures.map((row, i) => [...row, ...deltaFeatures[i]]);
  const frameEmbeddings = tokenInputs.map(features => matVec(w.embed.weight, features, w.embed.bias));
  const positionEmbeddings = w.pos_embedding.map(row => [...row]);
  const tokens = frameEmbeddings.map((token, i) => add(token, positionEmbeddings[i]));

  const norm1 = tokens.map(x => layerNorm(x, w.ln1.weight, w.ln1.bias));
  const q = norm1.map(x => matVec(w.q.weight, x, w.q.bias));
  const k = norm1.map(x => matVec(w.k.weight, x, w.k.bias));
  const v = norm1.map(x => matVec(w.v.weight, x, w.v.bias));
  const d = q[0].length;

  const qkProducts = q.map(qi => k.map(kj => qi.map((x, j) => x * kj[j])));
  const scores = qkProducts.map(row => row.map(products => products.reduce((a, b) => a + b, 0) / Math.sqrt(d)));
  const raw = scores.map((row, r) => row.map((value, c) => c > r ? -Infinity : value));
  const softmaxDetail = raw.map(softmaxDetails);
  const weights = softmaxDetail.map(detail => detail.weights);
  const weightedValueContributions = weights.map(row =>
    row.map((weight, i) => v[i].map(value => weight * value))
  );
  const perTokenContext = weightedValueContributions.map(contributions =>
    Array.from({ length: d }, (_, dim) => contributions.reduce((sum, vector) => sum + vector[dim], 0))
  );

  const attended = perTokenContext.map(x => matVec(w.o.weight, x, w.o.bias));
  const residual1 = tokens.map((x, i) => add(x, attended[i]));
  const norm2 = residual1.map(x => layerNorm(x, w.ln2.weight, w.ln2.bias));
  const mlpUp = norm2.map(x => matVec(w.ff1.weight, x, w.ff1.bias).map(geluTanh));
  const mlp = mlpUp.map(x => matVec(w.ff2.weight, x, w.ff2.bias));
  const hidden = residual1.map((x, i) => add(x, mlp[i]));
  const last = hidden.length - 1;
  const inferredStateNormalized = matVec(w.state.weight, hidden[last], w.state.bias).map(Math.tanh);
  const actionScore = inferredStateNormalized.reduce((sum, value, i) => sum + value * model.controller_gain[i], 0);
  const inferredState = inferredStateNormalized.map((value, i) => value * model.state_scale[i]);
  const inferredMotion = [inferredState[1], inferredState[3]];

  return {
    modelType: 'learned-vision-transformer',
    encoderType: 'pixels->16x16-patches+delta->linear',
    frameFeatures,
    deltaFeatures,
    tokenInputs,
    frameEmbeddings,
    positionEmbeddings,
    tokens,
    norm1,
    q,
    k,
    v,
    qkProducts,
    scores,
    raw,
    softmaxMax: softmaxDetail.map(detail => detail.max),
    softmaxShifted: softmaxDetail.map(detail => detail.shifted),
    softmaxExp: softmaxDetail.map(detail => detail.exp),
    softmaxDenominators: softmaxDetail.map(detail => detail.denominator),
    weights,
    weightedValueContributions,
    perTokenContext,
    context: perTokenContext[last],
    attended,
    residual1,
    norm2,
    mlpUp,
    mlp,
    hidden,
    inferredState,
    inferredStateNormalized,
    inferredMotion,
    modelWeights: w,
    actionScore,
  };
}
