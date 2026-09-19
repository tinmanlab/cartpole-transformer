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
function normalizeState(history, scale) {
  return history.map(row => row.map((v, i) => v / scale[i]));
}
function visualWithDelta(patchHistory) {
  const delta = patchHistory.map((row, i) =>
    i === 0 ? row.map(() => 0) : row.map((value, j) => value - patchHistory[i - 1][j])
  );
  return {
    delta,
    combined: patchHistory.map((row, i) => [...row, ...delta[i]]),
  };
}
function partialVisionMask(vector, patchDim) {
  const grid = Math.round(Math.sqrt(patchDim));
  const out = [...vector];
  for (let gy = 0; gy < grid; gy++) {
    for (let gx = Math.floor(grid / 2); gx < grid; gx++) {
      const idx = gy * grid + gx;
      out[idx] = 0;
      out[patchDim + idx] = 0;
    }
  }
  return out;
}

export async function loadFusionModel() {
  const url = import.meta.env.BASE_URL + 'model/fusion-transformer.json';
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error('fusion model artifact unavailable');
  const model = await response.json();
  if (model.format !== 'cartpole-fusion-transformer-v1') throw new Error('unsupported fusion model format');
  return model;
}

export function runFusionAttention(stateHistory, patchHistory, model, options = {}) {
  const {
    stateAvailable = true,
    visionAvailable = true,
    stateNoise = 0,
    partialVision = false,
  } = options;

  const w = model.weights;
  const seqLen = model.sequence_length;
  if (stateHistory.length !== seqLen || patchHistory.length !== seqLen) {
    throw new Error('fusion history length mismatch');
  }

  const normalizedStateClean = normalizeState(stateHistory, model.state_scale);
  const noisePattern = [-1, -1 / 3, 1 / 3, 1];
  const normalizedState = normalizedStateClean.map(row =>
    stateAvailable
      ? row.map((v, i) => v + stateNoise * noisePattern[i])
      : row.map(() => 0)
  );

  const patchDim = patchHistory[0].length;
  const visionBase = visualWithDelta(patchHistory);
  const visionCombined = visionBase.combined.map(row => {
    if (!visionAvailable) return row.map(() => 0);
    return partialVision ? partialVisionMask(row, patchDim) : [...row];
  });

  const stateAvail = stateAvailable ? 1 : 0;
  const visionAvail = visionAvailable ? 1 : 0;
  const stateInputs = normalizedState.map(row => [...row, stateAvail]);
  const visionInputs = visionCombined.map(row => [...row, visionAvail]);

  const stateEmbeddings = stateInputs.map(row => matVec(w.state_embed.weight, row, w.state_embed.bias));
  const visionEmbeddings = visionInputs.map(row => matVec(w.vision_embed.weight, row, w.vision_embed.bias));

  const tokens = [];
  const tokenMeta = [];
  for (let t = 0; t < seqLen; t++) {
    tokens.push(add(add(stateEmbeddings[t], w.time_embedding[t]), w.type_embedding[0]));
    tokenMeta.push({ index: tokens.length - 1, time: t, modality: 'state' });
    tokens.push(add(add(visionEmbeddings[t], w.time_embedding[t]), w.type_embedding[1]));
    tokenMeta.push({ index: tokens.length - 1, time: t, modality: 'vision' });
  }

  const norm1 = tokens.map(x => layerNorm(x, w.ln1.weight, w.ln1.bias));
  const q = norm1.map(x => matVec(w.q.weight, x, w.q.bias));
  const k = norm1.map(x => matVec(w.k.weight, x, w.k.bias));
  const v = norm1.map(x => matVec(w.v.weight, x, w.v.bias));
  const d = q[0].length;

  const qkProducts = q.map(qi => k.map(kj => qi.map((x, j) => x * kj[j])));
  const scores = qkProducts.map(row => row.map(products => products.reduce((a, b) => a + b, 0) / Math.sqrt(d)));
  const raw = scores.map((row, r) =>
    row.map((value, c) => tokenMeta[c].time > tokenMeta[r].time ? -Infinity : value)
  );
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

  const finalStateIndex = (seqLen - 1) * 2;
  const finalVisionIndex = finalStateIndex + 1;
  const fusedHidden = hidden[finalStateIndex].map((x, i) => 0.5 * (x + hidden[finalVisionIndex][i]));
  const inferredStateNormalized = matVec(w.state_head.weight, fusedHidden, w.state_head.bias).map(Math.tanh);
  const inferredState = inferredStateNormalized.map((v, i) => v * model.state_scale[i]);
  const actionScore = inferredStateNormalized.reduce((sum, v, i) => sum + v * model.controller_gain[i], 0);

  return {
    modelType: 'learned-state-vision-fusion',
    encoderType: 'typed-state+vision-tokens',
    stateAvailable,
    visionAvailable,
    stateNoise,
    partialVision,
    normalizedStateClean,
    normalizedState,
    visionPatches: patchHistory.map(row => [...row]),
    visionDelta: visionBase.delta,
    visionCombined,
    stateInputs,
    visionInputs,
    stateEmbeddings,
    visionEmbeddings,
    timeEmbeddings: w.time_embedding.map(row => [...row]),
    typeEmbeddings: w.type_embedding.map(row => [...row]),
    tokens,
    tokenMeta,
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
    attended,
    residual1,
    norm2,
    mlpUp,
    mlp,
    hidden,
    finalStateIndex,
    finalVisionIndex,
    fusedHidden,
    inferredStateNormalized,
    inferredState,
    actionScore,
    modelWeights: w,
  };
}
