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

function normalize(rawTokens, scale) {
  return rawTokens.map(token => token.map((value, i) => value / scale[i]));
}

export async function loadLearnedModel() {
  const url = import.meta.env.BASE_URL + 'model/tiny-transformer.json';
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error('learned model artifact unavailable');
  const model = await response.json();
  if (model.format !== 'cartpole-tiny-transformer-v1') throw new Error('unsupported learned model format');
  return model;
}

export function runLearnedAttention(history, model) {
  const w = model.weights;
  const rawTokens = history.map(state => [...state]);
  const normalizedTokens = normalize(rawTokens, model.normalization_scale);
  const stateEmbeddings = normalizedTokens.map(token =>
    matVec(w.embed.weight, token, w.embed.bias)
  );
  const positionEmbeddings = w.pos_embedding.map(row => [...row]);
  const embedded = stateEmbeddings.map((token, i) =>
    add(token, positionEmbeddings[i])
  );

  const norm1 = embedded.map(x => layerNorm(x, w.ln1.weight, w.ln1.bias));
  const q = norm1.map(x => matVec(w.q.weight, x, w.q.bias));
  const k = norm1.map(x => matVec(w.k.weight, x, w.k.bias));
  const v = norm1.map(x => matVec(w.v.weight, x, w.v.bias));
  const d = q[0].length;

  const qkProducts = q.map(qi =>
    k.map(kj => qi.map((x, j) => x * kj[j]))
  );
  const scores = qkProducts.map(row =>
    row.map(products => products.reduce((sum, x) => sum + x, 0) / Math.sqrt(d))
  );
  const raw = scores.map((row, r) =>
    row.map((value, c) => c > r ? -Infinity : value)
  );
  const softmaxDetail = raw.map(softmaxDetails);
  const weights = softmaxDetail.map(detail => detail.weights);
  const weightedValueContributions = weights.map(row =>
    row.map((weight, i) => v[i].map(value => weight * value))
  );
  const perTokenContext = weightedValueContributions.map(contributions =>
    Array.from({ length: d }, (_, dim) =>
      contributions.reduce((sum, vector) => sum + vector[dim], 0)
    )
  );

  const attended = perTokenContext.map(x => matVec(w.o.weight, x, w.o.bias));
  const residual1 = embedded.map((x, i) => add(x, attended[i]));
  const norm2 = residual1.map(x => layerNorm(x, w.ln2.weight, w.ln2.bias));
  const mlpUp = norm2.map(x =>
    matVec(w.ff1.weight, x, w.ff1.bias).map(geluTanh)
  );
  const mlp = mlpUp.map(x =>
    matVec(w.ff2.weight, x, w.ff2.bias)
  );
  const hidden = residual1.map((x, i) => add(x, mlp[i]));
  const last = hidden.length - 1;
  const actionScore = matVec(w.action.weight, hidden[last], w.action.bias)[0];

  return {
    modelType: 'learned-tiny-transformer',
    encoderType: 'learned-linear+position',
    rawTokens,
    normalizedTokens,
    stateEmbeddings,
    positionEmbeddings,
    tokens: embedded,
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
    modelWeights: w,
    actionScore
  };
}
