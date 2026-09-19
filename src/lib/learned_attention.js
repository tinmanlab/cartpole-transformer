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

function softmax(row) {
  const finite = row.map(x => Number.isFinite(x) ? x : -1e9);
  const max = Math.max(...finite);
  const exp = finite.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
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
  const embedded = normalizedTokens.map((token, i) =>
    add(matVec(w.embed.weight, token, w.embed.bias), w.pos_embedding[i])
  );

  const norm1 = embedded.map(x => layerNorm(x, w.ln1.weight, w.ln1.bias));
  const q = norm1.map(x => matVec(w.q.weight, x, w.q.bias));
  const k = norm1.map(x => matVec(w.k.weight, x, w.k.bias));
  const v = norm1.map(x => matVec(w.v.weight, x, w.v.bias));
  const d = q[0].length;

  const raw = q.map((qi, row) =>
    k.map((kj, col) => col > row
      ? -Infinity
      : qi.reduce((sum, x, j) => sum + x * kj[j], 0) / Math.sqrt(d))
  );
  const weights = raw.map(softmax);
  const perTokenContext = weights.map(row =>
    Array.from({ length: d }, (_, dim) =>
      v.reduce((sum, vi, i) => sum + row[i] * vi[dim], 0)
    )
  );

  const attended = perTokenContext.map(x => matVec(w.o.weight, x, w.o.bias));
  const residual1 = embedded.map((x, i) => add(x, attended[i]));
  const norm2 = residual1.map(x => layerNorm(x, w.ln2.weight, w.ln2.bias));
  const mlp = norm2.map(x => {
    const up = matVec(w.ff1.weight, x, w.ff1.bias).map(geluTanh);
    return matVec(w.ff2.weight, up, w.ff2.bias);
  });
  const hidden = residual1.map((x, i) => add(x, mlp[i]));
  const last = hidden.length - 1;
  const actionScore = matVec(w.action.weight, hidden[last], w.action.bias)[0];

  return {
    modelType: 'learned-tiny-transformer',
    encoderType: 'learned-linear+position',
    rawTokens,
    normalizedTokens,
    tokens: embedded,
    q,
    k,
    v,
    raw,
    weights,
    context: perTokenContext[last],
    hidden,
    actionScore
  };
}
