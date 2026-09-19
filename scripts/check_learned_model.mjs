import fs from 'node:fs';
import { runLearnedAttention } from '../src/lib/learned_attention.js';

const path = new URL('../public/model/tiny-transformer.json', import.meta.url);
if (!fs.existsSync(path)) {
  console.log('learned model artifact not present yet; skipping model runtime check');
  process.exit(0);
}

const model = JSON.parse(fs.readFileSync(path, 'utf8'));
const history = Array.from({ length: model.sequence_length }, (_, i) => [
  0.01 * i,
  -0.02 + 0.005 * i,
  0.03 - 0.003 * i,
  0.04 - 0.002 * i,
]);
const out = runLearnedAttention(history, model);

const assert = (ok, message) => {
  if (!ok) throw new Error(message);
};
const close = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

assert(out.modelType === 'learned-tiny-transformer', 'wrong model type');
assert(out.tokens.length === model.sequence_length, 'wrong token count');
assert(out.tokens[0].length === model.d_model, 'wrong embedding width');
assert(out.q[0].length === model.d_model, 'wrong Q width');
assert(out.k[0].length === model.d_model, 'wrong K width');
assert(out.v[0].length === model.d_model, 'wrong V width');
assert(out.qkProducts.length === model.sequence_length, 'wrong qkProducts row count');
assert(out.qkProducts[0][0].length === model.d_model, 'wrong qkProducts depth');
assert(out.softmaxExp.length === model.sequence_length, 'missing softmax exponentials');
assert(out.softmaxDenominators.length === model.sequence_length, 'missing softmax denominators');
assert(out.weightedValueContributions[0][0].length === model.d_model, 'wrong weighted V width');
assert(Number.isFinite(out.actionScore), 'non-finite action score');

for (let r = 0; r < model.sequence_length; r++) {
  for (let c = r + 1; c < model.sequence_length; c++) {
    assert(out.raw[r][c] === -Infinity, 'causal mask violation');
    assert(out.weights[r][c] === 0, 'future attention weight is not zero');
  }
  const sum = out.weights[r].reduce((a, b) => a + b, 0);
  assert(Math.abs(sum - 1) < 1e-6, 'softmax row does not sum to one');
}

// Verify one allowed attention cell all the way from Q/K products to V contribution.
{
  const r = model.sequence_length - 1;
  const c = model.sequence_length - 3;
  const dotSum = out.qkProducts[r][c].reduce((a, b) => a + b, 0);
  const expectedScore = dotSum / Math.sqrt(model.d_model);
  assert(close(expectedScore, out.scores[r][c]), 'QK product sum does not reproduce score');
  assert(close(out.softmaxExp[r][c] / out.softmaxDenominators[r], out.weights[r][c]), 'softmax arithmetic mismatch');
  for (let d = 0; d < model.d_model; d++) {
    assert(
      close(out.weights[r][c] * out.v[c][d], out.weightedValueContributions[r][c][d]),
      'weighted V contribution mismatch at dim ' + d
    );
  }
  for (let d = 0; d < model.d_model; d++) {
    const contextDim = out.weightedValueContributions[r].reduce((sum, vector) => sum + vector[d], 0);
    assert(close(contextDim, out.perTokenContext[r][d]), 'context contribution sum mismatch at dim ' + d);
  }
}

for (let i = 0; i < 100; i++) runLearnedAttention(history, model);
const runs = 2000;
const start = performance.now();
for (let i = 0; i < runs; i++) runLearnedAttention(history, model);
const averageMs = (performance.now() - start) / runs;
assert(averageMs < 10, 'learned inference is unexpectedly slow: ' + averageMs.toFixed(3) + ' ms');

console.log('learned model runtime check passed', {
  seq: model.sequence_length,
  dModel: model.d_model,
  actionScore: out.actionScore,
  traceContract: true,
  averageInferenceMs: Number(averageMs.toFixed(4)),
});
