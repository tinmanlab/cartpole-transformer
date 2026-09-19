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
assert(out.modelType === 'learned-tiny-transformer', 'wrong model type');
assert(out.tokens.length === model.sequence_length, 'wrong token count');
assert(out.tokens[0].length === model.d_model, 'wrong embedding width');
assert(out.q[0].length === model.d_model, 'wrong Q width');
assert(out.k[0].length === model.d_model, 'wrong K width');
assert(out.v[0].length === model.d_model, 'wrong V width');
assert(Number.isFinite(out.actionScore), 'non-finite action score');

for (let r = 0; r < model.sequence_length; r++) {
  for (let c = r + 1; c < model.sequence_length; c++) {
    assert(out.raw[r][c] === -Infinity, 'causal mask violation');
  }
  const sum = out.weights[r].reduce((a, b) => a + b, 0);
  assert(Math.abs(sum - 1) < 1e-6, 'softmax row does not sum to one');
}
console.log('learned model runtime check passed', {
  seq: model.sequence_length,
  dModel: model.d_model,
  actionScore: out.actionScore,
});
