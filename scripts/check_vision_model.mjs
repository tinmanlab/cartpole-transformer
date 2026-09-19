import fs from 'node:fs';
import { visionObservationFromState, VISION_FEATURE_DIM, VISION_SEQUENCE_LENGTH } from '../src/lib/vision.js';
import { runVisionAttention } from '../src/lib/vision_attention.js';

const path = new URL('../public/model/vision-transformer.json', import.meta.url);
if (!fs.existsSync(path)) {
  console.log('vision model artifact not present yet; skipping vision model check');
  process.exit(0);
}

const model = JSON.parse(fs.readFileSync(path, 'utf8'));
const assert = (ok, message) => { if (!ok) throw new Error(message); };
const close = (a,b,eps=1e-8) => Math.abs(a-b) <= eps;

assert(model.format === 'cartpole-vision-transformer-v1', 'wrong vision model format');
assert(model.feature_dim === VISION_FEATURE_DIM, 'vision feature dimension mismatch');
assert(model.token_input_dim === VISION_FEATURE_DIM * 2, 'vision token input dimension mismatch');
assert(model.sequence_length === VISION_SEQUENCE_LENGTH, 'vision sequence length mismatch');

const states = Array.from({length:model.sequence_length},(_,i)=>[
  -0.12 + i*0.03,
  0.15 - i*0.01,
  0.045 - i*0.006,
  -0.18 + i*0.025,
]);
const observations = states.map(s=>visionObservationFromState(s));
const history = observations.map(o=>o.patches);
for (const o of observations) {
  assert(o.frame.length === model.frame_size * model.frame_size, 'wrong rendered frame size');
  assert(o.patches.length === model.feature_dim, 'wrong patch feature width');
}
const out = runVisionAttention(history, model);
assert(out.modelType === 'learned-vision-transformer', 'wrong runtime vision model type');
assert(out.deltaFeatures.length === model.sequence_length, 'missing delta patch history');
assert(out.tokenInputs[0].length === model.token_input_dim, 'wrong visual token input width');
assert(out.tokens.length === model.sequence_length, 'wrong visual token count');
assert(out.tokens[0].length === model.d_model, 'wrong visual token width');
assert(out.inferredState.length === 4, 'missing inferred visual state');
assert(out.inferredMotion.length === 2, 'missing inferred motion');
assert(Number.isFinite(out.actionScore), 'non-finite visual action score');

for (let r=0;r<model.sequence_length;r++) {
  for (let c=r+1;c<model.sequence_length;c++) {
    assert(out.raw[r][c] === -Infinity, 'vision causal mask violation');
    assert(out.weights[r][c] === 0, 'vision future weight nonzero');
  }
  const sum=out.weights[r].reduce((a,b)=>a+b,0);
  assert(Math.abs(sum-1)<1e-6,'vision softmax row does not sum to one');
}

const expectedScore = out.inferredStateNormalized.reduce((sum,v,i)=>sum+v*model.controller_gain[i],0);
assert(close(expectedScore,out.actionScore),'visual inferred-state action equation mismatch');

assert(model.closed_loop.full_history.mean_steps >= 320, 'stored visual closed-loop acceptance below threshold');
assert(
  model.ablation.xdot_mae_full < model.ablation.xdot_mae_repeat_latest * 0.90,
  'stored xdot temporal ablation advantage missing'
);
assert(
  model.ablation.thetadot_mae_full < model.ablation.thetadot_mae_repeat_latest * 0.90,
  'stored thetadot temporal ablation advantage missing'
);

for(let i=0;i<50;i++) runVisionAttention(history,model);
const runs=500;
const start=performance.now();
for(let i=0;i<runs;i++) runVisionAttention(history,model);
const averageMs=(performance.now()-start)/runs;
assert(averageMs<12,'vision inference unexpectedly slow: '+averageMs.toFixed(3)+' ms');

console.log('vision model runtime check passed',{
  featureDim:model.feature_dim,
  dModel:model.d_model,
  closedLoopMean:model.closed_loop.full_history.mean_steps,
  repeatedMean:model.closed_loop.repeat_latest_frame.mean_steps,
  xdotHistoryRatio:model.ablation.xdot_mae_repeat_latest/model.ablation.xdot_mae_full,
  thetadotHistoryRatio:model.ablation.thetadot_mae_repeat_latest/model.ablation.thetadot_mae_full,
  averageInferenceMs:Number(averageMs.toFixed(4)),
});
