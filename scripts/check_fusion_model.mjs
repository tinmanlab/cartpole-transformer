import fs from 'node:fs';
import { visionObservationFromState } from '../src/lib/vision.js';
import { runFusionAttention } from '../src/lib/fusion_attention.js';

const path = new URL('../public/model/fusion-transformer.json', import.meta.url);
if (!fs.existsSync(path)) {
  console.log('fusion model artifact not present yet; skipping fusion model check');
  process.exit(0);
}

const model = JSON.parse(fs.readFileSync(path, 'utf8'));
const assert = (ok, message) => { if (!ok) throw new Error(message); };
const close = (a,b,eps=1e-8) => Math.abs(a-b) <= eps;

assert(model.format === 'cartpole-fusion-transformer-v1', 'wrong fusion model format');
assert(model.sequence_length === 8, 'unexpected fusion sequence length');
assert(model.token_count === 16, 'fusion must expose 16 typed tokens');

const states = Array.from({length:model.sequence_length},(_,i)=>[
  -0.10 + i*0.025,
  0.16 - i*0.012,
  0.05 - i*0.007,
  -0.17 + i*0.022,
]);
const patches = states.map(s=>visionObservationFromState(s).patches);
const out = runFusionAttention(states, patches, model);

assert(out.tokens.length === model.token_count, 'wrong fusion token count');
assert(out.tokenMeta.length === model.token_count, 'missing fusion token metadata');
assert(out.tokens[0].length === model.d_model, 'wrong fusion token width');
assert(out.inferredState.length === 4, 'missing fusion inferred state');
assert(Number.isFinite(out.actionScore), 'non-finite fusion action score');

for (let q=0;q<model.token_count;q++) {
  for (let k=0;k<model.token_count;k++) {
    const qt=out.tokenMeta[q].time;
    const kt=out.tokenMeta[k].time;
    if (kt > qt) {
      assert(out.raw[q][k] === -Infinity, 'future fusion timestamp not masked');
      assert(out.weights[q][k] === 0, 'future fusion timestamp has nonzero weight');
    } else {
      assert(Number.isFinite(out.raw[q][k]), 'allowed fusion timestamp unexpectedly masked');
    }
  }
  const sum=out.weights[q].reduce((a,b)=>a+b,0);
  assert(Math.abs(sum-1)<1e-6,'fusion softmax row does not sum to one');
}

for (let t=0;t<model.sequence_length;t++) {
  const s=t*2, v=s+1;
  assert(Number.isFinite(out.raw[s][v]), 'same-time state->vision unexpectedly masked');
  assert(Number.isFinite(out.raw[v][s]), 'same-time vision->state unexpectedly masked');
}

const expectedScore=out.inferredStateNormalized.reduce((sum,v,i)=>sum+v*model.controller_gain[i],0);
assert(close(expectedScore,out.actionScore),'fusion inferred-state action equation mismatch');

const missingState=runFusionAttention(states,patches,model,{stateAvailable:false});
const missingVision=runFusionAttention(states,patches,model,{visionAvailable:false});
const noisy=runFusionAttention(states,patches,model,{stateNoise:0.18});
const partial=runFusionAttention(states,patches,model,{partialVision:true});
for (const x of [missingState,missingVision,noisy,partial]) {
  assert(Number.isFinite(x.actionScore),'fusion degradation produced non-finite action');
}
assert(missingState.stateAvailable===false,'state availability control failed');
assert(missingVision.visionAvailable===false,'vision availability control failed');
assert(noisy.stateNoise===0.18,'state noise control failed');
assert(partial.partialVision===true,'partial vision control failed');

const cl=model.closed_loop;
assert(cl.clean.mean_steps>=420,'fusion clean closed-loop acceptance below threshold');
assert(cl.noisy_state_plus_vision.mean_steps>cl.noisy_state.mean_steps*1.20,'fusion noisy-state resilience advantage missing');
assert(cl.partial_vision_plus_state.mean_steps>cl.partial_vision.mean_steps*1.20,'fusion partial-vision resilience advantage missing');
assert(model.ablation.noisy_state_plus_vision.state_mae<model.ablation.noisy_state.state_mae,'fusion noisy-state MAE advantage missing');
assert(model.ablation.partial_vision_plus_state.state_mae<model.ablation.partial_vision.state_mae,'fusion partial-vision MAE advantage missing');

for(let i=0;i<30;i++) runFusionAttention(states,patches,model);
const runs=300;
const start=performance.now();
for(let i=0;i<runs;i++) runFusionAttention(states,patches,model);
const averageMs=(performance.now()-start)/runs;
assert(averageMs<15,'fusion browser inference unexpectedly slow: '+averageMs.toFixed(3)+' ms');

console.log('fusion model runtime check passed',{
  cleanMean:cl.clean.mean_steps,
  noisyStateOnly:cl.noisy_state.mean_steps,
  noisyStatePlusVision:cl.noisy_state_plus_vision.mean_steps,
  partialVisionOnly:cl.partial_vision.mean_steps,
  partialVisionPlusState:cl.partial_vision_plus_state.mean_steps,
  averageInferenceMs:Number(averageMs.toFixed(4)),
});