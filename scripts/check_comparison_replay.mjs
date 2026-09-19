import fs from 'node:fs';
import {
  COMPARISON_HORIZON_STEPS,
  COMPARISON_INITIAL_STATE,
  COMPARISON_DISTURBANCE_PULSES,
  createComparisonRun,
  stepComparison,
  runComparisonToEnd,
  comparisonSummary
} from '../src/lib/comparison.js';

function read(name) {
  return JSON.parse(fs.readFileSync(new URL('../public/model/'+name, import.meta.url), 'utf8'));
}
const models={
  state:read('tiny-transformer.json'),
  vision:read('vision-transformer.json'),
  fusion:read('fusion-transformer.json')
};
const assert=(ok,msg)=>{if(!ok) throw new Error(msg);};

const initial=createComparisonRun(models);
const initialStates=Object.values(initial.trace[0].controllers).map(x=>JSON.stringify(x.state));
assert(new Set(initialStates).size===1,'comparison initial controller states differ');
assert(initialStates[0]===JSON.stringify(COMPARISON_INITIAL_STATE),'comparison initial state contract mismatch');

for(let i=0;i<120;i++) stepComparison(initial);
for(const snap of initial.trace){
  const disturbances=Object.values(snap.controllers).map(x=>x.disturbance);
  assert(new Set(disturbances).size===1,'controllers received different disturbance at tick '+snap.tick);
  assert(disturbances[0]===snap.disturbance,'snapshot disturbance mismatch at tick '+snap.tick);
}

const a=runComparisonToEnd(models);
const b=runComparisonToEnd(models);
assert(a.trace.length===COMPARISON_HORIZON_STEPS+1,'comparison trace length mismatch');
assert(JSON.stringify(a.trace)===JSON.stringify(b.trace),'comparison replay is not deterministic');

const summary=comparisonSummary(a);
for(const name of ['state','vision','fusion']){
  const s=summary[name];
  assert(Number.isFinite(s.survivalSeconds),'non-finite survival for '+name);
  assert(Number.isFinite(s.maxAbsTheta),'non-finite max angle for '+name);
  assert(Number.isFinite(s.controlEffort),'non-finite control effort for '+name);

  if(s.failed){
    const failTick=s.survivalSteps;
    const frozen=JSON.stringify(a.trace[failTick].controllers[name].state);
    for(let tick=failTick+1;tick<a.trace.length;tick++){
      assert(JSON.stringify(a.trace[tick].controllers[name].state)===frozen,name+' state changed after failure at tick '+tick);
      assert(a.trace[tick].controllers[name].force===0,name+' force is nonzero after failure at tick '+tick);
    }
  }
}

assert(summary.vision.failed===true,'deterministic comparison no longer exercises failed-controller freeze path');

for(const pulse of COMPARISON_DISTURBANCE_PULSES){
  const snap=a.trace[pulse.start+1];
  assert(Math.abs(snap.disturbance-pulse.force)<1e-12,'scheduled disturbance missing at '+pulse.start);
}

console.log('comparison replay check passed',{
  horizon:COMPARISON_HORIZON_STEPS,
  pulses:COMPARISON_DISTURBANCE_PULSES,
  summary
});
