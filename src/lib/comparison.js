import { PHYSICS, stepCartPole, terminal, stateArray } from './physics.js';
import { forceFromScore } from './attention.js';
import { runLearnedAttention } from './learned_attention.js';
import {
  visionObservationFromState,
  sampleVisionHistory,
  VISION_BUFFER_LENGTH,
  VISION_SEQUENCE_LENGTH
} from './vision.js';
import { runVisionAttention } from './vision_attention.js';
import { runFusionAttention } from './fusion_attention.js';

export const COMPARISON_HORIZON_STEPS = 500;
export const COMPARISON_INITIAL_STATE = Object.freeze({
  x: 0,
  xDot: 0,
  theta: 0.08,
  thetaDot: 0
});

export const COMPARISON_DISTURBANCE_PULSES = Object.freeze([
  { start: 80, duration: 8, force: 4 },
  { start: 190, duration: 8, force: -4 },
  { start: 300, duration: 8, force: 4 },
  { start: 410, duration: 8, force: -4 }
]);

const CONTROLLERS = ['state', 'vision', 'fusion'];

function cloneState(s) {
  return { x:s.x, xDot:s.xDot, theta:s.theta, thetaDot:s.thetaDot };
}

function disturbanceAtStep(step) {
  for (const pulse of COMPARISON_DISTURBANCE_PULSES) {
    if (step >= pulse.start && step < pulse.start + pulse.duration) return pulse.force;
  }
  return 0;
}

export function comparisonDisturbanceAtStep(step) {
  return disturbanceAtStep(step);
}

function makeObservation(state) {
  const raw = stateArray(state);
  const obs = visionObservationFromState(raw);
  return { frame:[...obs.frame], patches:[...obs.patches], state:[...raw] };
}

function makeControllerRuntime() {
  const state = cloneState(COMPARISON_INITIAL_STATE);
  const raw = stateArray(state);
  const observation = makeObservation(state);
  return {
    state,
    stateHistory:Array.from({length:8},()=>[...raw]),
    visionBuffer:Array.from({length:VISION_BUFFER_LENGTH},()=>({
      frame:[...observation.frame],
      patches:[...observation.patches],
      state:[...observation.state]
    })),
    force:0,
    failed:false,
    failedAt:null,
    metrics:{
      maxAbsTheta:Math.abs(state.theta),
      maxAbsX:Math.abs(state.x),
      controlEffort:0,
      sumAbsTheta:Math.abs(state.theta),
      samples:1
    }
  };
}

function computeForce(name, runtime, models) {
  if (runtime.failed) return 0;

  if (name === 'state') {
    const result = runLearnedAttention(runtime.stateHistory, models.state);
    return forceFromScore(result.actionScore);
  }

  const samples = sampleVisionHistory(runtime.visionBuffer);
  const patchHistory = samples.map(x=>x.patches);

  if (name === 'vision') {
    const result = runVisionAttention(patchHistory, models.vision);
    return forceFromScore(result.actionScore);
  }

  const alignedStateHistory = samples.map(x=>x.state);
  const result = runFusionAttention(alignedStateHistory, patchHistory, models.fusion);
  return forceFromScore(result.actionScore);
}

function updateHistory(runtime) {
  const raw = stateArray(runtime.state);
  runtime.stateHistory = [...runtime.stateHistory.slice(1), raw];
  const observation = makeObservation(runtime.state);
  runtime.visionBuffer = [...runtime.visionBuffer.slice(1), observation];
}

function updateMetrics(runtime) {
  const m = runtime.metrics;
  m.maxAbsTheta = Math.max(m.maxAbsTheta, Math.abs(runtime.state.theta));
  m.maxAbsX = Math.max(m.maxAbsX, Math.abs(runtime.state.x));
  m.sumAbsTheta += Math.abs(runtime.state.theta);
  m.samples += 1;
  if (!runtime.failed) m.controlEffort += Math.abs(runtime.force) * PHYSICS.tau;
}

function snapshotController(runtime, sharedDisturbance) {
  const m = runtime.metrics;
  return {
    state:cloneState(runtime.state),
    force:runtime.force,
    disturbance:sharedDisturbance,
    failed:runtime.failed,
    failedAt:runtime.failedAt,
    metrics:{
      maxAbsTheta:m.maxAbsTheta,
      maxAbsX:m.maxAbsX,
      controlEffort:m.controlEffort,
      meanAbsTheta:m.sumAbsTheta / m.samples
    }
  };
}

function makeSnapshot(run, sharedDisturbance) {
  return {
    tick:run.tick,
    time:run.tick * PHYSICS.tau,
    disturbance:sharedDisturbance,
    controllers:Object.fromEntries(
      CONTROLLERS.map(name=>[name,snapshotController(run.controllers[name],sharedDisturbance)])
    )
  };
}

export function createComparisonRun(models) {
  if (!models?.state || !models?.vision || !models?.fusion) {
    throw new Error('comparison requires learned state, vision and fusion models');
  }
  const run = {
    models,
    tick:0,
    done:false,
    controllers:Object.fromEntries(CONTROLLERS.map(name=>[name,makeControllerRuntime()])),
    trace:[]
  };

  // Record one common initial frame before any controller can diverge.
  run.trace.push(makeSnapshot(run, disturbanceAtStep(0)));
  return run;
}

export function stepComparison(run) {
  if (run.done) return run.trace[run.trace.length - 1];
  if (run.tick >= COMPARISON_HORIZON_STEPS) {
    run.done = true;
    return run.trace[run.trace.length - 1];
  }

  const disturbance = disturbanceAtStep(run.tick);

  for (const name of CONTROLLERS) {
    const runtime = run.controllers[name];
    if (runtime.failed) {
      runtime.force = 0;
      continue;
    }

    runtime.force = computeForce(name, runtime, run.models);
    runtime.metrics.controlEffort += Math.abs(runtime.force) * PHYSICS.tau;
    runtime.state = stepCartPole(runtime.state, runtime.force, PHYSICS.tau, disturbance);

    if (terminal(runtime.state)) {
      runtime.failed = true;
      runtime.failedAt = run.tick + 1;
    }

    updateHistory(runtime);

    const m = runtime.metrics;
    m.maxAbsTheta = Math.max(m.maxAbsTheta, Math.abs(runtime.state.theta));
    m.maxAbsX = Math.max(m.maxAbsX, Math.abs(runtime.state.x));
    m.sumAbsTheta += Math.abs(runtime.state.theta);
    m.samples += 1;
  }

  run.tick += 1;
  if (run.tick >= COMPARISON_HORIZON_STEPS) run.done = true;

  const snapshot = makeSnapshot(run, disturbance);
  run.trace.push(snapshot);
  return snapshot;
}

export function runComparisonToEnd(models) {
  const run = createComparisonRun(models);
  while (!run.done) stepComparison(run);
  return run;
}

export function comparisonSummary(run) {
  return Object.fromEntries(CONTROLLERS.map(name=>{
    const r=run.controllers[name];
    return [name,{
      survivalSteps:r.failedAt ?? COMPARISON_HORIZON_STEPS,
      survivalSeconds:(r.failedAt ?? COMPARISON_HORIZON_STEPS) * PHYSICS.tau,
      failed:r.failed,
      maxAbsTheta:r.metrics.maxAbsTheta,
      maxAbsX:r.metrics.maxAbsX,
      meanAbsTheta:r.metrics.sumAbsTheta / r.metrics.samples,
      controlEffort:r.metrics.controlEffort
    }];
  }));
}
