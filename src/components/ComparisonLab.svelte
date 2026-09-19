<script>
  import { onMount } from 'svelte';
  import {
    COMPARISON_HORIZON_STEPS,
    COMPARISON_DISTURBANCE_PULSES,
    createComparisonRun,
    stepComparison,
    comparisonSummary
  } from '../lib/comparison.js';
  import { PHYSICS } from '../lib/physics.js';

  export let models;

  const names = {
    state: 'State',
    vision: 'Vision',
    fusion: 'Fusion'
  };
  const descriptions = {
    state: 'explicit state history',
    vision: 'pixels + temporal history',
    fusion: 'typed State + Vision'
  };

  let run;
  let snapshot;
  let currentIndex = 0;
  let running = true;
  let raf = 0;
  let previous = 0;
  let accumulator = 0;

  $: trace = run?.trace || [];
  $: snapshot = trace[currentIndex] || trace[trace.length - 1];
  $: summary = run ? comparisonSummary(run) : {};
  $: finished = run?.done || false;
  $: currentDisturbance = snapshot?.disturbance || 0;

  function init() {
    run = createComparisonRun(models);
    currentIndex = 0;
    running = true;
    accumulator = 0;
    previous = performance.now();
  }

  function toggle() {
    if (finished && currentIndex >= trace.length - 1) {
      currentIndex = 0;
      accumulator = 0;
      running = true;
      return;
    }
    running = !running;
  }

  function scrub(event) {
    running = false;
    currentIndex = Number(event.currentTarget.value);
  }

  function latest() {
    running = false;
    currentIndex = trace.length - 1;
  }

  function finishRun() {
    if (!run) return;
    while (!run.done) stepComparison(run);
    currentIndex = run.trace.length - 1;
    running = false;
    run = run;
  }

  function statePose(state) {
    const cx = 160 + state.x * 48;
    const pivotY = 126;
    const len = 72;
    return {
      cx,
      tipX: cx + Math.sin(state.theta) * len,
      tipY: pivotY - Math.cos(state.theta) * len
    };
  }

  function deg(rad){ return rad * 180 / Math.PI; }

  onMount(() => {
    init();
    const frame = now => {
      if (!previous) previous = now;
      const delta = Math.min(.08,(now-previous)/1000);
      previous = now;

      if (running && run) {
        accumulator += delta;
        let guard = 0;
        while (accumulator >= PHYSICS.tau && guard < 4) {
          if (currentIndex < run.trace.length - 1) {
            currentIndex += 1;
          } else if (!run.done) {
            stepComparison(run);
            currentIndex = run.trace.length - 1;
            run = run;
          } else {
            running = false;
            break;
          }
          accumulator -= PHYSICS.tau;
          guard += 1;
        }
      }

      if (run?.done && currentIndex >= run.trace.length - 1 && running) running = false;
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  });
</script>

<section
  class="comparison-lab"
  data-current-tick={snapshot?.tick ?? 0}
  data-trace-length={trace.length}
  data-done={finished}
>
  <header class="compare-head">
    <div>
      <span class="eyebrow">DETERMINISTIC REPLAY · SAME INITIAL STATE · SAME DISTURBANCE</span>
      <h2>State / Vision / Fusion side-by-side</h2>
      <p>세 controller는 서로 다른 physics state를 갖지만 시작값과 외란 tick은 완전히 같습니다.</p>
    </div>
    <div class="compare-clock">
      <b>{(snapshot?.time || 0).toFixed(2)} s</b>
      <span>{snapshot?.tick || 0} / {COMPARISON_HORIZON_STEPS}</span>
    </div>
  </header>

  <div class="shared-disturbance">
    <div class="disturbance-now">
      <span>shared disturbance</span>
      <b class:active={currentDisturbance !== 0}>{currentDisturbance>0?'→':currentDisturbance<0?'←':'—'} {Math.abs(currentDisturbance).toFixed(0)} N</b>
    </div>
    <div class="timeline" aria-label="shared disturbance schedule">
      {#each COMPARISON_DISTURBANCE_PULSES as pulse}
        <i
          class:negative={pulse.force < 0}
          style={"left:"+(pulse.start/COMPARISON_HORIZON_STEPS*100)+"%;width:"+(pulse.duration/COMPARISON_HORIZON_STEPS*100)+"%"}
          title={(pulse.start*PHYSICS.tau).toFixed(1)+'s · '+pulse.force+'N'}
        ></i>
      {/each}
      <em style={"left:"+((snapshot?.tick || 0)/COMPARISON_HORIZON_STEPS*100)+"%"}></em>
    </div>
  </div>

  <div class="controller-grid">
    {#each ['state','vision','fusion'] as name}
      {@const c = snapshot?.controllers?.[name]}
      {@const p = c ? statePose(c.state) : statePose({x:0,theta:0})}
      <article class:failed={c?.failed} class="controller-card" data-controller={name} data-disturbance={c?.disturbance ?? 0}>
        <div class="controller-head">
          <div><b>{names[name]}</b><span>{descriptions[name]}</span></div>
          <strong>{c?.failed ? 'FELL' : finished ? 'DONE' : 'ACTIVE'}</strong>
        </div>

        <svg viewBox="0 0 320 185" class="mini-sim" aria-label={names[name]+' comparison simulation'}>
          <rect x="24" y="145" width="272" height="18" rx="6" fill="#f1f3f6"/>
          <line x1="31" y1="154" x2="289" y2="154" stroke="#bbc3ce" stroke-width="4" stroke-linecap="round"/>
          <line x1="45" y1="139" x2="45" y2="169" stroke="#d78585" stroke-dasharray="3 3"/>
          <line x1="275" y1="139" x2="275" y2="169" stroke="#d78585" stroke-dasharray="3 3"/>

          <rect x={p.cx-34} y="126" width="68" height="26" rx="7" fill={c?.failed ? '#7a808a' : '#35445c'}/>
          <circle cx={p.cx-19} cy="159" r="8" fill="#1d2939"/>
          <circle cx={p.cx+19} cy="159" r="8" fill="#1d2939"/>
          <line x1={p.cx} y1="127" x2={p.tipX} y2={p.tipY} stroke={c?.failed ? '#989fa9' : '#e56e6e'} stroke-width="8" stroke-linecap="round"/>
          <circle cx={p.cx} cy="127" r="6" fill="#1d2939"/>

          {#if c && Math.abs(c.force) > .05}
            <line x1={p.cx} y1="176" x2={p.cx + Math.sign(c.force)*Math.min(45,Math.abs(c.force)*4)} y2="176" stroke="#8b5cf6" stroke-width="3"/>
          {/if}
        </svg>

        <div class="live-values">
          <div><span>x</span><b>{c?.state.x.toFixed(2) ?? '0.00'} m</b></div>
          <div><span>θ</span><b>{c ? deg(c.state.theta).toFixed(1) : '0.0'}°</b></div>
          <div><span>force</span><b>{c?.force.toFixed(2) ?? '0.00'} N</b></div>
          <div><span>disturbance</span><b>{c?.disturbance.toFixed(0) ?? '0'} N</b></div>
        </div>

        <div class="metrics">
          <div><span>max |θ|</span><b>{c ? deg(c.metrics.maxAbsTheta).toFixed(1) : '0.0'}°</b></div>
          <div><span>mean |θ|</span><b>{c ? deg(c.metrics.meanAbsTheta).toFixed(1) : '0.0'}°</b></div>
          <div><span>control effort</span><b>{c?.metrics.controlEffort.toFixed(2) ?? '0.00'} N·s</b></div>
          <div><span>survival</span><b>{c?.failed ? ((c.failedAt||0)*PHYSICS.tau).toFixed(2)+' s' : (snapshot?.time || 0).toFixed(2)+'+ s'}</b></div>
        </div>
      </article>
    {/each}
  </div>

  <div class="replay-controls">
    <button class="primary" type="button" on:click={toggle}>{running ? 'Pause' : finished && currentIndex===0 ? 'Replay' : 'Run'}</button>
    <button type="button" on:click={init}>Reset deterministic run</button>
    <button type="button" on:click={finishRun} disabled={finished}>Fast-forward to end</button>
    <button type="button" on:click={latest} disabled={currentIndex===trace.length-1}>Latest</button>
    <input
      type="range"
      min="0"
      max={Math.max(0,trace.length-1)}
      value={currentIndex}
      on:input={scrub}
      aria-label="comparison replay tick"
    />
    <span>{currentIndex} / {Math.max(0,trace.length-1)} recorded</span>
  </div>

  {#if finished}
    <div class="final-summary">
      <b>Completed deterministic 10 s replay.</b>
      <span>No controller is ranked here; the raw survival, angle and effort metrics stay visible above.</span>
    </div>
  {/if}
</section>

<style>
.comparison-lab{background:#fff;border:1px solid #e1e5ea;border-radius:16px;padding:15px;overflow:hidden}
.compare-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;padding-bottom:11px;border-bottom:1px solid #edf0f3}.eyebrow{font-size:9px;letter-spacing:.08em;color:#7f70aa}.compare-head h2{font-size:18px;margin:2px 0 3px}.compare-head p{font-size:10px;color:#717a88;margin:0}.compare-clock{text-align:right}.compare-clock b{display:block;font:18px ui-monospace,SFMono-Regular,Menlo,monospace}.compare-clock span{font-size:9px;color:#8b93a1}
.shared-disturbance{display:grid;grid-template-columns:130px 1fr;gap:12px;align-items:center;margin:11px 0}.disturbance-now span{display:block;font-size:8px;color:#89919d}.disturbance-now b{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085}.disturbance-now b.active{color:#b86713}.timeline{height:12px;border-radius:6px;background:#edf0f3;position:relative;overflow:hidden}.timeline i{position:absolute;top:0;bottom:0;background:#d7a05b}.timeline i.negative{background:#7da2cc}.timeline em{position:absolute;top:-2px;bottom:-2px;width:2px;background:#5d4b92;z-index:2}
.controller-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.controller-card{border:1px solid #e3e7ed;border-radius:11px;padding:10px;background:#fbfcfd}.controller-card.failed{background:#f5f5f6}.controller-head{display:flex;justify-content:space-between;align-items:flex-start}.controller-head>div{display:flex;flex-direction:column}.controller-head b{font-size:13px}.controller-head span{font-size:8px;color:#8a93a2}.controller-head strong{font-size:8px;color:#3d8a64}.controller-card.failed .controller-head strong{color:#b65c5c}.mini-sim{width:100%;height:180px;display:block;margin-top:5px;background:linear-gradient(#fff,#f7f8fa);border-radius:8px;border:1px solid #edf0f3}
.live-values,.metrics{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:7px}.live-values>div,.metrics>div{padding:5px 6px;border:1px solid #e6e9ed;border-radius:6px;background:#fff}.live-values span,.metrics span{display:block;font-size:7px;color:#89919d}.live-values b,.metrics b{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}
.replay-controls{display:grid;grid-template-columns:auto auto auto auto minmax(160px,1fr) auto;gap:7px;align-items:center;margin-top:11px}.replay-controls button{border:1px solid #d9dde5;border-radius:7px;background:#fff;padding:6px 9px;font-size:9px;cursor:pointer}.replay-controls .primary{background:#293548;color:#fff;border-color:#293548}.replay-controls button:disabled{opacity:.35}.replay-controls span{font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#78818f}.replay-controls input{width:100%}
.final-summary{margin-top:9px;padding:8px 10px;border-radius:7px;background:#f3f0f9;display:flex;gap:10px;align-items:baseline}.final-summary b{font-size:10px;color:#5f4a93}.final-summary span{font-size:9px;color:#707887}
@media(max-width:980px){.controller-grid{grid-template-columns:1fr}.mini-sim{height:155px}.replay-controls{grid-template-columns:1fr 1fr}.replay-controls input{grid-column:1/3}.replay-controls span{grid-column:1/3}.shared-disturbance{grid-template-columns:1fr}.compare-head h2{font-size:16px}}
</style>
