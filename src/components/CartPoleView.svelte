<script>
  export let state;
  export let force = 0;
  export let running = true;
  export let elapsed = 0;
  export let status = 'balancing';
  export let onToggle = () => {};
  export let onReset = () => {};
  export let onPush = () => {};
  export let onPushEnd = () => {};

  $: cx = 360 + state.x * 115;
  $: poleLength = 145;
  $: tipX = cx + Math.sin(state.theta) * poleLength;
  $: tipY = 218 - Math.cos(state.theta) * poleLength;
  $: deg = state.theta * 180 / Math.PI;
  $: forceWidth = Math.min(90, Math.abs(force) * 7);
</script>

<div class="sim-card">
  <div class="sim-head">
    <div><span>SIMULATION · 50 Hz</span><strong>{status === 'fell' ? 'Pole fell' : 'Live Cart-Pole'}</strong></div>
    <div class="time">{elapsed.toFixed(1)} s</div>
  </div>

  <svg viewBox="0 0 720 330" class="sim" role="img" aria-label="live Cart-Pole simulation">
    <defs><linearGradient id="trackGlow" x1="0" x2="1"><stop offset="0" stop-color="#e7eaf0"/><stop offset=".5" stop-color="#cbd2df"/><stop offset="1" stop-color="#e7eaf0"/></linearGradient></defs>
    <line x1="35" y1="268" x2="685" y2="268" stroke="url(#trackGlow)" stroke-width="6" stroke-linecap="round"/>
    <line x1="360" y1="56" x2="360" y2="282" stroke="#eef0f4" stroke-width="1"/>
    <g>
      <rect x={cx-57} y="225" width="114" height="42" rx="10" fill="#39485f"/>
      <circle cx={cx-30} cy="278" r="13" fill="#192334"/><circle cx={cx+30} cy="278" r="13" fill="#192334"/>
      <line x1={cx} y1="226" x2={tipX} y2={tipY} stroke="#e66161" stroke-width="15" stroke-linecap="round"/>
      <circle cx={cx} cy="226" r="10" fill="#1d2939"/>
    </g>
    {#if Math.abs(force) > .1}
      <line x1={cx} y1="305" x2={cx + Math.sign(force)*forceWidth} y2="305" stroke="#8b66d8" stroke-width="7" stroke-linecap="round"/>
      <polygon points={force>0 ? (cx+forceWidth+8)+",305 "+(cx+forceWidth-5)+",298 "+(cx+forceWidth-5)+",312" : (cx-forceWidth-8)+",305 "+(cx-forceWidth+5)+",298 "+(cx-forceWidth+5)+",312"} fill="#8b66d8"/>
    {/if}
    <g class="state-readout">
      <text x="24" y="28">x {state.x.toFixed(2)} m</text>
      <text x="24" y="48">ẋ {state.xDot.toFixed(2)} m/s</text>
      <text x="24" y="68">θ {deg.toFixed(1)}°</text>
      <text x="24" y="88">θ̇ {(state.thetaDot*180/Math.PI).toFixed(1)}°/s</text>
      <text x="570" y="28">F {force.toFixed(1)} N</text>
    </g>
  </svg>

  <div class="sim-controls">
    <button class="primary" on:click={onToggle}>{running ? 'Pause' : 'Run'}</button>
    <button on:click={onReset}>Reset</button>
    <span class="spacer"></span>
    <button on:pointerdown={() => onPush(-6)} on:pointerup={onPushEnd} on:pointerleave={onPushEnd}>← Push</button>
    <button on:pointerdown={() => onPush(6)} on:pointerup={onPushEnd} on:pointerleave={onPushEnd}>Push →</button>
  </div>
</div>

<style>
.sim-card{height:100%;min-height:430px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:13px;display:flex;flex-direction:column}.sim-head{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:0 2px 10px;border-bottom:1px solid #eef0f3}.sim-head span{display:block;font-size:9px;letter-spacing:.09em;color:#8a93a2}.sim-head strong{display:block;font-size:14px;margin-top:1px}.time{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085}.sim{width:100%;height:auto;display:block;margin-top:10px;background:linear-gradient(180deg,#fcfdff,#f5f7fa);border-radius:12px;border:1px solid #edf0f4;flex:1;min-height:0}.sim text{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#677184}.sim-controls{display:flex;gap:7px;align-items:center;margin-top:9px}.sim-controls button{border:1px solid #d9dde5;border-radius:8px;background:#fff;padding:7px 10px;min-height:36px;cursor:pointer;font-size:12px}.sim-controls .primary{background:#243047;color:#fff;border-color:#243047}.spacer{flex:1}
</style>
