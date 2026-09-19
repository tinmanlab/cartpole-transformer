<script>
  export let state;
  export let history = [];
  export let selectedToken = 7;
  export let controllerForce = 0;
  export let disturbance = 0;
  export let running = true;
  export let elapsed = 0;
  export let status = 'balancing';
  export let showStateOverlay = true;
  export let onToggle = () => {};
  export let onReset = () => {};
  export let onPush = () => {};
  export let onPushEnd = () => {};

  const centerX = 360;
  const pivotY = 224;
  const poleLength = 150;
  const worldScale = 122;

  function pose(values) {
    const [x, xDot, theta, thetaDot] = values;
    const cx = centerX + x * worldScale;
    return {
      x, xDot, theta, thetaDot, cx,
      tipX: cx + Math.sin(theta) * poleLength,
      tipY: pivotY - Math.cos(theta) * poleLength
    };
  }

  function arcPath(theta, radius=38) {
    const sx = centerX;
    const sy = pivotY - radius;
    const ex = centerX + Math.sin(theta) * radius;
    const ey = pivotY - Math.cos(theta) * radius;
    const sweep = theta >= 0 ? 1 : 0;
    return 'M '+sx+' '+sy+' A '+radius+' '+radius+' 0 0 '+sweep+' '+ex+' '+ey;
  }

  $: current = pose([state.x,state.xDot,state.theta,state.thetaDot]);
  $: ghosts = history.map(pose);
  $: selectedToken = Math.min(Math.max(0, selectedToken), Math.max(0,history.length-1));
  $: selectedPose = ghosts[selectedToken] || current;
  $: deg = state.theta * 180 / Math.PI;
  $: selectedDeg = selectedPose.theta * 180 / Math.PI;
  $: velocityLen = Math.min(80, Math.abs(state.xDot) * 42);
  $: controlLen = Math.min(88, Math.abs(controllerForce) * 7.5);
  $: disturbanceLen = Math.min(65, Math.abs(disturbance) * 8);
  $: last = Math.max(0,history.length-1);
</script>

<div class="sim-card resize-watch">
  <div class="sim-head">
    <div><span>PHYSICS · 50 Hz</span><strong>{status === 'fell' ? 'Pole fell' : 'Cart-Pole + 8-frame history'}</strong></div>
    <div class="time">{elapsed.toFixed(1)} s</div>
  </div>

  <svg viewBox="0 0 720 350" class="sim" role="img" aria-label="live Cart-Pole simulation with recent pose history">
    <defs>
      <linearGradient id="trackGlow" x1="0" x2="1"><stop offset="0" stop-color="#e5e7eb"/><stop offset=".5" stop-color="#bfc7d4"/><stop offset="1" stop-color="#e5e7eb"/></linearGradient>
      <marker id="simArrowPurple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#8b5cf6"/></marker>
      <marker id="simArrowGray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#667085"/></marker>
      <marker id="simArrowOrange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#d97706"/></marker>
    </defs>

    <rect x="52" y="265" width="616" height="38" rx="9" fill="#f7f8fa"/>
    <line x1="67" y1="282" x2="653" y2="282" stroke="url(#trackGlow)" stroke-width="6" stroke-linecap="round"/>
    <line x1="67" y1="257" x2="67" y2="309" stroke="#d36a6a" stroke-width="2" stroke-dasharray="4 4"/>
    <line x1="653" y1="257" x2="653" y2="309" stroke="#d36a6a" stroke-width="2" stroke-dasharray="4 4"/>
    <text x="67" y="326" text-anchor="middle">−2.4 m</text>
    <text x="360" y="326" text-anchor="middle">0</text>
    <text x="653" y="326" text-anchor="middle">+2.4 m</text>
    <line x1="360" y1="58" x2="360" y2="304" stroke="#e7eaf0" stroke-width="1" stroke-dasharray="3 5"/>

    {#if showStateOverlay}
      <g class="history-poses">
        {#each ghosts as g,i}
          {#if i < last}
            <g class:selected-ghost={i===selectedToken} opacity={i===selectedToken ? .82 : .05 + .035*i}>
              <rect x={g.cx-42} y="240" width="84" height="24" rx="5" fill="none" stroke={i===selectedToken ? '#6574c9' : '#7d8795'} stroke-width={i===selectedToken ? 2.4 : 1.1}/>
              <line x1={g.cx} y1="240" x2={g.tipX} y2={g.tipY+16} stroke={i===selectedToken ? '#6574c9' : '#7d8795'} stroke-width={i===selectedToken ? 5 : 3} stroke-linecap="round"/>
            </g>
          {/if}
        {/each}
      </g>

      {#if selectedToken < last}
        <g class="selected-history-label">
          <circle cx={selectedPose.tipX} cy={selectedPose.tipY+16} r="5" fill="#6574c9"/>
          <text x={selectedPose.tipX+9} y={selectedPose.tipY+12}>t−{last-selectedToken} · θ {selectedDeg.toFixed(1)}°</text>
        </g>
      {/if}
    {/if}

    <g class:current-selected={selectedToken===last} class="current-pose">
      <rect x={current.cx-56} y="226" width="112" height="42" rx="9" fill="#35445c"/>
      <rect x={current.cx-46} y="233" width="92" height="11" rx="4" fill="#53637a"/>
      <circle cx={current.cx-31} cy="281" r="13" fill="#182333"/>
      <circle cx={current.cx+31} cy="281" r="13" fill="#182333"/>
      <circle cx={current.cx-31} cy="281" r="5" fill="#7f8996"/>
      <circle cx={current.cx+31} cy="281" r="5" fill="#7f8996"/>
      <line x1={current.cx} y1="227" x2={current.tipX} y2={current.tipY} stroke="#db5b5b" stroke-width="13" stroke-linecap="round"/>
      <line x1={current.cx} y1="227" x2={current.tipX} y2={current.tipY} stroke="#f18a8a" stroke-width="4" stroke-linecap="round"/>
      <circle cx={current.cx} cy="227" r="10" fill="#182333"/>
      <circle cx={current.cx} cy="227" r="4" fill="#d7dce3"/>
    </g>

    {#if showStateOverlay}
      <path d={arcPath(state.theta)} fill="none" stroke="#9aa3b1" stroke-width="1.5" stroke-dasharray="3 3"/>
      <text x={current.cx + (state.theta>=0?46:-64)} y="180">θ {deg.toFixed(1)}°</text>
    {/if}

    {#if showStateOverlay && Math.abs(state.xDot) > .02}
      <line
        x1={current.cx}
        y1="211"
        x2={current.cx + Math.sign(state.xDot)*velocityLen}
        y2="211"
        stroke="#667085"
        stroke-width="2.5"
        marker-end="url(#simArrowGray)"
      />
      <text x={current.cx + Math.sign(state.xDot)*(velocityLen+12)} y="202" text-anchor={state.xDot>=0?'start':'end'}>ẋ</text>
    {/if}

    {#if Math.abs(controllerForce) > .05}
      <line
        x1={current.cx}
        y1="306"
        x2={current.cx + Math.sign(controllerForce)*controlLen}
        y2="306"
        stroke="#8b5cf6"
        stroke-width="4"
        marker-end="url(#simArrowPurple)"
      />
      <text x={current.cx + Math.sign(controllerForce)*(controlLen+13)} y="298" text-anchor={controllerForce>=0?'start':'end'} class="force-label">policy {controllerForce.toFixed(1)} N</text>
    {/if}

    {#if Math.abs(disturbance) > .05}
      <line
        x1={current.cx}
        y1="336"
        x2={current.cx + Math.sign(disturbance)*disturbanceLen}
        y2="336"
        stroke="#d97706"
        stroke-width="4"
        marker-end="url(#simArrowOrange)"
      />
      <text x={current.cx + Math.sign(disturbance)*(disturbanceLen+13)} y="345" text-anchor={disturbance>=0?'start':'end'} class="disturbance-label">push {disturbance.toFixed(0)} N</text>
    {/if}
  </svg>

  {#if showStateOverlay}
    <div class="state-readout">
      <span><b>x</b>{state.x.toFixed(2)} m</span>
      <span><b>ẋ</b>{state.xDot.toFixed(2)} m/s</span>
      <span><b>θ</b>{deg.toFixed(1)}°</span>
      <span><b>θ̇</b>{(state.thetaDot*180/Math.PI).toFixed(1)}°/s</span>
      <span class="selected-time"><b>selected</b>{selectedToken===last?'t':'t−'+(last-selectedToken)}</span>
    </div>
  {:else}
    <div class="vision-hidden-state">VISION-ONLY · state numbers hidden · controller sees rendered frames only</div>
  {/if}

  <div class="sim-controls">
    <button class="primary" on:click={onToggle}>{running ? 'Pause' : 'Run'}</button>
    <button on:click={onReset}>Reset</button>
    <span class="spacer"></span>
    <button class="push" on:pointerdown={() => onPush(-6)} on:pointerup={onPushEnd} on:pointerleave={onPushEnd}>← Push</button>
    <button class="push" on:pointerdown={() => onPush(6)} on:pointerup={onPushEnd} on:pointerleave={onPushEnd}>Push →</button>
  </div>
</div>

<style>
.sim-card{height:auto;min-height:500px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:11px;display:flex;flex-direction:column}
.sim-head{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:0 2px 8px;border-bottom:1px solid #eef0f3}.sim-head span{display:block;font-size:8px;letter-spacing:.09em;color:#8a93a2}.sim-head strong{display:block;font-size:12px;margin-top:1px}.time{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085}
.sim{width:100%;height:auto;display:block;margin-top:8px;background:linear-gradient(180deg,#fdfefe,#f7f8fa);border-radius:11px;border:1px solid #edf0f4;flex:1;min-height:0}.sim text{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#697386}.sim .force-label{fill:#7458b7}.sim .disturbance-label{fill:#b45f05}.current-pose.current-selected{filter:drop-shadow(0 0 3px rgba(101,116,201,.45))}.selected-history-label text{fill:#5968b7;font-weight:700}
.state-readout{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin-top:7px}.state-readout span{display:flex;justify-content:space-between;gap:4px;border:1px solid #e4e7ec;background:#fafbfc;border-radius:6px;padding:5px 6px;font:7px ui-monospace,SFMono-Regular,Menlo,monospace;color:#596273}.state-readout b{font-family:Inter,ui-sans-serif,system-ui;font-size:7px;color:#8b93a1}.state-readout .selected-time{background:#f2f0f9;color:#6855a1}.vision-hidden-state{margin-top:7px;padding:6px 8px;border:1px solid #e1e5ea;border-radius:6px;background:#f7f8fa;text-align:center;font-size:8px;letter-spacing:.04em;color:#7b8492}
.sim-controls{display:flex;gap:6px;align-items:center;margin-top:7px}.sim-controls button{border:1px solid #d9dde5;border-radius:7px;background:#fff;padding:6px 9px;min-height:33px;cursor:pointer;font-size:10px}.sim-controls .primary{background:#243047;color:#fff;border-color:#243047}.sim-controls .push{border-color:#e2d6c4;background:#fffaf2}.spacer{flex:1}
@media(max-width:560px){.state-readout{grid-template-columns:repeat(2,1fr)}.selected-time{grid-column:1/3}}
</style>
