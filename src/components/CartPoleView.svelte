<script>
  import { onMount } from 'svelte';

  export let state;
  export let history = [];
  export let selectedToken = 7;
  export let controllerForce = 0;
  export let disturbance = 0;
  export let running = true;
  export let elapsed = 0;
  export let status = 'balancing';
  export let showStateOverlay = true;
  export let guideActive = false;
  export let onToggle = () => {};
  export let onStep = () => {};
  export let onReset = () => {};
  export let onPush = () => {};
  export let onPushEnd = () => {};

  // Shared cartpole-v1 scene contract (docs/learning-suite.md): 640x320
  // schematic, worldScale110, track x56..584 <-> physical x -2.4..+2.4.
  const centerX = 320;
  const pivotY = 202;
  const poleLength = 132;
  const worldScale = 110;
  const viewBoxWidth = 640;
  const viewBoxHeight = 320;
  const railY = 251;
  const wheelY = 242;
  const wheelR = 9;
  const cartW = 78;
  const cartH = 28;

  let svgEl;
  let labelScale = 1;
  let showGhosts = false;

  onMount(() => {
    // viewBox scaling shrinks SVG <text> below readable size on narrow
    // layouts (getScreenCTM effective size != declared font-size);
    // counter-scale via CSS var so on-screen label size stays constant.
    const ro = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      if (width > 0) labelScale = viewBoxWidth / width;
    });
    ro.observe(svgEl);

    // A held push must not survive the pointer leaving the app entirely
    // (OS focus switch, tab hidden) or releasing outside the button --
    // window/document catch what the button's own pointer events miss.
    // Window pointerup fires for every pointer release on the page (e.g.
    // clicking Step), so it's gated on the pointerId that actually started
    // the held push, not treated as a blanket "clear on any release".
    const onWindowPointerUp = (e) => endPush(e.pointerId);
    const onBlur = () => forceEndPush();
    const onVisibility = () => { if (document.hidden) forceEndPush(); };
    window.addEventListener('blur', onBlur);
    window.addEventListener('pointerup', onWindowPointerUp);
    document.addEventListener('visibilitychange', onVisibility);

    return () => {
      ro.disconnect();
      window.removeEventListener('blur', onBlur);
      window.removeEventListener('pointerup', onWindowPointerUp);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  });

  let heldPointerId = null;

  function primaryPush(e, v) {
    if (e.button !== 0) return;
    heldPointerId = e.pointerId;
    onPush(v);
  }
  function endPush(pointerId) {
    if (pointerId !== undefined && pointerId !== heldPointerId) return;
    heldPointerId = null;
    onPushEnd();
  }
  function forceEndPush() {
    heldPointerId = null;
    onPushEnd();
  }

  function pose(values) {
    const [x, xDot, theta, thetaDot] = values;
    const cx = centerX + x * worldScale;
    return {
      x, xDot, theta, thetaDot, cx,
      tipX: cx + Math.sin(theta) * poleLength,
      tipY: pivotY - Math.cos(theta) * poleLength
    };
  }

  $: current = pose([state.x,state.xDot,state.theta,state.thetaDot]);
  $: ghosts = history.map(pose);
  $: selectedToken = Math.min(Math.max(0, selectedToken), Math.max(0,history.length-1));
  $: selectedPose = ghosts[selectedToken] || current;
  $: deg = state.theta * 180 / Math.PI;
  $: selectedDeg = selectedPose.theta * 180 / Math.PI;
  $: last = Math.max(0,history.length-1);
  $: tick = Math.round(elapsed / 0.02);
</script>

<div class="sim-card resize-watch" data-scene-contract="cartpole-v1">
  <div class="sim-head">
    <div>
      <span>{status === 'fell' ? 'FROZEN · fell' : running ? 'LIVE' : 'PAUSED'} · plant tick {tick} · x unit m · θ unit deg · terminal |x|&gt;2.4m or |θ|&gt;21.8°</span>
      <strong>{status === 'fell' ? 'Pole fell' : 'Cart-Pole'}</strong>
    </div>
    <div class="time">{elapsed.toFixed(1)} s</div>
  </div>

  <svg bind:this={svgEl} viewBox="0 0 {viewBoxWidth} {viewBoxHeight}" class="sim" role="img" aria-label="live Cart-Pole simulation" style="--label-scale: {labelScale}">
    <rect x="0" y="0" width={viewBoxWidth} height={viewBoxHeight} fill="#ffffff"/>
    <line x1="56" y1={railY} x2="584" y2={railY} stroke="#cbd5e1" stroke-width="6" stroke-linecap="round"/>
    <text x="56" y={railY+20} text-anchor="middle">−2.4 m</text>
    <text x={centerX} y={railY+20} text-anchor="middle">0</text>
    <text x="584" y={railY+20} text-anchor="middle">+2.4 m</text>

    {#if showStateOverlay && showGhosts}
      <g class="history-poses">
        {#each ghosts as g,i}
          {#if i < last}
            <g class:selected-ghost={i===selectedToken} opacity={i===selectedToken ? .82 : .08 + .04*i}>
              <rect x={g.cx-cartW/2} y={pivotY} width={cartW} height={cartH} rx="4" fill="none" stroke={i===selectedToken ? '#6574c9' : '#7d8795'} stroke-width={i===selectedToken ? 2.4 : 1.1}/>
              <line x1={g.cx} y1={pivotY} x2={g.tipX} y2={g.tipY} stroke={i===selectedToken ? '#6574c9' : '#7d8795'} stroke-width={i===selectedToken ? 4 : 2.4} stroke-linecap="round"/>
            </g>
          {/if}
        {/each}
      </g>

      {#if selectedToken < last}
        <g class="selected-history-label">
          <circle cx={selectedPose.tipX} cy={selectedPose.tipY} r="4" fill="#6574c9"/>
          <text x={selectedPose.tipX+8} y={selectedPose.tipY-4}>t−{last-selectedToken} · θ {selectedDeg.toFixed(1)}°</text>
        </g>
      {/if}
    {/if}

    <g class="current-pose">
      <line x1={current.cx} y1={pivotY} x2={current.tipX} y2={current.tipY} stroke="#dc5b60" stroke-width="7" stroke-linecap="round"/>
      <rect x={current.cx-cartW/2} y={pivotY} width={cartW} height={cartH} rx="4" fill="#334155"/>
      <circle cx={current.cx-24} cy={wheelY} r={wheelR} fill="#1e293b"/>
      <circle cx={current.cx+24} cy={wheelY} r={wheelR} fill="#1e293b"/>
    </g>
  </svg>

  {#if showStateOverlay}
    <label class="ghost-toggle">
      <input type="checkbox" bind:checked={showGhosts}/>
      History poses (t−1…t−7) · explicit inspection
    </label>
  {/if}

  <!-- Signed physical-force readout lane (cartpole-v1 contract): the pending
       controller command is green (it is the NEXT force to be applied on
       the next Step/tick, not yet measured or delivered -- there is no
       separate actuator model in this repo, so command == delivered once
       applied), external disturbance is amber. Zero stays a plain signed
       number, never an arrow, so the lane never shifts layout. -->
  <div class="force-lane">
    <span class="force-tile action-force"><b>next command</b>{controllerForce >= 0 ? '+' : ''}{controllerForce.toFixed(2)} N</span>
    <span class="force-tile disturbance-force"><b>external disturbance</b>{disturbance >= 0 ? '+' : ''}{disturbance.toFixed(2)} N</span>
  </div>

  {#if showStateOverlay}
    <div class="state-readout">
      <span><b>x [m]</b>{state.x.toFixed(2)}</span>
      <span><b>ẋ [m/s]</b>{state.xDot.toFixed(2)}</span>
      <span><b>θ [deg]</b>{deg.toFixed(1)}</span>
      <span><b>θ̇ [deg/s]</b>{(state.thetaDot*180/Math.PI).toFixed(1)}</span>
      <span class="selected-time"><b>selected</b>{selectedToken===last?'t':'t−'+(last-selectedToken)}</span>
    </div>
  {:else}
    <div class="vision-hidden-state">VISION-ONLY · state numbers hidden · controller sees rendered frames only</div>
  {/if}

  <div class="sim-controls">
    <button class="primary" disabled={status==='fell' || guideActive} on:click={onToggle}>{running ? 'Pause' : 'Run'}</button>
    <button disabled={running || status==='fell' || guideActive} on:click={onStep}>Step</button>
    <button on:click={onReset}>Reset</button>
    <span class="spacer"></span>
    <button class="push" disabled={guideActive} on:pointerdown={(e) => primaryPush(e,-6)} on:pointerup={(e) => endPush(e.pointerId)} on:pointerleave={(e) => endPush(e.pointerId)} on:pointercancel={(e) => endPush(e.pointerId)}>← Push</button>
    <button class="push" disabled={guideActive} on:pointerdown={(e) => primaryPush(e,6)} on:pointerup={(e) => endPush(e.pointerId)} on:pointerleave={(e) => endPush(e.pointerId)} on:pointercancel={(e) => endPush(e.pointerId)}>Push →</button>
  </div>
</div>

<style>
.sim-card{height:auto;min-height:500px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:11px;display:flex;flex-direction:column;min-width:0}
.sim-head{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:0 2px 8px;border-bottom:1px solid #eef0f3}
.sim-head span{display:block;font-size:14px;letter-spacing:.02em;color:#596273}
.sim-head strong{display:block;font-size:16px;margin-top:1px}
.time{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085}
.sim{width:100%;height:auto;display:block;margin-top:8px;background:#ffffff;border-radius:11px;border:1px solid #edf0f4;flex:1;min-height:0}
/* In-diagram SVG labels stay compact (scale-corrected to ~10px, never smaller); every value they annotate is duplicated in the >=14px HTML readout below, so this is a documented, always-readable exception rather than hidden content */
.sim text{font:calc(10px * var(--label-scale, 1)) ui-monospace,SFMono-Regular,Menlo,monospace;fill:#697386}
.selected-history-label text{fill:#5968b7;font-weight:700}
.ghost-toggle{display:flex;align-items:center;gap:6px;margin-top:7px;font-size:14px;color:#596273;min-height:44px}
.force-lane{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}.force-lane .force-tile{display:flex;justify-content:space-between;gap:5px;flex:1;min-width:150px;border:1px solid #e4e7ec;border-radius:6px;padding:6px 7px;font:14px ui-monospace,SFMono-Regular,Menlo,monospace}.force-lane b{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px}
.action-force{background:#e9f7f0;color:#16805d}
.disturbance-force{background:#fdf3e7;color:#b86b16}
.state-readout{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin-top:7px}.state-readout span{display:flex;justify-content:space-between;gap:5px;border:1px solid #e4e7ec;background:#fafbfc;border-radius:6px;padding:6px 7px;font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#596273}.state-readout b{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;color:#8b93a1}.state-readout .selected-time{background:#f2f0f9;color:#6855a1}.vision-hidden-state{margin-top:7px;padding:7px 9px;border:1px solid #e1e5ea;border-radius:6px;background:#f7f8fa;text-align:center;font-size:14px;line-height:1.4;letter-spacing:.03em;color:#7b8492}
.sim-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:7px}.sim-controls button{border:1px solid #d9dde5;border-radius:7px;background:#fff;padding:10px 14px;min-height:44px;min-width:44px;cursor:pointer;font-size:14px}.sim-controls .primary{background:#243047;color:#fff;border-color:#243047}.sim-controls button:disabled{opacity:.38;cursor:not-allowed}.sim-controls .push{border-color:#e2d6c4;background:#fffaf2}.spacer{flex:1 1 0;min-width:0}
@media(max-width:560px){.state-readout{grid-template-columns:repeat(2,1fr)}.selected-time{grid-column:1/3}}
</style>
