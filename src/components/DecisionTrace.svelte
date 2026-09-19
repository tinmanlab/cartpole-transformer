<script>
  export let trace = null;

  const deg = v => v * 180 / Math.PI;
  const f = (v,n=3) => Number(v).toFixed(n);

  function observationText(obs) {
    if (!obs) return '';
    if (obs.kind === 'state') return 'explicit state token · 4D';
    if (obs.kind === 'vision') return obs.frameCount+' frames · '+obs.inputDim+'D visual input';
    if (obs.kind === 'fusion') return obs.tokenCount+' typed tokens · State + Vision';
    return obs.kind || '';
  }
</script>

<section
  class="decision-trace"
  data-has-trace={trace ? 'true' : 'false'}
  data-mode={trace?.mode ?? ''}
  data-tick-from={trace?.tickFrom ?? ''}
  data-tick-to={trace?.tickTo ?? ''}
  data-before-state={trace?.beforeState?.join(',') ?? ''}
  data-applied-action-score={trace?.appliedActionScore ?? ''}
  data-applied-policy-force={trace?.appliedPolicyForce ?? ''}
  data-applied-control={trace?.appliedControl ?? ''}
  data-disturbance={trace?.disturbance ?? ''}
  data-total-force={trace?.totalForce ?? ''}
  data-x-acc={trace?.xAcc ?? ''}
  data-theta-acc={trace?.thetaAcc ?? ''}
  data-dt={trace?.dt ?? ''}
  data-next-state={trace?.nextState?.join(',') ?? ''}
  data-next-action-score={trace?.nextActionScore ?? ''}
  data-next-policy-force={trace?.nextPolicyForce ?? ''}
>
  <header>
    <div>
      <span class="eyebrow">CLOSED LOOP · ONE REAL 20 ms TRANSITION</span>
      <h2>Controller → Plant → Next state</h2>
    </div>
    {#if trace}
      <div class="tick">tick {trace.tickFrom} → {trace.tickTo}</div>
    {/if}
  </header>

  {#if !trace}
    <div class="empty">
      Pause the simulation and press <b>Step</b>. The exact controller action and plant dynamics used by that transition will appear here.
    </div>
  {:else}
    <div class="flow">
      <article class="stage observation">
        <div class="step">1</div>
        <h3>Observation at t</h3>
        <strong>{observationText(trace.observation)}</strong>
        {#if trace.observation?.kind === 'state'}
          <div class="state-row">
            {#each [['x',0],['ẋ',1],['θ',2],['θ̇',3]] as item}
              <span><small>{item[0]}</small>{f(trace.observation.values[item[1]])}</span>
            {/each}
          </div>
        {:else if trace.observation?.kind === 'vision'}
          <p>{trace.observation.frameSpanMs.toFixed(0)} ms history · latest pixels are the controller input.</p>
        {:else if trace.observation?.kind === 'fusion'}
          <p>{trace.observation.stateAvailable?'state on':'state off'} · {trace.observation.visionAvailable?'vision on':'vision off'}</p>
        {/if}
      </article>

      <div class="arrow">→</div>

      <article class="stage controller">
        <div class="step">2</div>
        <h3>Transformer / controller</h3>
        <div class="kv"><span>action score</span><b>{f(trace.appliedActionScore,5)}</b></div>
        <div class="force big"><span>policy force uₜ</span><b>{trace.appliedPolicyForce>=0?'→':'←'} {f(Math.abs(trace.appliedPolicyForce),2)} N</b></div>
        <p>This is the force that was visible before Step.</p>
      </article>

      <div class="arrow">→</div>

      <article class="stage plant-input">
        <div class="step">3</div>
        <h3>Plant input</h3>
        <div class="sum">
          <div><span>policy</span><b>{f(trace.appliedControl,2)} N</b></div>
          <i>+</i>
          <div><span>disturbance</span><b>{f(trace.disturbance,2)} N</b></div>
          <i>=</i>
          <div class="total"><span>total F</span><b>{f(trace.totalForce,2)} N</b></div>
        </div>
        <p>Policy and external push stay separate until the plant equation.</p>
      </article>

      <div class="arrow">→</div>

      <article class="stage dynamics">
        <div class="step">4</div>
        <h3>Cart-Pole dynamics</h3>
        <div class="accel">
          <div><span>ẍ</span><b>{f(trace.xAcc,3)} m/s²</b></div>
          <div><span>θ̈</span><b>{f(trace.thetaAcc,3)} rad/s²</b></div>
        </div>
        <code>temp = {f(trace.temp,5)}</code>
        <p>These are the actual intermediates returned by the environment.</p>
      </article>

      <div class="arrow">→</div>

      <article class="stage next-state">
        <div class="step">5</div>
        <h3>Euler integration → t+1</h3>
        <div class="state-row">
          {#each [['x',0],['ẋ',1],['θ',2],['θ̇',3]] as item}
            <span><small>{item[0]}</small>{f(trace.nextState[item[1]])}</span>
          {/each}
        </div>
        <div class="next-action">
          <span>new policy force for the next transition</span>
          <b>{trace.nextPolicyForce>=0?'→':'←'} {f(Math.abs(trace.nextPolicyForce),2)} N</b>
        </div>
      </article>
    </div>

    <div class="equations">
      <span>xₜ₊₁ = xₜ + Δt·ẋₜ</span>
      <span>ẋₜ₊₁ = ẋₜ + Δt·ẍₜ</span>
      <span>θₜ₊₁ = θₜ + Δt·θ̇ₜ</span>
      <span>θ̇ₜ₊₁ = θ̇ₜ + Δt·θ̈ₜ</span>
      <b>Δt = {(trace.dt*1000).toFixed(0)} ms</b>
    </div>
  {/if}
</section>

<style>
.decision-trace{margin-top:10px;background:#fff;border:1px solid #e1e5ea;border-radius:16px;padding:15px 18px 17px;overflow:hidden}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;padding-bottom:10px;border-bottom:1px solid #edf0f3}.eyebrow{display:block;font-size:9px;letter-spacing:.08em;color:#7f70aa}h2{font-size:17px;margin:2px 0 0}.tick{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6f7886;white-space:nowrap}.empty{padding:20px 8px 6px;text-align:center;color:#737c89;font-size:11px}
.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1.05fr auto 1fr auto 1.1fr;gap:8px;align-items:stretch;padding-top:13px}.stage{position:relative;min-width:0;border:1px solid #e4e7ec;border-radius:10px;background:#fbfcfd;padding:11px}.stage h3{font-size:10px;margin:0 0 9px;color:#4f5968}.stage>strong{font-size:10px;color:#445064}.stage p{font-size:8px;line-height:1.4;color:#7d8693;margin:8px 0 0}.step{position:absolute;right:7px;top:7px;width:18px;height:18px;line-height:18px;text-align:center;border-radius:50%;background:#293548;color:#fff;font-size:8px}.arrow{display:flex;align-items:center;color:#816db5;font-size:17px}.state-row{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:8px}.state-row span,.accel>div,.sum>div{padding:5px;border:1px solid #e7e9ed;border-radius:6px;background:#fff;font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}.state-row small,.accel span,.sum span,.kv span,.force span,.next-action span{display:block;font:7px Inter,ui-sans-serif,system-ui;color:#89919d;margin-bottom:2px}.kv,.force,.next-action{padding:6px;border:1px solid #e6e9ed;border-radius:6px;background:#fff}.kv b,.force b,.next-action b{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4f5968}.force.big{margin-top:5px;background:#f2eef9}.force.big b,.next-action b{color:#674e9f}.sum{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:4px;align-items:center}.sum i{font-style:normal;color:#7c6aab}.sum .total{background:#fff4e7}.accel{display:grid;grid-template-columns:1fr 1fr;gap:4px}.dynamics code{display:block;margin-top:6px;font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#687386}.next-action{margin-top:7px}.equations{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.equations span,.equations b{padding:5px 7px;border-radius:6px;background:#f5f6f8;font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085}.equations b{background:#ede9f7;color:#654e9a}
@media(max-width:980px){.flow{display:flex;flex-direction:column}.arrow{justify-content:center;transform:rotate(90deg);height:18px}.stage{width:100%}.equations{display:grid;grid-template-columns:1fr 1fr}}
</style>
