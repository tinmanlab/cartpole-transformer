<!--
Opt-in "Follow one decision" walkthrough for State/learned mode only.
Reuses App's existing result/expandedStage/DecisionTrace plumbing — this
component owns no physics and no model computation of its own. `result` and
`controllerForce` are the frozen snapshot captured once at open time (App
stops passing the live values in while this guide is mounted), so Input/
Calculation/Action keep showing the captured event even after Apply advances
the live plant by one tick. See docs/learning-suite.md for the four-stage
contract this follows.
-->
<script>
  import { onMount, tick } from 'svelte';
  import DecisionTrace from './DecisionTrace.svelte';

  export let capturedTick = 0;
  export let currentTick = 0;
  export let eventId = 0;
  export let result;
  export let controllerForce = 0;
  export let lastDecisionTrace = null;
  export let status = 'balancing';
  export let onSelectToken = () => {};
  export let onExpandedStageChange = () => {};
  export let onApplyStep = () => {};
  export let onClose = () => {};
  export let onNewDecision = () => {};

  const FIELDS = [['x','m'],['ẋ','m/s'],['θ','rad'],['θ̇','rad/s']];
  const STAGES = [
    ['input','Input','입력'],
    ['calculation','Calculation','계산'],
    ['action','Action','행동'],
    ['result','Result','결과']
  ];

  let stage = 'input';
  let stepApplied = false;
  let applyFailed = false;
  let capturedTrace = null;

  $: lastIndex = Math.max(0, (result?.rawTokens?.length || 1) - 1);
  $: inputValues = result?.rawTokens?.[lastIndex] || [0,0,0,0];

  function goStage(next) {
    stage = next;
    onSelectToken(lastIndex);
    if (next === 'calculation') onExpandedStageChange('attention');
    else if (next === 'action') onExpandedStageChange('action');
    else onExpandedStageChange(null);
  }

  async function applyStep() {
    if (stepApplied || status === 'fell') return;
    onApplyStep();
    // onApplyStep mutates the App-level lastDecisionTrace; tick() flushes
    // that prop update down to us before we check it, so this never reports
    // success on a step that didn't actually happen.
    await tick();
    if (lastDecisionTrace && lastDecisionTrace.tickFrom === capturedTick) {
      stepApplied = true;
      applyFailed = false;
    } else {
      applyFailed = true;
    }
  }

  $: if (stepApplied && !capturedTrace && lastDecisionTrace && lastDecisionTrace.tickFrom === capturedTick) {
    capturedTrace = lastDecisionTrace;
  }

  onMount(() => goStage('input'));
</script>

<section class="follow-decision-guide">
  <header>
    <div>
      <span class="fd-badge">FROZEN · state · event #{eventId} · captured tick {capturedTick} · live plant tick {currentTick}</span>
      <h3>한 판단 따라가기 · Follow one decision</h3>
    </div>
    <button type="button" class="fd-close" on:click={onClose} aria-label="close follow-one-decision guide">×</button>
  </header>

  <div class="fd-stages" role="tablist" aria-label="decision stages">
    {#each STAGES as [key,en,ko]}
      <button type="button" role="tab" aria-selected={stage===key} class:active={stage===key} on:click={()=>goStage(key)}>{en} · {ko}</button>
    {/each}
  </div>

  <div class="fd-content">
    {#if stage === 'input'}
      <p>Input / 입력 — real state token captured at tick {capturedTick} (frozen; not the live plant above).</p>
      <div class="fd-values">
        {#each FIELDS as [name,unit],i}
          <span><b>{name} [{unit}]</b>{inputValues[i].toFixed(3)}</span>
        {/each}
      </div>
    {:else if stage === 'calculation'}
      <p>Calculation / 계산 — real Q·K softmax attention for the captured tick {capturedTick} token. See the Self Attention detail opened above for the exact query/key/value trace (also frozen to this event).</p>
    {:else if stage === 'action'}
      <p>Action / 행동 — real action score → force command from the captured tick {capturedTick}. See the Action head detail opened above.</p>
      <div class="fd-values">
        <span><b>captured tick</b>{capturedTick}</span>
        <span><b>force command [N]</b>{controllerForce.toFixed(2)}</span>
      </div>
    {:else}
      <p>Result / 결과 — apply exactly one real 20&nbsp;ms physics step from this frozen event to see what actually happens next.</p>
      {#if !capturedTrace}
        <button type="button" class="fd-apply" disabled={status==='fell'} on:click={applyStep}>Apply one 20ms step · 1 step 실행</button>
        {#if applyFailed}
          <p class="fd-error">The step did not apply (simulation state changed under the guide). Close and re-open Follow one decision to capture a fresh event.</p>
        {/if}
      {:else}
        <DecisionTrace trace={capturedTrace}/>
        <button type="button" class="fd-new" disabled={status==='fell'} on:click={onNewDecision}>새 판단 따라가기 · Follow next decision</button>
      {/if}
    {/if}
  </div>
</section>

<style>
.follow-decision-guide{margin-top:12px;background:#fff;border:1px solid #e2e5ea;border-radius:14px;padding:14px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;padding-bottom:10px;border-bottom:1px solid #edf0f3}
.fd-badge{display:block;font-size:14px;letter-spacing:.02em;color:#674e9f;font:14px ui-monospace,SFMono-Regular,Menlo,monospace}
header h3{font-size:15px;margin:5px 0 0}
.fd-close{width:36px;height:36px;min-width:44px;min-height:44px;border:1px solid #dfe3e8;border-radius:9px;background:#fff;color:#667085;font-size:18px;cursor:pointer}
.fd-stages{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:12px}
.fd-stages button{border:1px solid #d9dde5;border-radius:8px;background:#fff;padding:10px 8px;min-height:44px;font-size:14px;cursor:pointer;color:#4b5563}
.fd-stages button.active{background:#243047;color:#fff;border-color:#243047}
.fd-content{margin-top:12px}
.fd-content p{font-size:14px;line-height:1.5;color:#596273;margin:0 0 10px}
.fd-error{color:#a3401b}
.fd-values{display:flex;flex-wrap:wrap;gap:6px}
.fd-values span{display:flex;justify-content:space-between;gap:6px;flex:1;min-width:110px;border:1px solid #e4e7ec;background:#fafbfc;border-radius:6px;padding:6px 7px;font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#596273}
.fd-values b{font-family:Inter,ui-sans-serif,system-ui;font-size:14px;color:#8b93a1}
.fd-apply,.fd-new{margin-top:6px;border:1px solid #243047;border-radius:8px;background:#243047;color:#fff;padding:10px 14px;min-height:44px;font-size:14px;cursor:pointer}
.fd-apply:disabled,.fd-new:disabled{opacity:.4;cursor:not-allowed}
@media(max-width:560px){.fd-stages{grid-template-columns:repeat(2,1fr)}}
</style>
