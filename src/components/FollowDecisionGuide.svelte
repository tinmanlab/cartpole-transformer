<!--
Opt-in "Follow one decision" walkthrough for State/learned mode only.
Reuses App's existing result/expandedStage/DecisionTrace plumbing — this
component owns no physics and no model computation of its own. `result` and
`controllerForce` are the frozen snapshot captured once at open time (App
stops passing the live values in while this guide is mounted), so Input/
Calculation/Action keep showing the captured event even after Apply advances
the live plant by one tick. The Calculation/Action numeric traces below read
straight out of that frozen `result` object (qkProducts/scores/raw/softmax*/
weights/weightedValueContributions/perTokenContext/actionScore) — no model or
softmax math is reimplemented here. Query is fixed to the latest token; Key
and value-dimension are selectable, each showing one real number in place
(score/mask/softmax stay visible; full per-dimension product/contribution/
context vectors sit behind native <details>, collapsed by default). The full
Self Attention / Action head drawer stays a single advanced instance opened
above (not copied into this guide).
See docs/learning-suite.md for the four-stage contract this follows.
-->
<script>
  import { onMount, tick } from 'svelte';
  import DecisionTrace from './DecisionTrace.svelte';
  import { forceFromScore } from '../lib/attention.js';

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
  let applying = false;
  let stepApplied = false;
  let applyFailed = false;
  let capturedTrace = null;
  let tabRefs = [];

  $: lastIndex = Math.max(0, (result?.rawTokens?.length || 1) - 1);
  $: inputValues = result?.rawTokens?.[lastIndex] || [0,0,0,0];

  // Selectable history/key token indices, each initialized once (to the
  // latest/query token) as soon as the frozen result is available, then left
  // alone so later user selections persist across stage switches.
  let selectedInputToken = 0;
  let selectedInputTokenInit = false;
  $: if (!selectedInputTokenInit && result) { selectedInputToken = lastIndex; selectedInputTokenInit = true; }

  let selectedKey = 0;
  let selectedKeyInit = false;
  $: if (!selectedKeyInit && result) { selectedKey = lastIndex; selectedKeyInit = true; }

  let selectedDim = 0;
  let selectedDimInit = false;
  $: if (!selectedDimInit && result) { selectedDim = 0; selectedDimInit = true; }

  function label(i) {
    return i === lastIndex ? 't' : 't−' + (lastIndex - i);
  }

  // Calculation-stage numeric trace: query is fixed to the actual controller
  // query (the latest/current token); only the key token is selectable.
  $: qkProductsSel = result?.qkProducts?.[lastIndex]?.[selectedKey] || [];
  $: dotSum = qkProductsSel.reduce((sum,x) => sum + x, 0);
  $: scaleDim = Math.sqrt(qkProductsSel.length || 1);
  $: scaledScore = result?.scores?.[lastIndex]?.[selectedKey] ?? 0;
  $: rawScoreSel = result?.raw?.[lastIndex]?.[selectedKey] ?? 0;
  $: maskedSel = !Number.isFinite(rawScoreSel);
  $: expValSel = result?.softmaxExp?.[lastIndex]?.[selectedKey] ?? 0;
  $: denomSel = result?.softmaxDenominators?.[lastIndex] ?? 0;
  $: weightSel = result?.weights?.[lastIndex]?.[selectedKey] ?? 0;
  $: contributionSel = result?.weightedValueContributions?.[lastIndex]?.[selectedKey] || [];
  $: contextSel = result?.perTokenContext?.[lastIndex] || [];
  $: weightsRow = result?.weights?.[lastIndex] || [];
  $: dimCount = contributionSel.length || contextSel.length || 0;
  $: vSel = result?.v?.[selectedKey]?.[selectedDim] ?? 0;
  $: contributionDim = contributionSel[selectedDim] ?? 0;
  $: contextDim = contextSel[selectedDim] ?? 0;

  $: forceFromActionScore = forceFromScore(result?.actionScore ?? 0);

  // Action-stage chain: the frozen final hidden vector h[last] dotted with
  // the real learned action-head weight row, plus its bias, reproduces the
  // already-computed actionScore above. All read from the frozen `result`
  // object -- no second inference, same convention as qkProductsSel/dotSum
  // in the Calculation stage above.
  $: hiddenVec = result?.hidden?.[lastIndex] || [];
  $: actionWeightRow = result?.modelWeights?.action?.weight?.[0] || [];
  $: actionBias = result?.modelWeights?.action?.bias?.[0];
  $: hasActionHead = hiddenVec.length > 0 && actionWeightRow.length === hiddenVec.length && Number.isFinite(actionBias);

  let selectedHiddenDim = 0;
  let selectedHiddenDimInit = false;
  $: if (!selectedHiddenDimInit && result) { selectedHiddenDim = 0; selectedHiddenDimInit = true; }

  $: hSel = hiddenVec[selectedHiddenDim] ?? 0;
  $: wActionSel = actionWeightRow[selectedHiddenDim] ?? 0;
  $: actionProductSel = hSel * wActionSel;
  $: actionProducts = hiddenVec.map((h, j) => h * (actionWeightRow[j] ?? 0));
  $: actionProductsSum = actionProducts.reduce((sum, x) => sum + x, 0);

  // Switching stages never auto-opens the full advanced detail drawer — that
  // used to recreate the exact detached/vertical-overload problem this guide
  // fixes. The drawer only opens when the user explicitly clicks the
  // full-detail button below (Calculation/Action), and closes again on any
  // stage change so it doesn't linger behind a later stage.
  function goStage(next) {
    stage = next;
    onSelectToken(lastIndex);
    onExpandedStageChange(null);
  }

  function openFullDetail(kind) {
    onExpandedStageChange(kind);
  }

  function handleTabKey(e, idx) {
    let nextIdx = idx;
    if (e.key === 'ArrowRight') nextIdx = (idx + 1) % STAGES.length;
    else if (e.key === 'ArrowLeft') nextIdx = (idx - 1 + STAGES.length) % STAGES.length;
    else if (e.key === 'Home') nextIdx = 0;
    else if (e.key === 'End') nextIdx = STAGES.length - 1;
    else return;
    e.preventDefault();
    goStage(STAGES[nextIdx][0]);
    tick().then(() => tabRefs[nextIdx]?.focus());
  }

  async function applyStep() {
    // `applying` is set synchronously, before any await, so two clicks
    // dispatched back-to-back in the same task (e.g. a double DOM click)
    // both see the guard: the second call returns immediately instead of
    // calling onApplyStep() a second time before the first has resolved.
    if (stepApplied || applying || status === 'fell') return;
    applying = true;
    try {
      onApplyStep();
      // onApplyStep mutates the App-level lastDecisionTrace; tick() flushes
      // that prop update down to us before we check it, so this never
      // reports success on a step that didn't actually happen.
      await tick();
      if (lastDecisionTrace && lastDecisionTrace.tickFrom === capturedTick) {
        stepApplied = true;
        applyFailed = false;
      } else {
        applyFailed = true;
      }
    } finally {
      applying = false;
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

  <nav class="fd-repo-row" aria-label="cartpole teaching suite">
    <span>PPO · learning</span>
    <span>Transformer · mixing</span>
    <span>Diffusion · generation</span>
  </nav>

  <div class="fd-stages" role="tablist" aria-label="decision stages">
    {#each STAGES as [key,en,ko],i}
      <button
        type="button"
        role="tab"
        id="fd-tab-{key}"
        aria-selected={stage===key}
        aria-controls="fd-panel"
        tabindex={stage===key ? 0 : -1}
        bind:this={tabRefs[i]}
        class:active={stage===key}
        on:click={()=>goStage(key)}
        on:keydown={(e)=>handleTabKey(e,i)}
      >{en} · {ko}</button>
    {/each}
  </div>

  <div class="fd-content" role="tabpanel" id="fd-panel" aria-labelledby="fd-tab-{stage}" tabindex="0">
    {#if stage === 'input'}
      <p>Input / 입력 — 실제 입력은 tick 하나가 아니라 {result?.rawTokens?.length || 0} token × {inputValues.length}D 전체 history입니다. 아래 token을 선택해 물리값을 확인하세요 (frozen tick {capturedTick}).</p>
      <div class="fd-token-select" role="group" aria-label="select input history token">
        {#each result?.rawTokens || [] as _,i}
          <button type="button" class:active={i===selectedInputToken} on:click={()=>selectedInputToken=i}>{label(i)}</button>
        {/each}
      </div>
      <div class="fd-values">
        {#each FIELDS as [name,unit],i}
          <span><b>{name} [{unit}]</b>{(result?.rawTokens?.[selectedInputToken]?.[i] ?? inputValues[i]).toFixed(3)}</span>
        {/each}
      </div>
    {:else if stage === 'calculation'}
      <p>Calculation / 계산 — score = Σ(QᵢKᵢ) ÷ √d → causal mask → softmax → weight × V. Query는 t (tick {capturedTick})로 고정, 아래에서 Key token과 value dimension을 선택하세요.</p>
      <div class="fd-selectors">
        <div class="fd-key-select" role="group" aria-label="select key token to inspect">
          <span class="fd-selector-label">Key</span>
          {#each weightsRow as _,i}
            <button type="button" class:active={i===selectedKey} on:click={()=>selectedKey=i}>{label(i)}</button>
          {/each}
        </div>
        <div class="fd-dim-select" role="group" aria-label="select value dimension">
          <span class="fd-selector-label">dim</span>
          {#each Array(dimCount) as _,d}
            <button type="button" class:active={d===selectedDim} on:click={()=>selectedDim=d}>d{d}</button>
          {/each}
        </div>
      </div>
      <div class="fd-weight-bars" role="group" aria-label="attention weight per key token, 0 to 1 scale">
        {#each weightsRow as w,i}
          <div class="fd-weight-bar" class:active={i===selectedKey}>
            <span class="fd-weight-bar-label">{label(i)}</span>
            <span class="fd-weight-bar-track"><span class="fd-weight-bar-fill" style="width:{(w*100).toFixed(1)}%"></span></span>
            <span class="fd-weight-bar-value">{(w*100).toFixed(1)}%</span>
          </div>
        {/each}
      </div>
      <div class="fd-calc-steps">
        <div class="fd-calc-step">
          <b>A · Q·K → score</b>
          <details class="fd-vector-details">
            <summary>dimension별 곱 전체 보기 · full per-dimension products</summary>
            <code>{qkProductsSel.map(x=>x.toFixed(3)).join(' + ')}</code>
          </details>
        </div>
        <div class="fd-calc-step">
          <b>B · Σ ÷ √d = score</b>
          <code>{dotSum.toFixed(4)} ÷ {scaleDim.toFixed(3)} = {scaledScore.toFixed(4)}</code>
        </div>
        <div class="fd-calc-step">
          <b>C · Causal mask</b>
          {#if maskedSel}
            <code>{label(selectedKey)} is future of {label(lastIndex)} → −∞, weight forced to 0</code>
          {:else}
            <code>{label(selectedKey)} is allowed (key ≤ query tick) → score passes through unchanged</code>
          {/if}
        </div>
        <div class="fd-calc-step">
          <b>D · Stable softmax</b>
          <code>exp(score − row max) = {expValSel.toExponential(3)} · Σexp (row) = {denomSel.toFixed(5)} · weight = {(weightSel*100).toFixed(3)}%</code>
        </div>
        <div class="fd-calc-step" data-weight={weightSel} data-v-dim={vSel} data-contribution-dim={contributionDim}>
          <b>E · weight × V[key][dim]</b>
          <code>{(weightSel*100).toFixed(3)}% × V[{label(selectedKey)}][d{selectedDim}]({vSel.toFixed(3)}) = {contributionDim.toFixed(3)}</code>
          <details class="fd-vector-details">
            <summary>contribution vector 전체 보기 · full contribution vector</summary>
            <code>[{contributionSel.map(x=>x.toFixed(3)).join(', ')}]</code>
          </details>
        </div>
        <div class="fd-calc-step" data-context-dim={contextDim}>
          <b>Summed context[d{selectedDim}]</b>
          <code>context[d{selectedDim}] = {contextDim.toFixed(3)}</code>
          <details class="fd-vector-details">
            <summary>context vector 전체 보기 · full context vector</summary>
            <code>[{contextSel.map(x=>x.toFixed(3)).join(', ')}]</code>
          </details>
        </div>
      </div>
      <p class="fd-caveat">attention weight는 정보를 섞을 뿐 action 확률도 causal 중요도도 아닙니다. weighted V는 힘(force)이 아닙니다.</p>
      <button type="button" class="fd-open-detail" on:click={()=>openFullDetail('attention')}>전체 Self Attention 상세 열기 · open full Self Attention detail</button>
    {:else if stage === 'action'}
      <p>Action / 행동 — force = 10·tanh(action score). tick {capturedTick}에서 실제로 나온 score와 force입니다.</p>
      <div class="fd-values">
        <span><b>captured tick</b>{capturedTick}</span>
        <span><b>action score</b>{(result?.actionScore ?? 0).toFixed(6)}</span>
        <span><b>10·tanh(score) [N]</b>{forceFromActionScore.toFixed(2)}</span>
        <span><b>force command [N]</b>{controllerForce.toFixed(2)}</span>
      </div>
      {#if hasActionHead}
        <div class="fd-selectors">
          <div class="fd-action-dim-select" role="group" aria-label="select final hidden dimension">
            <span class="fd-selector-label">dim</span>
            {#each hiddenVec as _,d}
              <button type="button" class:active={d===selectedHiddenDim} on:click={()=>selectedHiddenDim=d}>d{d}</button>
            {/each}
          </div>
        </div>
        <div class="fd-calc-steps">
          <div class="fd-calc-step">
            <b>context → output projection/residual → LayerNorm/MLP → final hidden</b>
            <code>attention context 자체가 아니라, output projection + residual + LayerNorm + MLP를 모두 거친 뒤의 final hidden h[t]를 사용합니다 · uses the final hidden h[t] after output projection, residual, LayerNorm and the MLP — not the attention context alone.</code>
          </div>
          <div class="fd-calc-step" data-action-product={actionProductSel} data-action-h={hSel} data-action-w={wActionSel}>
            <b>H · h[j] × action weight[0][j]</b>
            <code>h[d{selectedHiddenDim}]({hSel.toFixed(4)}) × w_action[0][d{selectedHiddenDim}]({wActionSel.toFixed(4)}) = {actionProductSel.toFixed(6)}</code>
            <details class="fd-vector-details">
              <summary>전체 hidden dimension 곱 보기 · full per-dimension products</summary>
              <code>{actionProducts.map(x=>x.toFixed(4)).join(' + ')}</code>
            </details>
          </div>
          <div class="fd-calc-step" data-action-sum={actionProductsSum} data-action-bias={actionBias} data-action-score-check={actionProductsSum + actionBias}>
            <b>I · Σ(h · w_action) + bias = action score</b>
            <code>{actionProductsSum.toFixed(6)} + {actionBias.toFixed(6)} = {(actionProductsSum + actionBias).toFixed(6)}</code>
          </div>
        </div>
        <p class="fd-caveat">hidden component 하나의 기여도가 곧 force나 causal 중요도를 뜻하지 않습니다 · one hidden component's contribution here is not itself a force or a causal-importance claim.</p>
      {:else}
        <p class="fd-caveat">이 기록에는 학습된 action head 가중치가 없어 dimension별 계산을 표시할 수 없습니다 · learned action-head weights are not present on this recorded event, so the per-dimension breakdown cannot be shown.</p>
      {/if}
      <button type="button" class="fd-open-detail" on:click={()=>openFullDetail('action')}>전체 Action head 상세 열기 · open full Action head detail</button>
    {:else}
      <p>Result / 결과 — apply exactly one real 20&nbsp;ms physics step from this frozen event to see what actually happens next.</p>
      {#if !capturedTrace}
        <button type="button" class="fd-apply" disabled={status==='fell' || applying} on:click={applyStep}>Apply one 20ms step · 1 step 실행</button>
        {#if applyFailed}
          <p class="fd-error">The step did not apply (simulation state changed under the guide). Close and re-open Follow one decision to capture a fresh event.</p>
        {/if}
      {:else}
        <div class="fd-result-summary">
          <div class="fd-result-tick">actual tick {capturedTrace.tickFrom} → {capturedTrace.tickTo}</div>
          <div class="fd-values">
            <span><b>applied force [N]</b>{capturedTrace.appliedPolicyForce.toFixed(2)}</span>
          </div>
          <div class="fd-before-after">
            <div>
              <span class="fd-ba-label">before (t)</span>
              <div class="fd-values">
                {#each FIELDS as [name,unit],i}
                  <span><b>{name} [{unit}]</b>{capturedTrace.beforeState[i].toFixed(3)}</span>
                {/each}
              </div>
            </div>
            <div>
              <span class="fd-ba-label">after (t+1)</span>
              <div class="fd-values">
                {#each FIELDS as [name,unit],i}
                  <span><b>{name} [{unit}]</b>{capturedTrace.nextState[i].toFixed(3)}</span>
                {/each}
              </div>
            </div>
          </div>
        </div>
        <details class="fd-details">
          <summary>동역학 계산 상세 · full dynamics detail</summary>
          <DecisionTrace trace={capturedTrace}/>
        </details>
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
.fd-repo-row{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:14px;color:#7d8593;margin:10px 0 0}
.fd-stages{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:12px}
.fd-stages button{border:1px solid #d9dde5;border-radius:8px;background:#fff;padding:10px 8px;min-height:44px;font-size:14px;cursor:pointer;color:#4b5563}
.fd-stages button.active{background:#243047;color:#fff;border-color:#243047}
.fd-content{margin-top:12px}
.fd-content p{font-size:14px;line-height:1.5;color:#596273;margin:0 0 10px}
.fd-error{color:#a3401b}
.fd-values{display:flex;flex-wrap:wrap;gap:6px}
.fd-values span{display:flex;justify-content:space-between;gap:6px;flex:1;min-width:110px;border:1px solid #e4e7ec;background:#fafbfc;border-radius:6px;padding:6px 7px;font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#596273}
.fd-values b{font-family:Inter,ui-sans-serif,system-ui;font-size:14px;color:#8b93a1}
.fd-token-select,.fd-key-select,.fd-dim-select,.fd-action-dim-select{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:10px}
.fd-selectors{display:flex;flex-wrap:wrap;gap:6px 16px}
.fd-selector-label{font-size:14px;color:#8b93a1;min-width:28px}
.fd-token-select button,.fd-key-select button,.fd-dim-select button,.fd-action-dim-select button{border:1px solid #d9dde5;border-radius:8px;background:#fff;min-height:44px;min-width:44px;padding:8px 10px;font-size:14px;color:#4b5563;cursor:pointer}
.fd-token-select button.active,.fd-key-select button.active,.fd-dim-select button.active,.fd-action-dim-select button.active{background:#ece7f7;border-color:#a895cf;color:#5e4894}
.fd-weight-bars{display:flex;flex-direction:column;gap:7px;margin-bottom:10px}
.fd-weight-bar{display:grid;grid-template-columns:40px 1fr 60px;align-items:center;gap:8px}
.fd-weight-bar-label{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#596273}
.fd-weight-bar-track{height:16px;border-radius:8px;background:#eef0f4;overflow:hidden}
.fd-weight-bar-fill{display:block;height:100%;background:#a895cf;border-radius:8px}
.fd-weight-bar.active .fd-weight-bar-fill{background:#674e9f}
.fd-weight-bar-value{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#596273;text-align:right}
.fd-calc-steps{display:flex;flex-direction:column;gap:8px;margin-bottom:10px}
.fd-calc-step{border:1px solid #e4e7ec;background:#fafbfc;border-radius:8px;padding:9px 10px}
.fd-calc-step b{display:block;font-size:14px;color:#596273;margin-bottom:4px}
.fd-calc-step code{display:block;font-size:14px;color:#4b5563;white-space:normal;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.5}
.fd-vector-details{margin-top:6px}
.fd-vector-details summary{font-size:14px;cursor:pointer;color:#7d6aaa;min-height:44px;display:flex;align-items:center}
.fd-vector-details code{margin-top:4px}
.fd-caveat{font-size:14px;color:#7b8492;font-style:italic}
.fd-apply,.fd-new{margin-top:6px;border:1px solid #243047;border-radius:8px;background:#243047;color:#fff;padding:10px 14px;min-height:44px;font-size:14px;cursor:pointer}
.fd-apply:disabled,.fd-new:disabled{opacity:.4;cursor:not-allowed}
.fd-open-detail{margin-top:4px;border:1px solid #d9dde5;border-radius:8px;background:#fff;color:#4b5563;padding:10px 14px;min-height:44px;font-size:14px;cursor:pointer}
.fd-result-summary{border:1px solid #ded4f3;background:#f8f6fd;border-radius:10px;padding:10px;margin-bottom:10px}
.fd-result-tick{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#5d4b92;margin-bottom:8px}
.fd-before-after{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.fd-ba-label{display:block;font-size:14px;font-weight:700;color:#674e9f;margin-bottom:4px}
.fd-details{margin-bottom:10px}
.fd-details summary{font-size:14px;cursor:pointer;padding:8px 0;color:#4b5563;min-height:44px;display:flex;align-items:center}
@media(max-width:560px){.fd-stages{grid-template-columns:repeat(2,1fr)}.fd-before-after{grid-template-columns:1fr}.fd-weight-bar{grid-template-columns:34px 1fr 52px}}
</style>
