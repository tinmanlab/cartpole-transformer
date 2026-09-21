<script>
  import { onMount } from 'svelte';
  import CartPoleView from './components/CartPoleView.svelte';
  import Pipeline from './components/Pipeline.svelte';
  import TransformerDetail from './components/TransformerDetail.svelte';
  import VisionPipeline from './components/VisionPipeline.svelte';
  import VisionDetail from './components/VisionDetail.svelte';
  import FusionPipeline from './components/FusionPipeline.svelte';
  import FusionDetail from './components/FusionDetail.svelte';
  import ComparisonLab from './components/ComparisonLab.svelte';
  import DecisionTrace from './components/DecisionTrace.svelte';
  import FollowDecisionGuide from './components/FollowDecisionGuide.svelte';
  import { resetState, computeCartPoleTransition, terminal, stateArray, PHYSICS } from './lib/physics.js';
  import { runAttention, forceFromScore } from './lib/attention.js';
  import { loadLearnedModel, runLearnedAttention } from './lib/learned_attention.js';
  import {
    visionObservationFromState,
    sampleVisionHistory,
    VISION_BUFFER_LENGTH,
    VISION_SEQUENCE_LENGTH
  } from './lib/vision.js';
  import { loadVisionModel, runVisionAttention } from './lib/vision_attention.js';
  import { loadFusionModel, runFusionAttention } from './lib/fusion_attention.js';

  const N = 8;
  let mode = 'state';
  let state = resetState();
  let history = Array.from({length:N}, () => stateArray(state));

  let selectedToken = N - 1;
  let selectedRow = N - 1;
  let selectedCol = N - 1;
  let selectedDim = 0;
  let expandedStage = null;

  let selectedVisionFrame = VISION_SEQUENCE_LENGTH - 1;
  let visionDetailOpen = false;

  let selectedFusionToken = VISION_SEQUENCE_LENGTH * 2 - 1;
  let fusionScenario = 'clean';
  let fusionDetailOpen = false;

  let learnedModel = null;
  let modelState = 'loading';
  let visionModel = null;
  let visionModelState = 'loading';
  let fusionModel = null;
  let fusionModelState = 'loading';

  let result = runAttention(history);

  const initialStateArray = stateArray(state);
  let initialVisionObservation = visionObservationFromState(initialStateArray);
  let visionBuffer = Array.from({length:VISION_BUFFER_LENGTH},()=>({
    frame:[...initialVisionObservation.frame],
    patches:[...initialVisionObservation.patches],
    state:[...initialStateArray]
  }));

  let visionSamples = sampleVisionHistory(visionBuffer);
  let visionFrames = visionSamples.map(x=>x.frame);
  let visionPatchHistory = visionSamples.map(x=>x.patches);
  let fusionStateHistory = visionSamples.map(x=>x.state);

  let visionResult = null;
  let visionRepeatedResult = null;
  let fusionResult = null;
  $: comparisonModels = { state:learnedModel, vision:visionModel, fusion:fusionModel };

  $: frameIntervalMs = (
    fusionModel?.frame_stride_seconds ||
    visionModel?.frame_stride_seconds ||
    PHYSICS.tau * 3
  ) * 1000;

  let controllerForce = forceFromScore(result.actionScore);
  let disturbance = 0;
  let running = true;
  let elapsed = 0;
  let status = 'balancing';
  let raf = 0;
  let lastDecisionTrace = null;

  let followDecisionOpen = false;
  let followDecisionSeq = 0;
  // Frozen once at open time: `result` is reassigned wholesale (never
  // mutated in place) on every refreshCurrentInference, so holding this
  // reference is enough to keep the captured event's tensors immutable even
  // after Apply-step advances the live `result`/`controllerForce` below.
  let followDecisionSnapshot = null;

  // The compact token-weight summary below is now the primary default-entry
  // reading; the full Pipeline/advanced graph stays a closed native
  // <details> overview on ordinary entry (never auto-opened), is force-
  // closed while the guide owns the workspace, and is reopenable at any
  // time by the user's own toggle — but no longer auto-reopens itself when
  // the guide closes, since that used to bury the summary again.
  let pipelineOpen = false;
  $: if (followDecisionOpen) pipelineOpen = false;

  // Reopening the Pipeline overview from inside the guided workspace must
  // show the SAME frozen event the guide/TransformerDetail are showing, not
  // the live plant (which has already advanced past the captured tick once
  // Apply runs) — same source convention as TransformerDetail's
  // followDecisionOpen/followDecisionSnapshot check below.
  $: pipelineSourceFrozen = followDecisionOpen && !!followDecisionSnapshot;
  $: pipelineResult = pipelineSourceFrozen ? followDecisionSnapshot.result : result;
  $: pipelineControllerForce = pipelineSourceFrozen ? followDecisionSnapshot.controllerForce : controllerForce;
  $: pipelineSourceLabel = pipelineSourceFrozen
    ? 'FROZEN captured tick ' + followDecisionSnapshot.tick + ' · LIVE current tick ' + syncTick
    : 'LIVE tick ' + syncTick;

  // Compact ACTUAL 8-token weighting summary (primary default-entry reading,
  // ahead of the collapsed Pipeline disclosure): the real final-query row of
  // result.weights, top-3 by value + remaining mass. Not an action
  // probability or causal-importance ranking (see docs/learning-suite.md).
  // Only rendered while the guide is closed, so there is exactly one
  // instance of this reading at a time (App.svelte:selectKey below is the
  // same App-owned SSOT the shared full inspector and the guide both use).
  $: summaryLast = N - 1;
  $: summaryFinalRow = result?.weights?.[summaryLast] || [];
  $: summaryTopWeights = summaryFinalRow
    .map((w,i) => ({i,w}))
    .sort((a,b) => b.w - a.w)
    .slice(0,3);
  $: summaryTopMass = summaryTopWeights.reduce((sum,t) => sum + t.w, 0);
  $: summaryRemainingMass = Math.max(0, 1 - summaryTopMass);

  function openSummaryWeight(i) {
    selectKey(i);
    expandedStage = 'attention';
  }

  $: decisionTraceSummary = lastDecisionTrace
    ? 'tick ' + lastDecisionTrace.tickFrom + ' → ' + lastDecisionTrace.tickTo + ' · ' + status
    : 'no step recorded yet · ' + status;

  $: syncTick = Math.round(elapsed / PHYSICS.tau);
  $: syncStateVector = stateArray(state);
  $: syncStateToken = result?.rawTokens?.[result.rawTokens.length - 1] || [];
  $: syncVisionState = fusionStateHistory?.[fusionStateHistory.length - 1] || [];
  $: activeActionScore = mode === 'fusion'
    ? fusionResult?.actionScore
    : mode === 'vision'
      ? visionResult?.actionScore
      : result?.actionScore;

  function inferState(sequence) {
    return learnedModel ? runLearnedAttention(sequence, learnedModel) : runAttention(sequence);
  }

  function fusionOptions() {
    if (fusionScenario === 'noisy-state') return {stateNoise:0.18};
    if (fusionScenario === 'missing-state') return {stateAvailable:false};
    if (fusionScenario === 'partial-vision') return {partialVision:true};
    if (fusionScenario === 'missing-vision') return {visionAvailable:false};
    return {};
  }

  function refreshFusion() {
    if (
      fusionModel &&
      fusionStateHistory.length === VISION_SEQUENCE_LENGTH &&
      visionPatchHistory.length === VISION_SEQUENCE_LENGTH
    ) {
      fusionResult = runFusionAttention(
        fusionStateHistory,
        visionPatchHistory,
        fusionModel,
        fusionOptions()
      );
    }
  }

  function refreshVision() {
    visionSamples = sampleVisionHistory(visionBuffer);
    visionFrames = visionSamples.map(x=>x.frame);
    visionPatchHistory = visionSamples.map(x=>x.patches);
    fusionStateHistory = visionSamples.map(x=>x.state);

    if (visionModel && visionPatchHistory.length === VISION_SEQUENCE_LENGTH) {
      visionResult = runVisionAttention(visionPatchHistory, visionModel);
      const latest = visionPatchHistory[visionPatchHistory.length - 1];
      const repeated = Array.from({length:VISION_SEQUENCE_LENGTH},()=>[...latest]);
      visionRepeatedResult = runVisionAttention(repeated, visionModel);
    }
    refreshFusion();
  }

  function currentActionScore() {
    if (mode === 'fusion' && fusionResult) return fusionResult.actionScore;
    if (mode === 'vision' && visionResult) return visionResult.actionScore;
    return result.actionScore;
  }

  function activeForce() {
    return forceFromScore(currentActionScore());
  }

  function currentObservationTrace() {
    if (mode === 'vision') {
      return {
        kind:'vision',
        frameCount:visionFrames.length,
        inputDim:visionResult?.tokenInputs?.[visionResult.tokenInputs.length - 1]?.length || 0,
        frameSpanMs:(VISION_SEQUENCE_LENGTH - 1) * frameIntervalMs
      };
    }
    if (mode === 'fusion') {
      return {
        kind:'fusion',
        tokenCount:fusionResult?.tokens?.length || 0,
        stateAvailable:fusionResult?.stateAvailable !== false,
        visionAvailable:fusionResult?.visionAvailable !== false
      };
    }
    return {
      kind:'state',
      values:[...(result?.rawTokens?.[result.rawTokens.length - 1] || stateArray(state))]
    };
  }

  function refreshCurrentInference(updateForce = mode !== 'compare') {
    result = inferState(history);
    refreshVision();
    if (updateForce) controllerForce = activeForce();
  }

  function advanceOneTick() {
    if (mode === 'compare' || status === 'fell') return false;

    const tickFrom = Math.round(elapsed / PHYSICS.tau);
    const appliedPolicyForce = controllerForce;
    const appliedActionScore = currentActionScore();
    const controllerObservation = currentObservationTrace();

    // The environment returns the exact intermediates used for this plant step.
    const plant = computeCartPoleTransition(
      state,
      appliedPolicyForce,
      PHYSICS.tau,
      disturbance
    );

    state = plant.nextState;
    const nextStateArray = stateArray(state);
    history = [...history.slice(1), nextStateArray];

    const observation = visionObservationFromState(nextStateArray);
    visionBuffer = [
      ...visionBuffer.slice(1),
      {
        frame:[...observation.frame],
        patches:[...observation.patches],
        state:[...nextStateArray]
      }
    ];

    elapsed += PHYSICS.tau;

    // Recompute the next controller snapshot only after the plant state/history
    // has advanced, so u_(t+1) is distinct from the applied u_t.
    refreshCurrentInference(true);

    lastDecisionTrace = {
      mode,
      tickFrom,
      tickTo:tickFrom + 1,
      observation:controllerObservation,
      beforeState:stateArray(plant.state),
      appliedActionScore,
      appliedPolicyForce,
      appliedControl:plant.control,
      disturbance:plant.disturbance,
      totalForce:plant.totalForce,
      temp:plant.temp,
      xAcc:plant.xAcc,
      thetaAcc:plant.thetaAcc,
      dt:plant.dt,
      nextState:[...nextStateArray],
      nextActionScore:currentActionScore(),
      nextPolicyForce:controllerForce
    };

    if (terminal(state)) {
      status = 'fell';
      running = false;
      disturbance = 0;
    }
    return true;
  }

  function stepOnce() {
    if (running || mode === 'compare' || status === 'fell') return;
    advanceOneTick();
  }

  function setMode(next) {
    if (next === 'vision' && visionModelState !== 'learned') return;
    if (next === 'fusion' && fusionModelState !== 'learned') return;
    if (next === 'compare' && (modelState !== 'learned' || visionModelState !== 'learned' || fusionModelState !== 'learned')) return;

    pushEnd();
    mode = next;
    lastDecisionTrace = null;
    expandedStage = null;
    visionDetailOpen = false;
    fusionDetailOpen = false;
    followDecisionOpen = false;
    followDecisionSnapshot = null;

    selectedToken = N - 1;
    selectedRow = N - 1;
    selectedCol = N - 1;
    selectedDim = clampDim(0);
    selectedVisionFrame = VISION_SEQUENCE_LENGTH - 1;
    selectedFusionToken = VISION_SEQUENCE_LENGTH * 2 - 1;

    refreshCurrentInference(next !== 'compare');
  }

  function setFusionScenario(next) {
    fusionScenario = next;
    lastDecisionTrace = null;
    refreshFusion();
    if (mode === 'fusion') controllerForce = activeForce();
  }

  // Central query lock: while the guide is open, Query (selectedRow) must
  // stay pinned to the last token no matter which caller tries to move it --
  // Pipeline's embedding/qkv token hover and its attention-matrix hover both
  // call selectToken/selectAttention directly, same as the shared drawer, so
  // the clamp has to live here (the one place both paths funnel through),
  // not just in TransformerDetail's own forwarding wrapper. Outside the
  // guide this is a no-op and Query/Key stay fully free.
  function selectToken(i) {
    selectedToken = i;
    selectedRow = followDecisionOpen ? N - 1 : i;
    selectedCol = i;
  }

  function selectAttention(r,c) {
    selectedRow = followDecisionOpen ? N - 1 : r;
    selectedCol = c;
    selectedToken = c;
  }

  // Guide-owned Key/dim selectors: while the guide is open, Query stays
  // locked to the last token (see lockQuery on TransformerDetail below), so
  // only Key/dim can move -- both are App state so the guide's own buttons
  // and the shared full-detail drawer read/write the exact same selection.
  // Equivalent to selectAttention(N-1, i) now that the lock is central, kept
  // as its own named function for readability at call sites.
  function selectKey(i) {
    selectAttention(N - 1, i);
  }

  // Clamp to the actual current context/hidden vector width (4D toy, 8D
  // learned) rather than assuming a fixed dimension count -- used both for
  // interactive selection and at deliberate lifecycle resets below.
  function clampDim(d) {
    const width = result?.perTokenContext?.[0]?.length ?? result?.context?.length;
    return Number.isFinite(width) && width > 0 ? Math.min(Math.max(0, d), width - 1) : Math.max(0, d);
  }

  function selectDim(d) {
    selectedDim = clampDim(d);
  }

  function reset() {
    state = resetState((Math.random()-.5)*.09);
    const initial = stateArray(state);
    history = Array.from({length:N}, () => [...initial]);

    const observation = visionObservationFromState(initial);
    visionBuffer = Array.from({length:VISION_BUFFER_LENGTH},()=>({
      frame:[...observation.frame],
      patches:[...observation.patches],
      state:[...initial]
    }));

    selectedToken = N - 1;
    selectedRow = N - 1;
    selectedCol = N - 1;
    selectedDim = clampDim(0);
    selectedVisionFrame = VISION_SEQUENCE_LENGTH - 1;
    selectedFusionToken = VISION_SEQUENCE_LENGTH * 2 - 1;

    expandedStage = null;
    visionDetailOpen = false;
    fusionDetailOpen = false;
    followDecisionOpen = false;
    followDecisionSnapshot = null;

    disturbance = 0;
    elapsed = 0;
    lastDecisionTrace = null;
    status = 'balancing';
    running = true;

    refreshCurrentInference(mode !== 'compare');
  }

  function toggle(){
    if (mode === 'compare' || status === 'fell') return;
    running = !running;
    pushEnd();
  }
  function push(v){ disturbance = v; }
  function pushEnd(){ disturbance = 0; }

  function openFollowDecision() {
    if (mode !== 'state' || modelState !== 'learned' || status === 'fell') return;
    running = false;
    pushEnd();
    followDecisionSnapshot = { tick: syncTick, result, controllerForce };
    followDecisionSeq += 1;
    followDecisionOpen = true;
    expandedStage = null;
    // A new decision event deliberately resets Query/Key/dim to the guide's
    // fixed last token -- it never inherits a selection left over from a
    // previous event or from free browsing outside the guide.
    selectedToken = N - 1;
    selectedRow = N - 1;
    selectedCol = N - 1;
    selectedDim = clampDim(0);
  }

  function closeFollowDecision() {
    followDecisionOpen = false;
    followDecisionSnapshot = null;
    expandedStage = null;
    selectedToken = N - 1;
    selectedRow = N - 1;
    selectedCol = N - 1;
    selectedDim = clampDim(0);
  }

  onMount(() => {
    let cancelled = false;

    loadLearnedModel()
      .then(model => {
        if (cancelled) return;
        learnedModel = model;
        modelState = 'learned';
        refreshCurrentInference(mode !== 'compare');
        // Toy (4D context) -> learned (8D hidden) is a real width change;
        // widening never needs a clamp, but re-clamp anyway so this stays
        // correct if a future model ever shipped a narrower width.
        selectedDim = clampDim(selectedDim);
      })
      .catch(() => {
        if (!cancelled) modelState = 'toy-fallback';
      });

    loadVisionModel()
      .then(model => {
        if (cancelled) return;
        visionModel = model;
        visionModelState = 'learned';
        refreshCurrentInference(mode !== 'compare');
      })
      .catch(() => {
        if (!cancelled) visionModelState = 'unavailable';
      });

    loadFusionModel()
      .then(model => {
        if (cancelled) return;
        fusionModel = model;
        fusionModelState = 'learned';
        refreshCurrentInference(mode !== 'compare');
      })
      .catch(() => {
        if (!cancelled) fusionModelState = 'unavailable';
      });

    let previous = performance.now(), accumulator = 0;
    const frame = now => {
      const delta = Math.min(.08, (now-previous)/1000);
      previous = now;

      if (running && mode !== 'compare') {
        accumulator += delta;
        while (accumulator >= PHYSICS.tau) {
          const advanced = advanceOneTick();
          accumulator -= PHYSICS.tau;
          if (!advanced || status === 'fell') break;
        }
      }

      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  });
</script>

<svelte:head>
  <title>Cart-Pole Transformer Explainer</title>
  <meta name="description" content="State, pixels-only and typed-token state+vision Transformer learning modes for live Cart-Pole."/>
</svelte:head>

<main
  data-observation-mode={mode}
  data-sync-tick={syncTick}
  data-sim-state={syncStateVector.join(',')}
  data-state-token={syncStateToken.join(',')}
  data-vision-sample-state={syncVisionState.join(',')}
  data-active-action-score={activeActionScore ?? ''}
  data-controller-force={controllerForce}
  data-model-state={modelState}
>
  <header class="topbar">
    <strong>Cart-Pole Transformer</strong>
    <div class="mode-switch" role="group" aria-label="observation mode">
      <button type="button" class:active={mode==='state'} on:click={()=>setMode('state')}>State</button>
      <button type="button" class:active={mode==='vision'} disabled={visionModelState!=='learned'} on:click={()=>setMode('vision')}>Vision</button>
      <button type="button" class:active={mode==='fusion'} disabled={fusionModelState!=='learned'} on:click={()=>setMode('fusion')}>Fusion</button>
      <button type="button" class:active={mode==='compare'} disabled={modelState!=='learned' || visionModelState!=='learned' || fusionModelState!=='learned'} on:click={()=>setMode('compare')}>Compare</button>
    </div>
    <span>
      {mode==='state'
        ? (modelState === 'learned' ? 'learned state · 1 head · 8D' : modelState === 'loading' ? 'loading state model…' : 'transparent fallback')
        : mode==='vision'
          ? 'pixels only · 8 frames · 1 head · 24D'
          : mode==='fusion'
            ? 'typed fusion · 16 tokens · 1 head · 24D'
            : 'deterministic replay · same initial state + disturbance'}
    </span>
  </header>

  {#if mode === 'compare'}
    <ComparisonLab models={comparisonModels}/>
  {:else}
  <section class="lab-grid" aria-label="live Cart-Pole and Transformer visualization">
    <CartPoleView
      {state}
      {history}
      selectedToken={mode==='state'?selectedToken:N-1}
      {controllerForce}
      {disturbance}
      {running}
      {elapsed}
      {status}
      showStateOverlay={mode!=='vision'}
      guideActive={followDecisionOpen}
      onToggle={toggle}
      onStep={stepOnce}
      onReset={reset}
      onPush={push}
      onPushEnd={pushEnd}
    />

    {#if mode === 'state'}
      <div class="state-column">
        {#if modelState === 'toy-fallback'}
          <div class="model-fallback-banner" role="status">⚠ learned state model failed to load — showing the transparent toy fallback (fixed weights, not learned)</div>
        {/if}
        <section class="follow-decision" aria-label="follow one decision guide">
          {#if !followDecisionOpen}
            <button
              type="button"
              class="follow-decision-entry"
              disabled={modelState!=='learned' || status==='fell'}
              on:click={openFollowDecision}
            >한 판단 따라가기 · Follow one decision</button>
          {:else}
            {#key followDecisionSeq}
              <FollowDecisionGuide
                capturedTick={followDecisionSnapshot.tick}
                eventId={followDecisionSeq}
                result={followDecisionSnapshot.result}
                controllerForce={followDecisionSnapshot.controllerForce}
                currentTick={syncTick}
                {lastDecisionTrace}
                {status}
                selectedKey={selectedCol}
                {selectedDim}
                onSelectKey={selectKey}
                onSelectDim={selectDim}
                onExpandedStageChange={(stage)=>expandedStage=stage}
                onApplyStep={stepOnce}
                onClose={closeFollowDecision}
                onNewDecision={openFollowDecision}
              />
            {/key}
          {/if}
        </section>

        <!-- Compact 8-token weighting summary: the primary default-entry
             reading, ahead of the collapsed Pipeline disclosure below. The
             guide above replaces this while open, so there is exactly one
             live summary instance at a time. -->
        {#if !followDecisionOpen}
          <section class="decision-summary" data-decision-summary aria-label="actual eight-token attention weighting summary">
            <div class="decision-summary-head">Actual 8-token history → final-query mixture weights → score → force</div>
            <div class="decision-summary-weights">
              {#each summaryTopWeights as t}
                <button type="button" class="decision-summary-weight" data-index={t.i} on:click={() => openSummaryWeight(t.i)}>
                  <span class="dsw-label">{t.i===summaryLast?'t':'t−'+(summaryLast-t.i)}</span>
                  <span class="dsw-value">{(t.w*100).toFixed(1)}%</span>
                </button>
              {/each}
              <span class="decision-summary-remaining">remaining mass {(summaryRemainingMass*100).toFixed(1)}%</span>
            </div>
            <p class="decision-summary-caveat">Mixture weights over the token history, not an action probability or a causal-importance ranking. <button type="button" class="decision-summary-open" on:click={() => openSummaryWeight(summaryTopWeights[0]?.i ?? summaryLast)}>open full Self Attention inspector</button></p>
            <div class="decision-summary-force"><b>score</b>{(result?.actionScore ?? 0).toFixed(4)}<b>→ 10·tanh →</b>{controllerForce>=0?'+':''}{controllerForce.toFixed(2)} N</div>
          </section>
        {/if}

        <!-- Full Pipeline/advanced graph: closed by default on ordinary
             entry now that the summary above is the primary reading; force-
             closed while the guide owns the workspace; always reopenable by
             the user via this same native <details>. -->
        <details class="disclosure-toggle pipeline-disclosure" bind:open={pipelineOpen} data-source={pipelineSourceFrozen ? 'frozen' : 'live'}>
          <summary>Pipeline overview · Embedding → Action (5 stages) · {pipelineSourceLabel}</summary>
          <Pipeline
            result={pipelineResult}
            controllerForce={pipelineControllerForce}
            {selectedToken}
            {selectedRow}
            {selectedCol}
            {expandedStage}
            onSelectToken={selectToken}
            onSelectAttention={selectAttention}
            onExpandedStageChange={(stage)=>expandedStage=stage}
          />
        </details>
      </div>
    {:else if mode === 'vision' && visionResult}
      <VisionPipeline
        frames={visionFrames}
        result={visionResult}
        {controllerForce}
        selectedFrame={selectedVisionFrame}
        onSelectFrame={(i)=>selectedVisionFrame=i}
        onOpenDetail={()=>visionDetailOpen=true}
        frameIntervalMs={frameIntervalMs}
      />
    {:else if mode === 'fusion' && fusionResult}
      <FusionPipeline
        result={fusionResult}
        {controllerForce}
        scenario={fusionScenario}
        onScenarioChange={setFusionScenario}
        selectedToken={selectedFusionToken}
        onSelectToken={(i)=>selectedFusionToken=i}
        onOpenDetail={()=>fusionDetailOpen=true}
        frameIntervalMs={frameIntervalMs}
      />
    {/if}
  </section>
    {#if mode === 'state'}
      <!-- Secondary detailed physical transition: closed by default so it
           never buries the primary guide/Pipeline column above it; the
           summary always shows the real tick range/status so users know
           what's inside without opening it. -->
      <details class="disclosure-toggle decision-trace-disclosure">
        <summary>Plant transition detail · {decisionTraceSummary}</summary>
        <DecisionTrace trace={lastDecisionTrace}/>
      </details>
    {:else}
      <DecisionTrace trace={lastDecisionTrace}/>
    {/if}
  {/if}

  {#if mode === 'vision' && visionDetailOpen && visionResult}
    <VisionDetail
      frames={visionFrames}
      result={visionResult}
      repeatedResult={visionRepeatedResult}
      selectedFrame={selectedVisionFrame}
      groundTruthState={stateArray(state)}
      onSelectFrame={(i)=>selectedVisionFrame=i}
      onClose={()=>visionDetailOpen=false}
      frameIntervalMs={frameIntervalMs}
    />
  {:else if mode === 'fusion' && fusionDetailOpen && fusionResult && fusionModel}
    <FusionDetail
      result={fusionResult}
      model={fusionModel}
      scenario={fusionScenario}
      onScenarioChange={setFusionScenario}
      selectedToken={selectedFusionToken}
      onSelectToken={(i)=>selectedFusionToken=i}
      groundTruthState={stateArray(state)}
      onClose={()=>fusionDetailOpen=false}
      frameIntervalMs={frameIntervalMs}
    />
  {/if}

  {#if mode === 'state'}
    <!-- Placed after the guide (only for State) so the guide's self-contained
         arithmetic is never pushed below this much larger advanced drawer.
         While the guide is open, this shared detail drawer must show the
         frozen event's tensors (Calculation/Action stages), not the live
         result — live advances past the captured tick the instant Apply
         runs one real step. -->
    <TransformerDetail
      result={followDecisionOpen && followDecisionSnapshot ? followDecisionSnapshot.result : result}
      controllerForce={followDecisionOpen && followDecisionSnapshot ? followDecisionSnapshot.controllerForce : controllerForce}
      {selectedToken}
      {selectedRow}
      {selectedCol}
      {expandedStage}
      source={followDecisionOpen && followDecisionSnapshot ? 'frozen' : 'live'}
      lockQuery={followDecisionOpen}
      highlightDim={selectedDim}
      onClose={()=>expandedStage=null}
      onSelectToken={selectToken}
      onSelectAttention={selectAttention}
      onSelectDim={selectDim}
    />
  {/if}

  <details class="disclosure-toggle explain-disclosure">
    <summary>읽는 순서 참고 · Reading order reference</summary>
    <section class="explain">
    <div class="explain-head">
      <h2>{mode==='state'?'State 읽는 순서':mode==='vision'?'Vision-only 읽는 순서':mode==='fusion'?'State + Vision 읽는 순서':'Deterministic 비교 replay'}</h2>

      {#if mode==='state'}
        <div class="qkv-key"><span class="q">Q</span> 찾는 기준 <span class="k">K</span> 비교 표지 <span class="v">V</span> 가져올 내용</div>
      {:else if mode==='vision'}
        <div class="vision-note">explicit simulator state is hidden from the controller</div>
      {:else if mode==='fusion'}
        <div class="vision-note">same episode · aligned timestamps · typed-token self-attention</div>
      {:else}
        <div class="vision-note">three independent environments · identical start · identical disturbance ticks</div>
      {/if}
    </div>

    {#if mode==='state'}
      {#if modelState==='toy-fallback'}
        <div class="claim">LEARNED MODEL REFERENCE — NOT ACTIVE IN TOY FALLBACK. 아래 5단계는 학습된 모델 기준 설명이며, 지금 실제로 동작 중인 toy fallback은 학습되지 않은 고정 recency bias(+1.20×j)와 고정 feedback gain으로 force를 계산합니다.</div>
      {/if}
      <div class="steps">
        <article><b>1. Embedding</b><p><code>[x,ẋ,θ,θ̇]</code> → normalize → learned 4→8 projection + position.</p></article>
        <article><b>2. Q / K / V</b><p>LayerNorm 뒤 같은 token을 세 learned projection으로 나눕니다.</p></article>
        <article><b>3. Attention</b><p><code>QKᵀ/√d → causal mask → softmax</code>. matrix cell을 가리키면 같은 시간쌍이 연결됩니다.</p></article>
        <article><b>4. Residual + MLP</b><p>Attention output을 더하고 LN→Linear→GELU→Linear→residual을 통과합니다.</p></article>
        <article><b>5. Action</b><p>마지막 hidden token만 읽어 <code>tanh(score)×10 N</code> force를 만듭니다.</p></article>
        <article><b>실시간 연결</b><p>time token 선택은 왼쪽 ghost pose와 전체 상세 계산에 동일하게 적용됩니다.</p></article>
      </div>
    {:else if mode==='vision'}
      <div class="steps">
        <article><b>1. Pixels</b><p>Controller는 32×32 rendered frame만 받습니다. <code>x, ẋ, θ, θ̇</code> 숫자는 입력하지 않습니다.</p></article>
        <article><b>2. Patches + Δ</b><p>16×16 patch means와 이전 sampled frame의 Δpatch를 concat해 512D visual input을 만듭니다.</p></article>
        <article><b>3. Frame token</b><p>512D visual feature를 learned Linear로 24D visual token으로 압축합니다.</p></article>
        <article><b>4. Temporal attention</b><p>현재 frame과 60 ms 간격의 과거 7장을 비교해 총 0.42 s의 움직임을 봅니다.</p></article>
        <article><b>5. Motion inference</b><p>마지막 hidden token이 pixels에서 state를 추정합니다. 특히 ẋ와 θ̇는 frame history가 핵심입니다.</p></article>
        <article><b>Ablation</b><p>상세 화면에서 8-frame 추정과 같은 최신 frame을 8번 반복한 결과를 직접 비교합니다.</p></article>
      </div>
    {:else if mode==='fusion'}
      <div class="steps">
        <article><b>1. Same timestamps</b><p>Vision과 같은 60 ms 샘플 시점에서 explicit state도 함께 snapshot합니다.</p></article>
        <article><b>2. Typed tokens</b><p>각 시점마다 State token 1개 + Vision token 1개. 총 <code>8×2=16</code> token입니다.</p></article>
        <article><b>3. Type + time</b><p>두 modality는 같은 time embedding을 공유하고 서로 다른 modality/type embedding으로 구분됩니다.</p></article>
        <article><b>4. One self-attention</b><p>별도 cross-attention 없이 16 token이 한 matrix에서 상호작용합니다. 같은 시점 S↔V는 허용됩니다.</p></article>
        <article><b>5. Degradation</b><p>Noisy/missing state, partial/missing vision을 같은 episode에서 즉시 바꿔 결과를 비교합니다.</p></article>
        <article><b>6. Negative result도 보존</b><p>단순 fusion이 항상 낫지는 않습니다. noisy-state에서는 vision 추가가 control을 악화시킨 결과도 그대로 보여줍니다.</p></article>
      </div>
    {:else}
      <div class="steps">
        <article><b>1. Identical start</b><p>세 controller는 <code>x=0, ẋ=0, θ=0.08 rad, θ̇=0</code>에서 동시에 시작합니다.</p></article>
        <article><b>2. Shared disturbance</b><p>모든 환경이 정확히 같은 simulation tick에 같은 ±4 N pulse를 받습니다.</p></article>
        <article><b>3. Independent physics</b><p>각 policy의 force가 다르므로 그 이후 trajectory는 독립적으로 갈라집니다.</p></article>
        <article><b>4. Freeze on failure</b><p>하나가 쓰러져도 그 pose에서 freeze하고 공통 replay clock은 계속 진행됩니다.</p></article>
        <article><b>5. Scrub</b><p>Pause 후 slider를 움직이면 세 controller가 같은 tick으로 함께 이동합니다.</p></article>
        <article><b>6. Raw metrics</b><p>survival, max |θ|, mean |θ|, control effort를 그대로 보여주며 별도 winner를 만들지 않습니다.</p></article>
      </div>
    {/if}
    </section>
  </details>

  <div class="claim">Cart-Pole에는 Transformer나 multimodal fusion이 필요하지 않습니다. 이 lab은 observation, temporal/multimodal attention, 그리고 동일 조건 replay에서 나타나는 실제 제어 차이를 관찰하기 위한 교육용 모델입니다.</div>

  <footer>Polo Club Transformer Explainer의 MIT-licensed VectorCanvas, MatrixSvg, Sankey, attention expansion/animation 패턴을 vendoring·adaptation하여 사용합니다.</footer>
</main>
