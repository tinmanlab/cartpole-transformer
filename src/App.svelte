<script>
  import { onMount } from 'svelte';
  import CartPoleView from './components/CartPoleView.svelte';
  import Pipeline from './components/Pipeline.svelte';
  import TransformerDetail from './components/TransformerDetail.svelte';
  import VisionPipeline from './components/VisionPipeline.svelte';
  import VisionDetail from './components/VisionDetail.svelte';
  import FusionPipeline from './components/FusionPipeline.svelte';
  import FusionDetail from './components/FusionDetail.svelte';
  import { resetState, stepCartPole, terminal, stateArray, PHYSICS } from './lib/physics.js';
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

  function activeForce() {
    if (mode === 'fusion' && fusionResult) return forceFromScore(fusionResult.actionScore);
    if (mode === 'vision' && visionResult) return forceFromScore(visionResult.actionScore);
    return forceFromScore(result.actionScore);
  }

  function setMode(next) {
    if (next === 'vision' && visionModelState !== 'learned') return;
    if (next === 'fusion' && fusionModelState !== 'learned') return;

    mode = next;
    expandedStage = null;
    visionDetailOpen = false;
    fusionDetailOpen = false;

    selectedToken = N - 1;
    selectedRow = N - 1;
    selectedCol = N - 1;
    selectedVisionFrame = VISION_SEQUENCE_LENGTH - 1;
    selectedFusionToken = VISION_SEQUENCE_LENGTH * 2 - 1;

    result = inferState(history);
    refreshVision();
    controllerForce = activeForce();
  }

  function setFusionScenario(next) {
    fusionScenario = next;
    refreshFusion();
    if (mode === 'fusion') controllerForce = activeForce();
  }

  function selectToken(i) {
    selectedToken = i;
    selectedRow = i;
    selectedCol = i;
  }

  function selectAttention(r,c) {
    selectedRow = r;
    selectedCol = c;
    selectedToken = c;
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
    selectedVisionFrame = VISION_SEQUENCE_LENGTH - 1;
    selectedFusionToken = VISION_SEQUENCE_LENGTH * 2 - 1;

    expandedStage = null;
    visionDetailOpen = false;
    fusionDetailOpen = false;

    result = inferState(history);
    refreshVision();
    controllerForce = activeForce();

    disturbance = 0;
    elapsed = 0;
    status = 'balancing';
    running = true;
  }

  function toggle(){ running = !running; }
  function push(v){ disturbance = v; }
  function pushEnd(){ disturbance = 0; }

  onMount(() => {
    let cancelled = false;

    loadLearnedModel()
      .then(model => {
        if (cancelled) return;
        learnedModel = model;
        modelState = 'learned';
        result = inferState(history);
        if (mode === 'state') controllerForce = activeForce();
      })
      .catch(() => {
        if (!cancelled) modelState = 'toy-fallback';
      });

    loadVisionModel()
      .then(model => {
        if (cancelled) return;
        visionModel = model;
        visionModelState = 'learned';
        refreshVision();
        if (mode === 'vision') controllerForce = activeForce();
      })
      .catch(() => {
        if (!cancelled) visionModelState = 'unavailable';
      });

    loadFusionModel()
      .then(model => {
        if (cancelled) return;
        fusionModel = model;
        fusionModelState = 'learned';
        refreshFusion();
        if (mode === 'fusion') controllerForce = activeForce();
      })
      .catch(() => {
        if (!cancelled) fusionModelState = 'unavailable';
      });

    let previous = performance.now(), accumulator = 0;
    const frame = now => {
      const delta = Math.min(.08, (now-previous)/1000);
      previous = now;

      if (running) {
        accumulator += delta;
        while (accumulator >= PHYSICS.tau) {
          result = inferState(history);
          refreshVision();
          controllerForce = activeForce();

          state = stepCartPole(state, controllerForce, PHYSICS.tau, disturbance);
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
          accumulator -= PHYSICS.tau;

          if (terminal(state)) {
            status = 'fell';
            running = false;
            disturbance = 0;
            break;
          }
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

<main data-observation-mode={mode}>
  <header class="topbar">
    <strong>Cart-Pole Transformer</strong>
    <div class="mode-switch" role="group" aria-label="observation mode">
      <button type="button" class:active={mode==='state'} on:click={()=>setMode('state')}>State</button>
      <button type="button" class:active={mode==='vision'} disabled={visionModelState!=='learned'} on:click={()=>setMode('vision')}>Vision</button>
      <button type="button" class:active={mode==='fusion'} disabled={fusionModelState!=='learned'} on:click={()=>setMode('fusion')}>Fusion</button>
    </div>
    <span>
      {mode==='state'
        ? (modelState === 'learned' ? 'learned state · 1 head · 8D' : modelState === 'loading' ? 'loading state model…' : 'transparent fallback')
        : mode==='vision'
          ? 'pixels only · 8 frames · 1 head · 24D'
          : 'typed fusion · 16 tokens · 1 head · 24D'}
    </span>
  </header>

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
      onToggle={toggle}
      onReset={reset}
      onPush={push}
      onPushEnd={pushEnd}
    />

    {#if mode === 'state'}
      <Pipeline
        {result}
        {controllerForce}
        {selectedToken}
        {selectedRow}
        {selectedCol}
        {expandedStage}
        onSelectToken={selectToken}
        onSelectAttention={selectAttention}
        onExpandedStageChange={(stage)=>expandedStage=stage}
      />
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
    <TransformerDetail
      {result}
      {controllerForce}
      {selectedToken}
      {selectedRow}
      {selectedCol}
      {expandedStage}
      onClose={()=>expandedStage=null}
      onSelectToken={selectToken}
      onSelectAttention={selectAttention}
    />
  {:else if mode === 'vision' && visionDetailOpen && visionResult}
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

  <section class="explain">
    <div class="explain-head">
      <h2>{mode==='state'?'State 읽는 순서':mode==='vision'?'Vision-only 읽는 순서':'State + Vision 읽는 순서'}</h2>

      {#if mode==='state'}
        <div class="qkv-key"><span class="q">Q</span> 찾는 기준 <span class="k">K</span> 비교 표지 <span class="v">V</span> 가져올 내용</div>
      {:else if mode==='vision'}
        <div class="vision-note">explicit simulator state is hidden from the controller</div>
      {:else}
        <div class="vision-note">same episode · aligned timestamps · typed-token self-attention</div>
      {/if}
    </div>

    {#if mode==='state'}
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
    {:else}
      <div class="steps">
        <article><b>1. Same timestamps</b><p>Vision과 같은 60 ms 샘플 시점에서 explicit state도 함께 snapshot합니다.</p></article>
        <article><b>2. Typed tokens</b><p>각 시점마다 State token 1개 + Vision token 1개. 총 <code>8×2=16</code> token입니다.</p></article>
        <article><b>3. Type + time</b><p>두 modality는 같은 time embedding을 공유하고 서로 다른 modality/type embedding으로 구분됩니다.</p></article>
        <article><b>4. One self-attention</b><p>별도 cross-attention 없이 16 token이 한 matrix에서 상호작용합니다. 같은 시점 S↔V는 허용됩니다.</p></article>
        <article><b>5. Degradation</b><p>Noisy/missing state, partial/missing vision을 같은 episode에서 즉시 바꿔 결과를 비교합니다.</p></article>
        <article><b>6. Negative result도 보존</b><p>단순 fusion이 항상 낫지는 않습니다. noisy-state에서는 vision 추가가 control을 악화시킨 결과도 그대로 보여줍니다.</p></article>
      </div>
    {/if}
  </section>

  <div class="claim">Cart-Pole에는 Transformer나 multimodal fusion이 필요하지 않습니다. 이 lab은 observation과 temporal/multimodal attention의 실제 계산과 한계를 비교하기 위한 교육용 모델입니다.</div>

  <footer>Polo Club Transformer Explainer의 MIT-licensed VectorCanvas, MatrixSvg, Sankey, attention expansion/animation 패턴을 vendoring·adaptation하여 사용합니다.</footer>
</main>
