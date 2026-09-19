<script>
  import { onMount } from 'svelte';
  import CartPoleView from './components/CartPoleView.svelte';
  import Pipeline from './components/Pipeline.svelte';
  import TransformerDetail from './components/TransformerDetail.svelte';
  import VisionPipeline from './components/VisionPipeline.svelte';
  import VisionDetail from './components/VisionDetail.svelte';
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

  let learnedModel = null;
  let modelState = 'loading';
  let visionModel = null;
  let visionModelState = 'loading';

  let result = runAttention(history);
  let initialVisionObservation = visionObservationFromState(stateArray(state));
  let visionBuffer = Array.from({length:VISION_BUFFER_LENGTH},()=>({
    frame:[...initialVisionObservation.frame],
    patches:[...initialVisionObservation.patches]
  }));
  let visionSamples = sampleVisionHistory(visionBuffer);
  let visionFrames = visionSamples.map(x=>x.frame);
  let visionPatchHistory = visionSamples.map(x=>x.patches);
  let visionResult = null;
  let visionRepeatedResult = null;
  $: visionFrameIntervalMs = (visionModel?.frame_stride_seconds || PHYSICS.tau * 3) * 1000;

  let controllerForce = forceFromScore(result.actionScore);
  let disturbance = 0;
  let running = true;
  let elapsed = 0;
  let status = 'balancing';
  let raf = 0;

  function inferState(sequence) {
    return learnedModel ? runLearnedAttention(sequence, learnedModel) : runAttention(sequence);
  }

  function refreshVision() {
    visionSamples = sampleVisionHistory(visionBuffer);
    visionFrames = visionSamples.map(x=>x.frame);
    visionPatchHistory = visionSamples.map(x=>x.patches);
    if (visionModel && visionPatchHistory.length === VISION_SEQUENCE_LENGTH) {
      visionResult = runVisionAttention(visionPatchHistory, visionModel);
      const latest = visionPatchHistory[visionPatchHistory.length - 1];
      const repeated = Array.from({length:VISION_SEQUENCE_LENGTH},()=>[...latest]);
      visionRepeatedResult = runVisionAttention(repeated, visionModel);
    }
  }

  function activeForce() {
    if (mode === 'vision' && visionResult) return forceFromScore(visionResult.actionScore);
    return forceFromScore(result.actionScore);
  }

  function setMode(next) {
    if (next === 'vision' && visionModelState !== 'learned') return;
    mode = next;
    expandedStage = null;
    visionDetailOpen = false;
    selectedToken = N - 1;
    selectedRow = N - 1;
    selectedCol = N - 1;
    selectedVisionFrame = VISION_SEQUENCE_LENGTH - 1;
    result = inferState(history);
    refreshVision();
    controllerForce = activeForce();
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
    history = Array.from({length:N}, () => stateArray(state));
    const observation = visionObservationFromState(stateArray(state));
    visionBuffer = Array.from({length:VISION_BUFFER_LENGTH},()=>({
      frame:[...observation.frame],
      patches:[...observation.patches]
    }));
    selectedToken = N - 1;
    selectedRow = N - 1;
    selectedCol = N - 1;
    selectedVisionFrame = VISION_SEQUENCE_LENGTH - 1;
    expandedStage = null;
    visionDetailOpen = false;
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
          visionBuffer = [...visionBuffer.slice(1), observation];

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
  <meta name="description" content="State and pixels-only Transformer learning modes for live Cart-Pole."/>
</svelte:head>

<main data-observation-mode={mode}>
  <header class="topbar">
    <strong>Cart-Pole Transformer</strong>
    <div class="mode-switch" role="group" aria-label="observation mode">
      <button type="button" class:active={mode==='state'} on:click={()=>setMode('state')}>State</button>
      <button type="button" class:active={mode==='vision'} disabled={visionModelState!=='learned'} on:click={()=>setMode('vision')}>Vision</button>
    </div>
    <span>
      {mode==='state'
        ? (modelState === 'learned' ? 'learned state · 1 head · 8D' : modelState === 'loading' ? 'loading state model…' : 'transparent fallback')
        : 'pixels only · 8 frames · 1 head · 24D'}
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
      showStateOverlay={mode==='state'}
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
    {:else if visionResult}
      <VisionPipeline
        frames={visionFrames}
        result={visionResult}
        {controllerForce}
        selectedFrame={selectedVisionFrame}
        onSelectFrame={(i)=>selectedVisionFrame=i}
        onOpenDetail={()=>visionDetailOpen=true}
        frameIntervalMs={visionFrameIntervalMs}
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
  {:else if visionDetailOpen && visionResult}
    <VisionDetail
      frames={visionFrames}
      result={visionResult}
      repeatedResult={visionRepeatedResult}
      selectedFrame={selectedVisionFrame}
      groundTruthState={stateArray(state)}
      onSelectFrame={(i)=>selectedVisionFrame=i}
      onClose={()=>visionDetailOpen=false}
      frameIntervalMs={visionFrameIntervalMs}
    />
  {/if}

  <section class="explain">
    <div class="explain-head">
      <h2>{mode==='state'?'읽는 순서':'Vision-only 읽는 순서'}</h2>
      {#if mode==='state'}
        <div class="qkv-key"><span class="q">Q</span> 찾는 기준 <span class="k">K</span> 비교 표지 <span class="v">V</span> 가져올 내용</div>
      {:else}
        <div class="vision-note">explicit simulator state is hidden from the controller</div>
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
    {:else}
      <div class="steps">
        <article><b>1. Pixels</b><p>Controller는 32×32 rendered frame만 받습니다. <code>x, ẋ, θ, θ̇</code> 숫자는 입력하지 않습니다.</p></article>
        <article><b>2. Patches</b><p>2×2 pixel 평균을 사용해 16×16 = 256개의 inspectable visual feature를 만듭니다.</p></article>
        <article><b>3. Frame token</b><p>256D patch feature를 learned Linear로 24D visual token으로 압축합니다.</p></article>
        <article><b>4. Temporal attention</b><p>현재 frame과 60 ms 간격의 과거 7장을 비교해 총 0.42 s의 움직임을 봅니다.</p></article>
        <article><b>5. Motion inference</b><p>마지막 hidden token이 pixels에서 state를 추정합니다. 특히 ẋ와 θ̇는 frame history가 핵심입니다.</p></article>
        <article><b>Ablation</b><p>상세 화면에서 8-frame 추정과 같은 최신 frame을 8번 반복한 결과를 직접 비교합니다.</p></article>
      </div>
    {/if}
  </section>

  <div class="claim">Cart-Pole에는 Transformer가 필요하지 않습니다. 이 lab은 state와 pixels라는 두 observation 경로에서 temporal attention의 실제 계산을 관찰하기 위한 교육용 모델입니다.</div>

  <footer>Polo Club Transformer Explainer의 MIT-licensed VectorCanvas, MatrixSvg, Sankey, attention expansion/animation 패턴을 vendoring·adaptation하여 사용합니다.</footer>
</main>
