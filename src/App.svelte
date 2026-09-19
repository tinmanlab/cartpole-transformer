<script>
  import { onMount } from 'svelte';
  import CartPoleView from './components/CartPoleView.svelte';
  import Pipeline from './components/Pipeline.svelte';
  import UpstreamSankeyFlow from './upstream/SankeyFlow.svelte';
  import { resetState, stepCartPole, terminal, stateArray, PHYSICS } from './lib/physics.js';
  import { runAttention, forceFromScore } from './lib/attention.js';
  import { loadLearnedModel, runLearnedAttention } from './lib/learned_attention.js';

  const N = 8;
  let state = resetState();
  let history = Array.from({length:N}, () => stateArray(state));
  let selectedToken = N - 1;
  let learnedModel = null;
  let modelState = 'loading';
  let result = runAttention(history);
  let controllerForce = forceFromScore(result.actionScore);
  let disturbance = 0;
  let running = true;
  let elapsed = 0;
  let status = 'balancing';
  let raf = 0;

  const bridgePathMap = {
    state: [{
      from: '.sim-card',
      to: '.embedding-overview',
      type: 'stroke',
      gradientId: 'gray-blue',
      opacity: .55,
      curve: 28,
      strokeWidth: 1.8
    }]
  };
  $: bridgeRedrawKey = modelState + '|' + selectedToken;

  function infer(sequence) {
    return learnedModel ? runLearnedAttention(sequence, learnedModel) : runAttention(sequence);
  }

  function reset() {
    state = resetState((Math.random()-.5)*.09);
    history = Array.from({length:N}, () => stateArray(state));
    selectedToken = N - 1;
    result = infer(history);
    controllerForce = forceFromScore(result.actionScore);
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
        result = infer(history);
        controllerForce = forceFromScore(result.actionScore);
      })
      .catch(() => {
        if (!cancelled) modelState = 'toy-fallback';
      });

    let previous = performance.now(), accumulator = 0;
    const frame = now => {
      const delta = Math.min(.08, (now-previous)/1000);
      previous = now;
      if (running) {
        accumulator += delta;
        while (accumulator >= PHYSICS.tau) {
          result = infer(history);
          controllerForce = forceFromScore(result.actionScore);
          state = stepCartPole(state, controllerForce, PHYSICS.tau, disturbance);
          history = [...history.slice(1), stateArray(state)];
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
  <meta name="description" content="Live Cart-Pole simulation beside a detailed interactive Transformer walkthrough."/>
</svelte:head>

<main>
  <header class="topbar">
    <strong>Cart-Pole Transformer</strong>
    <span>{modelState === 'learned' ? 'learned · 1 block · 1 head · 8D' : modelState === 'loading' ? 'loading learned model…' : 'transparent fallback'}</span>
  </header>

  <section class="lab-grid" aria-label="live Cart-Pole and Transformer visualization">
    <UpstreamSankeyFlow pathMap={bridgePathMap} redrawKey={bridgeRedrawKey}/>
    <CartPoleView
      {state}
      {history}
      {selectedToken}
      {controllerForce}
      {disturbance}
      {running}
      {elapsed}
      {status}
      onToggle={toggle}
      onReset={reset}
      onPush={push}
      onPushEnd={pushEnd}
    />
    <Pipeline
      {history}
      {result}
      {controllerForce}
      {selectedToken}
      onSelectToken={(i)=>selectedToken=i}
    />
  </section>

  <section class="explain">
    <div class="explain-head">
      <h2>읽는 순서</h2>
      <div class="qkv-key"><span class="q">Q</span> 찾는 기준 <span class="k">K</span> 비교 표지 <span class="v">V</span> 가져올 내용</div>
    </div>
    <div class="steps">
      <article><b>1. Embedding</b><p><code>[x,ẋ,θ,θ̇]</code> → normalize → learned 4→8 projection + position.</p></article>
      <article><b>2. Q / K / V</b><p>LayerNorm 뒤 같은 token을 세 learned projection으로 나눕니다.</p></article>
      <article><b>3. Attention</b><p><code>QKᵀ/√d → causal mask → softmax</code>. matrix를 가리키면 해당 time pair가 연결됩니다.</p></article>
      <article><b>4. Residual + MLP</b><p>Attention output을 더하고 LN→Linear→GELU→Linear→residual을 통과합니다.</p></article>
      <article><b>5. Action</b><p>마지막 hidden token만 읽어 <code>tanh(score)×10 N</code> force를 만듭니다.</p></article>
      <article><b>실시간 연결</b><p>오른쪽 time token을 가리키면 왼쪽의 같은 과거 pose가 강조됩니다.</p></article>
    </div>
  </section>

  <div class="claim">Cart-Pole에는 Transformer가 필요하지 않습니다. 이 모델은 sequence attention의 실제 learned tensor를 관찰하기 위한 교육용 모델이며 attention weight를 인과 설명으로 해석하지 않습니다.</div>

  <footer>Polo Club Transformer Explainer의 MIT-licensed VectorCanvas, MatrixSvg, Sankey, attention expansion/animation 패턴을 vendoring·adaptation하여 사용합니다.</footer>
</main>
