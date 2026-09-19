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
  let learnedModel = null;
  let modelState = 'loading';
  let result = runAttention(history);
  let controllerForce = forceFromScore(result.actionScore);
  let disturbance = 0;
  let appliedForce = controllerForce;
  let running = true;
  let elapsed = 0;
  let status = 'balancing';
  let raf = 0;
  const bridgePathMap = {
    state: [{
      from: '.sim-card .state-readout',
      to: '.stage-raw .anchor',
      type: 'stroke',
      gradientId: 'gray-blue',
      opacity: .95,
      curve: 80
    }]
  };
  $: bridgeRedrawKey = stateArray(state).map(v=>v.toFixed(4)).join('|') + result.modelType;

  function infer(sequence) {
    return learnedModel ? runLearnedAttention(sequence, learnedModel) : runAttention(sequence);
  }

  function reset() {
    state = resetState((Math.random()-.5)*.09);
    history = Array.from({length:N}, () => stateArray(state));
    result = infer(history);
    controllerForce = forceFromScore(result.actionScore);
    disturbance = 0;
    appliedForce = controllerForce;
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
          appliedForce = controllerForce + disturbance;
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
  <meta name="description" content="Live Cart-Pole simulation beside a step-by-step Transformer attention dataflow."/>
</svelte:head>

<main>
  <header class="topbar">
    <strong>Cart-Pole Transformer</strong>
    <span>{modelState === 'learned' ? 'learned 1-head tiny Transformer' : modelState === 'loading' ? 'loading learned model…' : 'transparent fallback'}</span>
  </header>

  <section class="lab-grid" aria-label="live Cart-Pole and Transformer visualization">
    <UpstreamSankeyFlow pathMap={bridgePathMap} redrawKey={bridgeRedrawKey}/>
    <CartPoleView {state} force={appliedForce} {running} {elapsed} {status} onToggle={toggle} onReset={reset} onPush={push} onPushEnd={pushEnd}/>
    <Pipeline {history} {result} {controllerForce}/>
  </section>

  <section class="explain">
    <div class="explain-head">
      <h2>이 순서만 보면 됩니다</h2>
      <div class="qkv-key"><span class="q">Q</span> 찾는 기준 <span class="k">K</span> 비교용 표지 <span class="v">V</span> 가져올 내용</div>
    </div>
    <div class="steps">
      <article><b>1. Raw state</b><p>시뮬레이터가 <code>[x, ẋ, θ, θ̇]</code> 네 숫자를 냅니다.</p></article>
      <article><b>2. Encode</b><p>먼저 고정 scale로 normalize합니다. learned 모델에서는 그 4개 숫자를 <code>Linear 4→8 + position</code>으로 8차원 token으로 바꿉니다.</p></article>
      <article><b>3. Q / K / V</b><p>encoded token에 실제 학습된 <code>WQ, WK, WV</code>를 곱합니다.</p></article>
      <article><b>4. QK → mask → softmax</b><p>Polo Club 원본처럼 score, causal mask, attention weight가 순서대로 보입니다.</p></article>
      <article><b>5. Weighted V</b><p>attention 비율만큼 V를 섞어 context를 만듭니다.</p></article>
      <article><b>6. Force</b><p>context를 action head가 읽어 cart force를 만들고 다시 physics에 넣습니다.</p></article>
    </div>
    <div class="formula-line"><code>raw state → normalize → learned embedding + position → Q/K/V → QKᵀ/√d → causal mask → softmax → Σ(a·V) → force</code></div>
  </section>

  <div class="claim">Cart-Pole에는 Transformer가 필요하지 않습니다. learned mode는 sequence attention을 관찰하기 위한 교육용 비교 모델이며 attention weight를 정책 행동의 인과 설명으로 해석하지 않습니다.</div>

  <footer>Polo Club Transformer Explainer의 MIT-licensed VectorCanvas, MatrixSvg, Sankey 경로 방식을 vendoring·adaptation하여 사용합니다.</footer>
</main>
