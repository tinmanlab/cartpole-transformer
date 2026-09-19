<script>
  import { onMount } from 'svelte';
  import CartPoleView from './components/CartPoleView.svelte';
  import Pipeline from './components/Pipeline.svelte';
  import { resetState, stepCartPole, terminal, stateArray, PHYSICS } from './lib/physics.js';
  import { runAttention, forceFromScore } from './lib/attention.js';

  const N = 8;
  let state = resetState();
  let history = Array.from({length:N}, () => stateArray(state));
  let result = runAttention(history);
  let controllerForce = forceFromScore(result.actionScore);
  let disturbance = 0;
  let appliedForce = controllerForce;
  let running = true;
  let elapsed = 0;
  let status = 'balancing';
  let raf = 0;

  function reset() {
    state = resetState((Math.random()-.5)*.09);
    history = Array.from({length:N}, () => stateArray(state));
    result = runAttention(history);
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
    let previous = performance.now(), accumulator = 0;
    const frame = now => {
      const delta = Math.min(.08, (now-previous)/1000);
      previous = now;
      if (running) {
        accumulator += delta;
        while (accumulator >= PHYSICS.tau) {
          result = runAttention(history);
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
    return () => cancelAnimationFrame(raf);
  });
</script>

<svelte:head>
  <title>Cart-Pole Transformer Explainer</title>
  <meta name="description" content="Live Cart-Pole simulation beside a step-by-step Transformer attention dataflow."/>
</svelte:head>

<main>
  <header class="topbar">
    <strong>Cart-Pole Transformer</strong>
    <span>live 50 Hz · transparent 1-head attention</span>
  </header>

  <section class="lab-grid" aria-label="live Cart-Pole and Transformer visualization">
    <CartPoleView {state} force={appliedForce} {running} {elapsed} {status} onToggle={toggle} onReset={reset} onPush={push} onPushEnd={pushEnd}/>
    <Pipeline {history} {result} {controllerForce}/>
  </section>

  <section class="explain">
    <div class="explain-head">
      <h2>아래 순서만 보면 됩니다</h2>
      <div class="qkv-key"><span class="q">Q</span> 찾는 기준 <span class="k">K</span> 비교용 표지 <span class="v">V</span> 가져올 내용</div>
    </div>
    <div class="steps">
      <article><b>1. Raw state</b><p>시뮬레이터가 <code>[x, ẋ, θ, θ̇]</code> 네 숫자를 냅니다.</p></article>
      <article><b>2. Normalize / encode</b><p>단위와 크기가 다른 네 숫자를 고정 scale로 나눠 비슷한 범위의 state token으로 만듭니다. <b>현재 버전에는 별도 learned embedding이 없습니다.</b></p></article>
      <article><b>3. Q / K / V</b><p>각 token에 <code>WQ, WK, WV</code>를 곱해 검색 기준 Q, 비교용 K, 실제 내용 V를 만듭니다.</p></article>
      <article><b>4. Compare</b><p><code>QKᵀ / √d</code>로 모든 시간 token을 비교하고 미래 칸은 causal mask로 가립니다.</p></article>
      <article><b>5. Softmax + V</b><p>비교 점수를 확률처럼 합이 1인 비율로 바꾸고 그 비율만큼 V를 섞어 context를 만듭니다.</p></article>
      <article><b>6. Force</b><p>context를 작은 action head가 읽어 실제 cart force를 만들고 다시 왼쪽 simulation에 넣습니다.</p></article>
    </div>
    <div class="formula-line"><code>raw state → normalize → token → WQ/WK/WV → QKᵀ/√d + mask → softmax → Σ(a·V) → force → physics</code></div>
  </section>

  <div class="claim">현재 controller는 학습된 Transformer가 아니라 계산을 투명하게 보기 위한 hand-authored attention controller입니다. 시각화 구조는 그대로 두고 다음 단계에서 learned tensor로 교체합니다.</div>

  <footer>Polo Club Transformer Explainer의 MIT-licensed vector/matrix visualization 방식을 Cart-Pole에 맞게 재구성했습니다.</footer>
</main>
