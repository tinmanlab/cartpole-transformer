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
  <meta name="description" content="Live Cart-Pole simulation beside its Transformer attention computation."/>
</svelte:head>

<main>
  <header class="hero">
    <div class="kicker">CART-POLE TRANSFORMER · LIVE EXPLAINER</div>
    <h1>시뮬레이션과 Attention을 한 화면에서</h1>
    <p>왼쪽에서 움직이는 Cart-Pole의 같은 상태가 오른쪽에서 바로 token → Q/K/V → attention → force로 계산됩니다.</p>
  </header>

  <section class="lab-grid" aria-label="live Cart-Pole and Transformer visualization">
    <CartPoleView {state} force={appliedForce} {running} {elapsed} {status} onToggle={toggle} onReset={reset} onPush={push} onPushEnd={pushEnd}/>
    <Pipeline {history} {result} {controllerForce}/>
  </section>

  <section class="explain">
    <div class="explain-head">
      <div>
        <div class="kicker">READ AFTER WATCHING THE LIVE VIEW</div>
        <h2>오른쪽 그림은 무엇을 하는가?</h2>
      </div>
      <div class="qkv-key"><span class="q">Q</span> 찾을 기준 <span class="k">K</span> 비교용 꼬리표 <span class="v">V</span> 실제 가져올 정보</div>
    </div>
    <div class="steps">
      <article><b>1 · State token</b><p>각 시점의 <code>[x, ẋ, θ, θ̇]</code>가 token 하나입니다. 8개 token은 짧은 시간 기억입니다.</p></article>
      <article><b>2 · Q / K / V</b><p>같은 token을 세 방식으로 바꿉니다. Q와 K는 “어디를 볼지”, V는 “무엇을 가져올지” 담당합니다.</p></article>
      <article><b>3 · Attention</b><p><code>QKᵀ / √d → softmax</code>로 과거 token의 중요도를 만듭니다. 현재 결정은 matrix의 마지막 row입니다.</p></article>
      <article><b>4 · Action</b><p>attention 비율만큼 V를 합친 context가 force를 만들고, 그 force가 다시 왼쪽 물리 시뮬레이션에 들어갑니다.</p></article>
    </div>
    <div class="formula-line"><code>state history → Q, K, V → QKᵀ/√d → causal mask → softmax → Σ(attention·V) → force</code></div>
  </section>

  <div class="claim">현재는 계산을 완전히 보이게 만든 <b>1-head transparent attention controller</b>입니다. 아직 학습된 Transformer policy는 아니며, 다음 단계에서 이 동일한 시각화 인터페이스에 실제 learned Q/K/V tensor를 연결합니다.</div>

  <footer>Polo Club Transformer Explainer의 MIT-licensed 시각화 방법을 참고해 vector strips, D3 matrix, DOM flow path, GSAP path animation을 Cart-Pole용으로 재구성했습니다.</footer>
</main>
