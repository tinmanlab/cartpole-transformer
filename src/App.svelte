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
  <meta name="description" content="Live Cart-Pole simulation connected to an upstream-inspired Transformer attention visualization."/>
</svelte:head>

<main>
  <header class="hero">
    <div>
      <div class="kicker">CART-POLE TRANSFORMER · INTERACTIVE LAB</div>
      <h1>Attention이 실제로 cart를 움직이는 순간을 본다</h1>
      <p>왼쪽의 물리 시뮬레이션과 오른쪽의 Transformer 내부 계산은 같은 시간축입니다. 막대가 움직일 때마다 8개의 상태 token, Q/K/V, attention matrix, context와 force가 실시간으로 함께 바뀝니다.</p>
    </div>
    <div class="legend">
      <div><i class="q"></i><b>Q</b><span>무엇을 찾나?</span></div>
      <div><i class="k"></i><b>K</b><span>어떤 정보인가?</span></div>
      <div><i class="v"></i><b>V</b><span>무엇을 가져오나?</span></div>
    </div>
  </header>

  <section class="top-grid">
    <CartPoleView {state} force={appliedForce} {running} {elapsed} {status} onToggle={toggle} onReset={reset} onPush={push} onPushEnd={pushEnd}/>
    <aside class="now">
      <div class="eyebrow">RIGHT NOW</div>
      <h2>현재 Query는 어디를 보고 있나?</h2>
      <div class="attn-bars">
        {#each result.weights[result.weights.length-1] as weight, i}
          <div class="bar-row">
            <span>{i===N-1?'t':'t−'+(N-1-i)}</span>
            <div class="bar"><i style={"width:"+(weight*100)+"%"}></i></div>
            <b>{(weight*100).toFixed(0)}%</b>
          </div>
        {/each}
      </div>
      <div class="force-readout">
        <span>attention → context → controller</span>
        <strong>{controllerForce >= 0 ? 'RIGHT' : 'LEFT'} {Math.abs(controllerForce).toFixed(1)} N</strong>
      </div>
      <p class="small">← / → push 버튼을 누르고 있으면 외란이 추가됩니다. controller는 중단되지 않고 같은 50 Hz로 계속 계산합니다.</p>
    </aside>
  </section>

  <Pipeline {history} {result} {controllerForce}/>

  <section class="concepts">
    <article><span>1</span><h3>Token</h3><p>텍스트의 단어 대신 <code>[x, ẋ, θ, θ̇]</code>라는 한 시점의 상태가 token입니다. 8개 token은 약 0.16초의 짧은 기억입니다.</p></article>
    <article><span>2</span><h3>Q / K / V</h3><p>같은 상태를 세 방식으로 투영합니다. Q/K는 “누구를 볼지”, V는 “무슨 정보를 가져올지”를 분리합니다.</p></article>
    <article><span>3</span><h3>Attention</h3><p>Q·K 점수를 softmax로 바꿔 가중치를 만들고, 그만큼 V를 섞습니다. matrix에서 현재 row가 실제 이번 제어 결정입니다.</p></article>
    <article><span>4</span><h3>Action</h3><p>섞인 context가 force를 만들고 그 force가 실제 물리식으로 다음 Cart-Pole 상태를 만듭니다. 그래서 계산과 움직임이 끊어지지 않습니다.</p></article>
  </section>

  <div class="claim">현재 모델은 <b>학습된 Transformer가 아니라</b> attention 계산을 숨김없이 보여주기 위한 1-head transparent controller입니다. 다음 단계에서 동일한 시각화 인터페이스에 실제 학습된 tiny causal Transformer의 intermediate tensor를 연결합니다.</div>

  <footer>Visualization methods adapted from the interaction patterns of Polo Club Transformer Explainer (MIT): canvas vector strips, D3 matrix encoding, DOM-to-DOM flow paths, and expandable attention animation.</footer>
</main>
