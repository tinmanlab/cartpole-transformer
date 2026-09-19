<script>
  import { onMount, tick } from 'svelte';
  import { gsap } from 'gsap';
  import VectorStrip from './VectorStrip.svelte';
  import AttentionMatrix from './AttentionMatrix.svelte';

  export let history = [];
  export let result;
  export let controllerForce = 0;

  let root, stateCol, qkvCol, attentionCol, outCol, actionCol;
  let paths = [];
  let selectedRow = 0;
  let selectedCol = 0;
  let expanded = false;
  let ro;

  function curve(a, b) {
    const dx = Math.max(45, (b.x - a.x) * .42);
    return 'M '+a.x+' '+a.y+' C '+(a.x+dx)+' '+a.y+' '+(b.x-dx)+' '+b.y+' '+b.x+' '+b.y;
  }
  function point(el, side) {
    const r = el.getBoundingClientRect(), rr = root.getBoundingClientRect();
    return { x:(side==='right'?r.right:r.left)-rr.left+root.scrollLeft, y:r.top-rr.top+r.height/2 };
  }
  async function updatePaths() {
    await tick();
    if (!root || !stateCol || !qkvCol || !attentionCol || !outCol || !actionCol) return;
    paths = [
      { d:curve(point(stateCol,'right'), point(qkvCol,'left')), cls:'state-qkv' },
      { d:curve(point(qkvCol,'right'), point(attentionCol,'left')), cls:'qkv-attn' },
      { d:curve(point(attentionCol,'right'), point(outCol,'left')), cls:'attn-out' },
      { d:curve(point(outCol,'right'), point(actionCol,'left')), cls:'out-action' }
    ];
  }
  function selectCell(r,c){ selectedRow=r; selectedCol=c; }
  function toggleExpand() {
    expanded = !expanded;
    requestAnimationFrame(() => {
      root?.querySelectorAll('.flow-path').forEach(path => {
        const length = path.getTotalLength();
        gsap.fromTo(path,{strokeDasharray:length,strokeDashoffset:length},{strokeDashoffset:0,duration:.65,ease:'power2.out'});
      });
    });
  }

  onMount(() => {
    selectedRow = Math.max(0, history.length-1);
    selectedCol = Math.max(0, history.length-1);
    ro = new ResizeObserver(updatePaths);
    ro.observe(root);
    window.addEventListener('resize', updatePaths);
    updatePaths();
    return () => { ro?.disconnect(); window.removeEventListener('resize', updatePaths); };
  });

  $: history, result, root && updatePaths();
</script>

<section class="pipeline-shell">
  <div class="section-head">
    <div>
      <div class="eyebrow">UPSTREAM-STYLE LIVE DATAFLOW</div>
      <h2>한 번의 제어 결정이 지나가는 길</h2>
    </div>
    <button class:active={expanded} on:click={toggleExpand}>{expanded ? '계산 접기' : 'Attention 계산 펼치기'}</button>
  </div>

  <div class="scroll">
    <div class:expanded class="pipeline" bind:this={root}>
      <svg class="flow" aria-hidden="true">
        <defs>
          <linearGradient id="gState" x1="0" x2="1"><stop stop-color="#9ca3af"/><stop offset="1" stop-color="#6f83dd"/></linearGradient>
          <linearGradient id="gQkv" x1="0" x2="1"><stop stop-color="#5d7df5"/><stop offset=".45" stop-color="#d76565"/><stop offset="1" stop-color="#48a779"/></linearGradient>
          <linearGradient id="gAttn" x1="0" x2="1"><stop stop-color="#8a6bd5"/><stop offset="1" stop-color="#a477d8"/></linearGradient>
        </defs>
        {#each paths as path}
          <path class={"flow-path "+path.cls} d={path.d}/>
        {/each}
      </svg>

      <div class="column state-col" bind:this={stateCol}>
        <div class="col-title">State tokens</div>
        <div class="token-stack">
          {#each result.tokens as token, i}
            <div class:current={i===history.length-1} class="token-row">
              <span>{i===history.length-1 ? 't' : 't−'+(history.length-1-i)}</span>
              <VectorStrip data={token} kind="state" active={selectedCol===i} />
            </div>
          {/each}
        </div>
        <small>[x, ẋ, θ, θ̇]</small>
      </div>

      <div class="column qkv-col" bind:this={qkvCol}>
        <div class="col-title">Q · K · V projection</div>
        <div class="qkv-legend"><b class="q">Q</b><b class="k">K</b><b class="v">V</b></div>
        <div class="qkv-stack">
          {#each result.q as q, i}
            <div class:current={i===history.length-1} class="qkv-row">
              <VectorStrip data={q} kind="q" active={selectedRow===i}/>
              <VectorStrip data={result.k[i]} kind="k" active={selectedCol===i}/>
              <VectorStrip data={result.v[i]} kind="v" active={selectedCol===i}/>
            </div>
          {/each}
        </div>
        <small>같은 token → 서로 다른 역할</small>
      </div>

      <div class="column attention-col" bind:this={attentionCol} on:click={toggleExpand} on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && toggleExpand()} role="button" tabindex="0">
        <div class="col-title">Self Attention</div>
        <div class="matrices">
          <div><b>Q · Kᵀ / √d</b><AttentionMatrix data={result.raw} mode="raw" {selectedRow} {selectedCol} onSelect={selectCell}/></div>
          <div><b>Softmax</b><AttentionMatrix data={result.weights} mode="weights" {selectedRow} {selectedCol} onSelect={selectCell}/></div>
        </div>
        <small>hover: row=Query, column=Key</small>
      </div>

      <div class="column out-col" bind:this={outCol}>
        <div class="col-title">Attention out</div>
        <div class="out-vector"><VectorStrip data={result.context} kind="out" active={true} label="ΣV"/></div>
        <div class="numbers">{result.context.map(v=>v.toFixed(2)).join(' · ')}</div>
        <small>softmax 비율만큼 V를 합침</small>
      </div>

      <div class="column action-col" bind:this={actionCol}>
        <div class="col-title">Action head</div>
        <div class="action-arrow">{controllerForce >= 0 ? '→' : '←'}</div>
        <strong>{Math.abs(controllerForce).toFixed(1)} N</strong>
        <small>실제로 simulator에 입력</small>
      </div>

      {#if expanded}
        <div class="explain-panel">
          <div><span>1</span><b>Query</b><p>현재 token이 “지금 어떤 과거 상태가 유용한가?”라고 묻습니다.</p></div>
          <div><span>2</span><b>Q·Kᵀ</b><p>각 Query와 과거 Key를 비교합니다. 미래 token은 causal mask로 볼 수 없습니다.</p></div>
          <div><span>3</span><b>Softmax</b><p>비교 점수를 합이 1인 attention 비율로 바꿉니다.</p></div>
          <div><span>4</span><b>Σ attention·V</b><p>그 비율만큼 실제 정보 Value를 섞고, 그 결과가 cart force를 만듭니다.</p></div>
        </div>
      {/if}
    </div>
  </div>
</section>

<style>
.pipeline-shell{background:#fff;border:1px solid #e2e5ea;border-radius:18px;padding:16px;overflow:hidden}.section-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start}.eyebrow{font-size:11px;letter-spacing:.08em;color:#8a93a2}.section-head h2{margin:3px 0 0;font-size:20px}.section-head button{border:1px solid #d9dde5;background:#fff;border-radius:9px;padding:8px 11px;cursor:pointer}.section-head button.active{background:#f4f0ff;border-color:#b5a5df}
.scroll{overflow-x:auto;margin-top:14px;padding-bottom:5px}.pipeline{position:relative;min-width:1080px;display:grid;grid-template-columns:150px 190px 430px 135px 120px;gap:30px;align-items:center;padding:34px 18px 30px}.pipeline.expanded{padding-bottom:178px}
.flow{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:0}.flow-path{fill:none;stroke-width:10;opacity:.24}.state-qkv{stroke:url(#gState)}.qkv-attn{stroke:url(#gQkv)}.attn-out,.out-action{stroke:url(#gAttn)}
.column{position:relative;z-index:2;min-height:330px;display:flex;flex-direction:column;align-items:center;justify-content:center}.col-title{font-size:13px;font-weight:700;color:#657083;margin-bottom:10px;white-space:nowrap}.column small{font-size:10px;color:#8a93a2;margin-top:8px;text-align:center}
.token-stack,.qkv-stack{display:flex;flex-direction:column;gap:4px}.token-row{display:flex;align-items:center;gap:7px;padding:2px 5px;border-radius:7px}.token-row.current,.qkv-row.current{background:#f6f4ff}.token-row span{width:29px;text-align:right;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#778190}
.qkv-row{display:grid;grid-template-columns:repeat(3,26px);gap:5px;padding:2px 5px;border-radius:7px}.qkv-legend{display:grid;grid-template-columns:repeat(3,26px);gap:5px;text-align:center;font-size:11px}.q{color:#4f74ff}.k{color:#df5f5f}.v{color:#3d9d70}
.matrices{display:flex;gap:12px;align-items:flex-start}.matrices>div{background:#fbfbfd;border:1px solid #eceef2;border-radius:12px;padding:8px}.matrices b{display:block;text-align:center;font-size:11px;color:#697386;margin-bottom:3px}
.attention-col{cursor:pointer}.out-vector{height:96px}.numbers{margin-top:10px;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6d7583;text-align:center}.action-arrow{font-size:58px;line-height:1;color:#8b66d8}.action-col strong{font-size:18px;margin-top:8px}
.explain-panel{position:absolute;z-index:4;left:230px;right:160px;bottom:16px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;background:white;border:1px solid #ded8ef;border-radius:15px;padding:12px;box-shadow:0 12px 30px rgba(35,39,55,.08)}.explain-panel div{position:relative;padding-left:28px}.explain-panel span{position:absolute;left:0;top:0;width:21px;height:21px;border-radius:50%;background:#eee9fb;color:#7357b7;text-align:center;font-size:11px;line-height:21px;font-weight:700}.explain-panel b{font-size:12px}.explain-panel p{font-size:11px;color:#697386;margin:3px 0 0;line-height:1.45}
</style>
