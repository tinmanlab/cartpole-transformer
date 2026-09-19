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
  let ro;
  let lastDirection = null;

  function curve(a, b) {
    const dx = Math.max(24, (b.x - a.x) * .36);
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
  function animateFlow(){
    requestAnimationFrame(() => {
      root?.querySelectorAll('.flow-path').forEach((path,i) => {
        const length = path.getTotalLength();
        gsap.fromTo(path,{strokeDasharray:length,strokeDashoffset:length,opacity:.08},{strokeDashoffset:0,opacity:.25,duration:.42,delay:i*.035,ease:'power2.out'});
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
    animateFlow();
    return () => { ro?.disconnect(); window.removeEventListener('resize', updatePaths); };
  });

  $: history, result, root && updatePaths();
  $: direction = controllerForce >= 0 ? 1 : -1;
  $: if (root && direction !== lastDirection) { lastDirection = direction; animateFlow(); }
</script>

<section class="pipeline-shell">
  <div class="pipeline-head">
    <div><span>TRANSFORMER · LIVE</span><strong>State → Attention → Force</strong></div>
    <div class="force"><b>{controllerForce >= 0 ? '→' : '←'}</b>{Math.abs(controllerForce).toFixed(1)} N</div>
  </div>

  <div class="pipeline" bind:this={root}>
    <svg class="flow" aria-hidden="true">
      <defs>
        <linearGradient id="gState" x1="0" x2="1"><stop stop-color="#9ca3af"/><stop offset="1" stop-color="#6f83dd"/></linearGradient>
        <linearGradient id="gQkv" x1="0" x2="1"><stop stop-color="#5d7df5"/><stop offset=".45" stop-color="#d76565"/><stop offset="1" stop-color="#48a779"/></linearGradient>
        <linearGradient id="gAttn" x1="0" x2="1"><stop stop-color="#8a6bd5"/><stop offset="1" stop-color="#a477d8"/></linearGradient>
      </defs>
      {#each paths as path}<path class={"flow-path "+path.cls} d={path.d}/>{/each}
    </svg>

    <div class="column state-col" bind:this={stateCol}>
      <div class="col-title">State</div>
      <div class="token-stack">
        {#each result.tokens as token, i}
          <div class:current={i===history.length-1} class="token-row">
            <span>{i===history.length-1 ? 't' : '−'+(history.length-1-i)}</span>
            <VectorStrip data={token} kind="state" active={selectedCol===i} height={25} width={17}/>
          </div>
        {/each}
      </div>
    </div>

    <div class="column qkv-col" bind:this={qkvCol}>
      <div class="col-title"><i class="q">Q</i> <i class="k">K</i> <i class="v">V</i></div>
      <div class="qkv-stack">
        {#each result.q as q, i}
          <div class:current={i===history.length-1} class="qkv-row">
            <VectorStrip data={q} kind="q" active={selectedRow===i} height={25} width={17}/>
            <VectorStrip data={result.k[i]} kind="k" active={selectedCol===i} height={25} width={17}/>
            <VectorStrip data={result.v[i]} kind="v" active={selectedCol===i} height={25} width={17}/>
          </div>
        {/each}
      </div>
    </div>

    <div class="column attention-col" bind:this={attentionCol}>
      <div class="col-title">Attention</div>
      <div class="matrices">
        <div><b>QKᵀ / √d</b><AttentionMatrix data={result.raw} mode="raw" {selectedRow} {selectedCol} onSelect={selectCell} cellSize={11} gap={1.5} pad={19}/></div>
        <div><b>Softmax</b><AttentionMatrix data={result.weights} mode="weights" {selectedRow} {selectedCol} onSelect={selectCell} cellSize={11} gap={1.5} pad={19}/></div>
      </div>
      <small>hover: Query row · Key column</small>
    </div>

    <div class="column out-col" bind:this={outCol}>
      <div class="col-title">ΣV</div>
      <VectorStrip data={result.context} kind="out" active={true} height={74} width={22}/>
      <small>{result.context.map(v=>v.toFixed(1)).join(' ')}</small>
    </div>

    <div class="column action-col" bind:this={actionCol}>
      <div class="col-title">Force</div>
      <div class="action-arrow">{controllerForce >= 0 ? '→' : '←'}</div>
      <strong>{Math.abs(controllerForce).toFixed(1)}</strong>
    </div>
  </div>
</section>

<style>
.pipeline-shell{height:100%;min-height:430px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:13px;overflow:hidden;display:flex;flex-direction:column}
.pipeline-head{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:0 2px 10px;border-bottom:1px solid #eef0f3}.pipeline-head span{display:block;font-size:9px;letter-spacing:.09em;color:#8a93a2}.pipeline-head strong{display:block;font-size:14px;margin-top:1px}.force{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6f5bab;white-space:nowrap}.force b{font-size:22px;margin-right:5px;vertical-align:-2px}
.pipeline{position:relative;flex:1;display:grid;grid-template-columns:62px 78px minmax(210px,1fr) 48px 48px;gap:11px;align-items:center;padding:18px 4px 8px;min-height:350px}
.flow{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:0}.flow-path{fill:none;stroke-width:7;opacity:.22}.state-qkv{stroke:url(#gState)}.qkv-attn{stroke:url(#gQkv)}.attn-out,.out-action{stroke:url(#gAttn)}
.column{position:relative;z-index:2;display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:0}.col-title{font-size:10px;font-weight:700;color:#667085;margin-bottom:7px;white-space:nowrap}.col-title i{font-style:normal}.q{color:#4f74ff}.k{color:#df5f5f}.v{color:#3d9d70}.column small{font-size:8px;color:#8a93a2;margin-top:6px;text-align:center;line-height:1.25}
.token-stack,.qkv-stack{display:flex;flex-direction:column;gap:3px}.token-row{display:flex;align-items:center;gap:4px;padding:1px 3px;border-radius:5px}.token-row span{width:17px;text-align:right;font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#778190}.token-row.current,.qkv-row.current{background:#f5f2fb}.qkv-row{display:grid;grid-template-columns:repeat(3,17px);gap:3px;padding:1px 3px;border-radius:5px}
.matrices{display:flex;justify-content:center;gap:7px;min-width:0}.matrices>div{flex:0 1 128px;min-width:0;background:#fbfbfd;border:1px solid #eceef2;border-radius:9px;padding:5px}.matrices b{display:block;text-align:center;font-size:8px;color:#697386;margin-bottom:2px}.attention-col small{margin-top:5px}
.action-arrow{font-size:40px;line-height:.9;color:#8b66d8}.action-col strong{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:6px;color:#5f4ba1}
@media(max-width:1080px){.pipeline{grid-template-columns:58px 72px minmax(200px,1fr) 44px 44px;gap:7px}.matrices{gap:4px}}
@media(max-width:760px){.pipeline-shell{min-height:410px}.pipeline{min-width:610px}.pipeline-shell{overflow-x:auto}}
</style>
