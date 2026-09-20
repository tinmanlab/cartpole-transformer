<script>
  import * as d3 from 'd3';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  import UpstreamSankeyFlow from '../upstream/SankeyFlow.svelte';

  export let result;
  export let controllerForce = 0;
  export let scenario = 'clean';
  export let onScenarioChange = () => {};
  export let selectedToken = 15;
  export let onSelectToken = () => {};
  export let onOpenDetail = () => {};
  export let frameIntervalMs = 60;

  $: n = result.tokenMeta.length;
  $: selectedToken = Math.min(Math.max(0, selectedToken), Math.max(0,n-1));
  $: selectedMeta = result.tokenMeta[selectedToken];
  $: selectedVector = result.tokens[selectedToken] || [];
  $: latestState = result.finalStateIndex;
  $: latestVision = result.finalVisionIndex;
  $: weightColor = d3.scaleSequential(d3.interpolatePurples)
    .domain([0,Math.max(.001,d3.max(result.weights.flat()) || 1)]);
  $: redrawKey=[selectedToken,scenario,result.modelType].join('|');

  const pathMap = {
    obsToToken:[{
      from:'.fusion-stage-observation .pair.selected .pair-anchor',
      to:'.fusion-stage-token .token.selected .token-anchor',
      type:'stroke',gradientId:'gray-blue',opacity:.8,curve:28,strokeWidth:1.4
    }],
    tokenToAttention:[{
      from:'.fusion-stage-token .token.selected .token-anchor',
      to:'.fusion-stage-attention .attention-anchor',
      type:'stroke',gradientId:'blue-purple',opacity:.8,curve:30,strokeWidth:1.4
    }],
    attentionToState:[{
      from:'.fusion-stage-attention .attention-anchor',
      to:'.fusion-stage-output .state-anchor',
      type:'stroke',gradientId:'purple-purple',opacity:.8,curve:30,strokeWidth:1.4
    }]
  };

  const scenarios = [
    ['clean','Clean'],
    ['noisy-state','Noisy state'],
    ['missing-state','State missing'],
    ['partial-vision','Partial vision'],
    ['missing-vision','Vision missing']
  ];

  function timeLabel(t) {
    const last = 7;
    return t===last ? 't' : '−'+Math.round((last-t)*frameIntervalMs)+'ms';
  }
</script>

<section class="fusion-pipeline resize-watch">
  <div class="panel-head">
    <div><strong>State + Vision · typed-token fusion</strong><small>8 timestamps × 2 modalities = 16 tokens · no cross-attention block</small></div>
    <span>{controllerForce>=0?'→':'←'} {Math.abs(controllerForce).toFixed(2)} N</span>
  </div>

  <div class="scenario-bar">
    {#each scenarios as s}
      <button type="button" class:active={scenario===s[0]} on:click={()=>onScenarioChange(s[0])}>{s[1]}</button>
    {/each}
  </div>

  <div class="fusion-overview resize-watch">
    <UpstreamSankeyFlow {pathMap} {redrawKey}/>

    <button class="stage fusion-stage-observation" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>1</b><span>Aligned observations</span><small>same environment timestamps</small></div>
      <div class="pair-stack">
        {#each Array(8) as _,t}
          <div class:selected={selectedMeta?.time===t} class="pair" on:mouseenter={()=>onSelectToken(t*2+1)}>
            <span class="time">{timeLabel(t)}</span>
            <div class="pair-anchor pair-values">
              <span class:off={!result.stateAvailable} class="state-chip">S</span>
              <span class:off={!result.visionAvailable} class:partial={result.partialVision} class="vision-chip">V</span>
            </div>
          </div>
        {/each}
      </div>
    </button>

    <button class="stage fusion-stage-token" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>2</b><span>Typed tokens</span><small>state / vision + time + type</small></div>
      <div class="token-stack">
        {#each result.tokens as token,i}
          <div class:selected={i===selectedToken} class:state={result.tokenMeta[i].modality==='state'} class:vision={result.tokenMeta[i].modality==='vision'} class="token"
            on:mouseenter={()=>onSelectToken(i)}>
            <span>{result.tokenMeta[i].modality==='state'?'S':'V'} {timeLabel(result.tokenMeta[i].time)}</span>
            <div class="token-anchor token-vector">
              <UpstreamVectorCanvas data={token} colorScale={result.tokenMeta[i].modality==='state'?'blue':'green'} active={i===selectedToken}/>
            </div>
          </div>
        {/each}
      </div>
    </button>

    <button class="stage fusion-stage-attention" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>3</b><span>16×16 attention</span><small>time-causal, same-time cross-modal</small></div>
      <div class="attention-anchor matrix-wrap">
        <UpstreamMatrixSvg data={result.weights} cellHeight={9} cellWidth={9} rowGap={1.5} colGap={1.5}
          shape="rect" colorScale={(v)=>weightColor(v)}
          highlightRow={selectedToken} highlightCol={selectedToken}
          onMouseOverCell={(e,d)=>onSelectToken(d.colIndex)}
          showTooltip={(e,v)=>(v*100).toFixed(1)+'%'}/>
      </div>
      <div class="cross-read">
        <span>S(t) → V(t)</span><b>{(result.weights[latestState][latestVision]*100).toFixed(1)}%</b>
        <span>V(t) → S(t)</span><b>{(result.weights[latestVision][latestState]*100).toFixed(1)}%</b>
      </div>
    </button>

    <button class="stage fusion-stage-output" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>4</b><span>Fused state → force</span><small>average final S/V hidden → state head</small></div>
      <div class="state-anchor output-grid">
        <div><span>x</span><b>{result.inferredState[0].toFixed(2)}</b></div>
        <div><span>ẋ</span><b>{result.inferredState[1].toFixed(2)}</b></div>
        <div><span>θ</span><b>{(result.inferredState[2]*180/Math.PI).toFixed(1)}°</b></div>
        <div><span>θ̇</span><b>{(result.inferredState[3]*180/Math.PI).toFixed(1)}°/s</b></div>
      </div>
      <div class="force">{controllerForce>=0?'→':'←'} {Math.abs(controllerForce).toFixed(2)} N</div>
    </button>
  </div>
</section>

<style>
.fusion-pipeline{height:auto;min-height:500px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:12px;display:flex;flex-direction:column;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:0 3px 8px;border-bottom:1px solid #edf0f3;flex-wrap:wrap}.panel-head>div{display:flex;flex-direction:column}.panel-head strong{font-size:14px}.panel-head small{font-size:14px;color:#697386;white-space:normal}.panel-head>span{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7257b5}
.scenario-bar{display:flex;gap:4px;padding:8px 2px 0;flex-wrap:wrap}.scenario-bar button{border:1px solid #dfe3e8;background:#fff;border-radius:6px;padding:5px 8px;min-height:28px;font-size:10px;color:#6f7886;cursor:pointer}.scenario-bar button.active{background:#efeaf8;border-color:#a895cf;color:#604b95}
.fusion-overview{position:relative;flex:1;display:grid;grid-template-columns:.8fr 1.05fr 1.35fr .9fr;gap:18px;align-items:center;padding:14px 5px 8px}
.stage{appearance:none;border:0;background:transparent;padding:4px;position:relative;z-index:2;min-width:0;cursor:pointer;font:inherit}.stage:hover,.stage:focus-visible{transform:translateY(-2px)}
.stage-head{display:grid;grid-template-columns:22px 1fr;column-gap:7px;align-items:center;margin-bottom:8px;text-align:left}.stage-head>b{grid-row:1/3;width:22px;height:22px;line-height:22px;border-radius:50%;background:#293548;color:#fff;font-size:10px;text-align:center}.stage-head>span{font-size:14px;font-weight:700;color:#4b5563}.stage-head>small{font-size:14px;color:#697386}
.pair-stack{display:flex;flex-direction:column;gap:4px}.pair{display:flex;align-items:center;justify-content:center;gap:5px;padding:3px;border-radius:5px}.pair.selected{background:#f1eef9}.pair .time{width:46px;text-align:right;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7b8492}.pair-values{display:flex;gap:3px}.state-chip,.vision-chip{width:22px;height:18px;line-height:18px;border-radius:4px;text-align:center;font-size:10px;font-weight:700}.state-chip{background:#e7efff;color:#3c67b2}.vision-chip{background:#e7f7ef;color:#30805a}.state-chip.off,.vision-chip.off{opacity:.2;text-decoration:line-through}.vision-chip.partial{background:linear-gradient(90deg,#e7f7ef 50%,#f0f1f3 50%)}
.token-stack{display:grid;grid-template-columns:1fr 1fr;gap:3px}.token{display:flex;align-items:center;justify-content:center;gap:4px;padding:2px;border-radius:4px}.token.selected{background:#f1eef9}.token>span{width:46px;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#687386}.token-vector{position:relative;width:20px;height:28px;border:1px solid #e0e4e9;border-radius:3px;overflow:hidden}
.matrix-wrap{display:flex;justify-content:flex-start;overflow-x:auto;max-width:100%;padding:7px;border:1px solid #e4e7ec;border-radius:8px;background:#fbfbfd}.cross-read{display:grid;grid-template-columns:1fr auto;gap:4px 8px;margin-top:7px;font-size:10px;color:#717a88}.cross-read b{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7059aa}
.output-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}.output-grid>div{padding:6px;border:1px solid #e3e6eb;border-radius:6px;background:#fafbfc}.output-grid span{display:block;font-size:10px;color:#7b8492}.output-grid b{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}.force{margin-top:8px;padding:8px;border-radius:7px;background:#f1eef9;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#674e9f}
@media(max-width:980px){.fusion-pipeline{height:auto;overflow:visible;min-width:0}.fusion-pipeline .upstream-sankey{display:none!important}.fusion-overview{display:flex;flex-direction:column;gap:18px;min-width:0}.stage{width:100%;max-width:520px;min-width:0;border:1px solid #edf0f3;border-radius:10px;padding:10px}.stage:not(:last-child)::after{content:'↓';display:block;margin:8px auto -16px;color:#8b5cf6;font-size:18px}.token-stack{grid-template-columns:repeat(4,1fr)}}
/* 4 fixed-width (74px) token cells per row need ~300px; below that, drop to 3
   per row instead of clipping/overflowing the page. */
@media(max-width:400px){.token-stack{grid-template-columns:repeat(3,1fr)}}
</style>
