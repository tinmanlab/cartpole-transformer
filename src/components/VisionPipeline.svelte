<script>
  import * as d3 from 'd3';
  import VisionFrameCanvas from './VisionFrameCanvas.svelte';
  import VisionPatchGrid from './VisionPatchGrid.svelte';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  import UpstreamSankeyFlow from '../upstream/SankeyFlow.svelte';

  export let frames = [];
  export let result;
  export let controllerForce = 0;
  export let selectedFrame = 7;
  export let onSelectFrame = () => {};
  export let onOpenDetail = () => {};

  $: n = frames.length;
  $: last = Math.max(0,n-1);
  $: selectedFrame = Math.min(Math.max(0,selectedFrame),last);
  $: patches = result.frameFeatures[selectedFrame] || [];
  $: patchGridSize = Math.round(Math.sqrt(patches.length));
  $: token = result.tokens[selectedFrame] || [];
  $: inferredMotion = result.inferredMotion || [0,0];
  $: weightColor = d3.scaleSequential(d3.interpolatePurples)
    .domain([0,Math.max(.001,d3.max(result.weights.flat()) || 1)]);
  $: redrawKey=[selectedFrame,result.modelType].join('|');

  const pathMap = {
    frameToPatch:[{
      from:'.vision-stage-frame .selected-frame-anchor',
      to:'.vision-stage-patch .patch-anchor',
      type:'stroke',gradientId:'gray-blue',opacity:.8,curve:30,strokeWidth:1.5
    }],
    patchToToken:[{
      from:'.vision-stage-patch .patch-anchor',
      to:'.vision-stage-token .selected-token-anchor',
      type:'stroke',gradientId:'gray-blue',opacity:.8,curve:30,strokeWidth:1.5
    }],
    tokenToAttention:[{
      from:'.vision-stage-token .selected-token-anchor',
      to:'.vision-stage-attention .attention-anchor',
      type:'stroke',gradientId:'blue-purple',opacity:.8,curve:30,strokeWidth:1.5
    }],
    attentionToAction:[{
      from:'.vision-stage-attention .attention-anchor',
      to:'.vision-stage-output .output-anchor',
      type:'stroke',gradientId:'purple-purple',opacity:.8,curve:30,strokeWidth:1.5
    }]
  };

  function label(i){return i===last?'t':'t−'+(last-i);}
</script>

<section class="vision-pipeline resize-watch">
  <div class="panel-head">
    <div><strong>Vision-only · pixels → temporal Transformer</strong><small>controller input에는 simulator state 숫자가 들어가지 않습니다</small></div>
    <span>{controllerForce>=0?'→':'←'} {Math.abs(controllerForce).toFixed(2)} N</span>
  </div>

  <div class="vision-overview resize-watch">
    <UpstreamSankeyFlow {pathMap} {redrawKey}/>

    <button class="stage vision-stage-frame" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>1</b><span>8 Frames</span><small>32×32 pixels each</small></div>
      <div class="frame-strip">
        {#each frames as frame,i}
          <div class:selected={i===selectedFrame} class:selected-frame-anchor={i===selectedFrame} class="frame-thumb"
            on:mouseenter={()=>onSelectFrame(i)} on:focus={()=>onSelectFrame(i)}>
            <VisionFrameCanvas {frame} size={32} label={label(i)} active={i===selectedFrame} pixelScale={1.05}/>
          </div>
        {/each}
      </div>
    </button>

    <button class="stage vision-stage-patch" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>2</b><span>Patch features</span><small>2×2 average → 256D</small></div>
      <div class="patch-anchor patch-mini">
        <VisionPatchGrid {patches} gridSize={patchGridSize} patchSize={2} cellSize={patchGridSize > 8 ? 5 : 9}/>
      </div>
    </button>

    <button class="stage vision-stage-token" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>3</b><span>Frame tokens</span><small>learned 64→16 + time</small></div>
      <div class="token-stack">
        {#each result.tokens as t,i}
          <div class:selected={i===selectedFrame} class="frame-token" on:mouseenter={()=>onSelectFrame(i)}>
            <span>{label(i)}</span>
            <div class:selected-token-anchor={i===selectedFrame} class="token-vector">
              <UpstreamVectorCanvas data={t} colorScale="blue" active={i===selectedFrame}/>
            </div>
          </div>
        {/each}
      </div>
    </button>

    <button class="stage vision-stage-attention" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>4</b><span>Temporal attention</span><small>which earlier frames matter?</small></div>
      <div class="attention-anchor matrix-wrap">
        <UpstreamMatrixSvg data={result.weights} cellHeight={14} cellWidth={14} rowGap={2} colGap={2}
          shape="rect" colorScale={(v)=>weightColor(v)}
          highlightRow={last} highlightCol={selectedFrame}
          onMouseOverCell={(e,d)=>onSelectFrame(d.colIndex)}
          showTooltip={(e,v)=>(v*100).toFixed(1)+'%'}/>
      </div>
      <div class="latest-weight">Q t → {label(selectedFrame)} <b>{(result.weights[last][selectedFrame]*100).toFixed(1)}%</b></div>
    </button>

    <button class="stage vision-stage-output" type="button" on:click={onOpenDetail}>
      <div class="stage-head"><b>5</b><span>Motion + action</span><small>inferred from pixels</small></div>
      <div class="output-anchor output">
        <div><span>ẋ estimate</span><b>{inferredMotion[0].toFixed(2)} m/s</b></div>
        <div><span>θ̇ estimate</span><b>{(inferredMotion[1]*180/Math.PI).toFixed(1)}°/s</b></div>
        <div class="force"><span>force</span><strong>{controllerForce>=0?'→':'←'} {Math.abs(controllerForce).toFixed(2)} N</strong></div>
      </div>
    </button>
  </div>
</section>

<style>
.vision-pipeline{height:500px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:12px;display:flex;flex-direction:column;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:0 3px 10px;border-bottom:1px solid #edf0f3}.panel-head>div{display:flex;flex-direction:column}.panel-head strong{font-size:14px}.panel-head small{font-size:10px;color:#8a93a2}.panel-head>span{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7257b5}
.vision-overview{position:relative;flex:1;display:grid;grid-template-columns:1.35fr .9fr .8fr 1fr .75fr;gap:18px;align-items:center;padding:20px 5px 10px}
.stage{appearance:none;border:0;background:transparent;padding:4px;position:relative;z-index:2;min-width:0;cursor:pointer;font:inherit}.stage:hover,.stage:focus-visible{transform:translateY(-2px)}
.stage-head{display:grid;grid-template-columns:22px 1fr;column-gap:7px;align-items:center;margin-bottom:9px;text-align:left}.stage-head>b{grid-row:1/3;width:22px;height:22px;line-height:22px;border-radius:50%;background:#293548;color:#fff;font-size:10px;text-align:center}.stage-head>span{font-size:11px;font-weight:700;color:#4b5563}.stage-head>small{font-size:9px;color:#9299a5}
.frame-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;justify-items:center}.frame-thumb{opacity:.58}.frame-thumb.selected{opacity:1}
.patch-mini{display:flex;justify-content:center}.patch-mini :global(.caption),.patch-mini :global(.axis){display:none}
.token-stack{display:flex;flex-direction:column;gap:3px}.frame-token{display:flex;align-items:center;justify-content:center;gap:5px;padding:2px;border-radius:5px}.frame-token.selected{background:#eef3fb}.frame-token>span{width:23px;text-align:right;font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7b8492}.token-vector{position:relative;width:24px;height:29px;border:1px solid #dfe3e8;border-radius:4px;overflow:hidden}
.matrix-wrap{display:flex;justify-content:center;padding:8px;border:1px solid #e4e7ec;border-radius:9px;background:#fbfbfd}.latest-weight{text-align:center;margin-top:7px;font-size:9px;color:#6f7886}.latest-weight b{color:#7257b5}
.output{display:flex;flex-direction:column;gap:8px}.output>div{padding:7px;border:1px solid #e4e7ec;border-radius:7px;background:#fafbfc}.output span{display:block;font-size:8px;color:#87909e}.output b{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}.output .force{background:#f3f0fa}.output strong{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#674e9f}
@media(max-width:980px){.vision-pipeline{height:auto;overflow:visible}.vision-pipeline .upstream-sankey{display:none!important}.vision-overview{display:flex;flex-direction:column;gap:18px}.stage{width:100%;max-width:520px;border:1px solid #edf0f3;border-radius:10px;padding:10px}.stage:not(:last-child)::after{content:'↓';display:block;margin:8px auto -16px;color:#8b5cf6;font-size:18px}.frame-strip{grid-template-columns:repeat(4,auto)}}
</style>
