<script>
  import * as d3 from 'd3';
  import VisionFrameCanvas from './VisionFrameCanvas.svelte';
  import VisionPatchGrid from './VisionPatchGrid.svelte';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';

  export let frames = [];
  export let result;
  export let repeatedResult = null;
  export let selectedFrame = 7;
  export let groundTruthState = null;
  export let onSelectFrame = () => {};
  export let onClose = () => {};
  export let frameIntervalMs = 60;

  let showGroundTruth = false;
  let selectedPatch = null;

  $: n=frames.length;
  $: last=Math.max(0,n-1);
  $: selectedFrame=Math.min(Math.max(0,selectedFrame),last);
  $: patches=result.frameFeatures[selectedFrame] || [];
  $: gridSize=Math.round(Math.sqrt(patches.length));
  $: token=result.tokens[selectedFrame] || [];
  $: latestWeights=result.weights[last] || [];
  $: inferred=result.inferredState || [0,0,0,0];
  $: repeated=repeatedResult?.inferredState || [0,0,0,0];
  $: weightColor=d3.scaleSequential(d3.interpolatePurples).domain([0,Math.max(.001,d3.max(result.weights.flat()) || 1)]);

  function label(i){return i===last?'t':'−'+Math.round((last-i)*frameIntervalMs)+'ms';}
</script>

<section class="vision-detail-wide">
  <header class="detail-head">
    <div>
      <div class="eyebrow">VISION-ONLY · LIVE PIXEL PIPELINE</div>
      <h2>Pixels → patches → frame token → temporal attention</h2>
      <p>아래 값은 controller가 실제로 소비하는 visual observation과 같은 데이터입니다. simulator state는 입력에 포함되지 않습니다.</p>
    </div>
    <button type="button" on:click={onClose} aria-label="close vision detail">×</button>
  </header>

  <div class="frame-history">
    {#each frames as frame,i}
      <button type="button" class:active={i===selectedFrame} on:click={()=>onSelectFrame(i)}>
        <VisionFrameCanvas {frame} size={32} label={label(i)} active={i===selectedFrame} pixelScale={1.45}/>
      </button>
    {/each}
  </div>

  <div class="vision-flow">
    <article>
      <div class="step">A</div>
      <h3>선택한 pixel frame</h3>
      <VisionFrameCanvas frame={frames[selectedFrame]} size={32} label={label(selectedFrame)} active={true} pixelScale={4}/>
      <p>한 장에서는 위치와 각도는 보이지만, 어느 방향으로 움직이는지는 완전히 결정되지 않습니다.</p>
    </article>

    <div class="arrow">→</div>

    <article>
      <div class="step">B</div>
      <h3>{gridSize}×{gridSize} patch features</h3>
      <VisionPatchGrid {patches} {gridSize} patchSize={2} cellSize={7}
        selected={selectedPatch}
        onSelect={(row,col,value)=>selectedPatch={row,col,value}}/>
      {#if selectedPatch}
        <div class="patch-read">patch [{selectedPatch.row},{selectedPatch.col}] mean = <b>{selectedPatch.value.toFixed(3)}</b></div>
      {/if}
    </article>

    <div class="arrow">→</div>

    <article>
      <div class="step">C</div>
      <h3>Learned frame token</h3>
      <div class="token-large"><UpstreamVectorCanvas data={token} colorScale="blue" active={true}/></div>
      <code>{token.slice(0,8).map(v=>v.toFixed(3)).join(' ')}</code>
      <p>{patches.length} patch values를 learned Linear가 {token.length}D token으로 압축하고 time position을 더합니다.</p>
    </article>

    <div class="arrow">→</div>

    <article class="attention-article">
      <div class="step">D</div>
      <h3>8-frame temporal attention</h3>
      <div class="matrix">
        <UpstreamMatrixSvg data={result.weights} cellHeight={18} cellWidth={18} rowGap={3} colGap={3}
          shape="rect" colorScale={(v)=>weightColor(v)}
          highlightRow={last} highlightCol={selectedFrame}
          onMouseOverCell={(e,d)=>onSelectFrame(d.colIndex)}
          showTooltip={(e,v)=>(v*100).toFixed(2)+'%'}/>
      </div>
      <div class="weight-row">
        {#each latestWeights as w,i}
          <button type="button" class:active={i===selectedFrame} on:click={()=>onSelectFrame(i)}>
            <span>{label(i)}</span><b>{(w*100).toFixed(1)}%</b>
          </button>
        {/each}
      </div>
    </article>

    <div class="arrow">→</div>

    <article class="inference-article">
      <div class="step">E</div>
      <h3>Pixels에서 추정한 state</h3>
      <div class="infer-grid">
        <div><span>x</span><b>{inferred[0].toFixed(2)} m</b></div>
        <div><span>ẋ</span><b>{inferred[1].toFixed(2)} m/s</b></div>
        <div><span>θ</span><b>{(inferred[2]*180/Math.PI).toFixed(1)}°</b></div>
        <div><span>θ̇</span><b>{(inferred[3]*180/Math.PI).toFixed(1)}°/s</b></div>
      </div>
      {#if repeatedResult}
        <div class="ablation">
          <h4>왜 여러 frame이 필요한가?</h4>
          <div><span>8-frame ẋ</span><b>{inferred[1].toFixed(2)}</b><span>latest-only ẋ</span><b>{repeated[1].toFixed(2)}</b></div>
          <div><span>8-frame θ̇</span><b>{(inferred[3]*180/Math.PI).toFixed(1)}°/s</b><span>latest-only θ̇</span><b>{(repeated[3]*180/Math.PI).toFixed(1)}°/s</b></div>
        </div>
      {/if}
      <button class="truth-toggle" type="button" on:click={()=>showGroundTruth=!showGroundTruth}>{showGroundTruth?'Hide':'Reveal'} ground-truth reference</button>
      {#if showGroundTruth && groundTruthState}
        <div class="truth">
          <span>x {groundTruthState[0].toFixed(2)} m</span>
          <span>ẋ {groundTruthState[1].toFixed(2)} m/s</span>
          <span>θ {(groundTruthState[2]*180/Math.PI).toFixed(1)}°</span>
          <span>θ̇ {(groundTruthState[3]*180/Math.PI).toFixed(1)}°/s</span>
        </div>
      {/if}
    </article>
  </div>
</section>

<style>
.vision-detail-wide{margin-top:10px;background:#fff;border:1px solid #e1e5ea;border-radius:16px;padding:18px 22px 22px;overflow:hidden}
.detail-head{display:flex;justify-content:space-between;gap:20px;padding-bottom:14px;border-bottom:1px solid #edf0f3}.eyebrow{font-size:10px;letter-spacing:.08em;color:#7d6aaa}.detail-head h2{font-size:20px;margin:2px 0 5px}.detail-head p{font-size:12px;color:#667085;margin:0;line-height:1.5}.detail-head button{width:36px;height:36px;border:1px solid #dfe3e8;border-radius:9px;background:#fff;font-size:20px;color:#667085;cursor:pointer}
.frame-history{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;padding:14px 0}.frame-history>button{border:0;background:transparent;padding:0;opacity:.58;cursor:pointer}.frame-history>button.active{opacity:1}
.vision-flow{display:grid;grid-template-columns:1.1fr 28px 1.2fr 28px .9fr 28px 1.2fr 28px 1.2fr;gap:8px;align-items:stretch}.vision-flow article{position:relative;min-width:0;padding:14px;border:1px solid #e4e7ec;border-radius:10px;background:#fbfcfd;display:flex;flex-direction:column;align-items:center;gap:9px}.vision-flow h3{font-size:12px;margin:0;color:#4b5563}.vision-flow p{font-size:10px;line-height:1.45;color:#737d8b;margin:0;text-align:center}.step{position:absolute;top:8px;right:8px;width:20px;height:20px;line-height:20px;border-radius:50%;background:#293548;color:#fff;text-align:center;font-size:9px;font-weight:700}.arrow{display:flex;align-items:center;justify-content:center;color:#826db7;font-size:19px}
.token-large{position:relative;width:38px;height:132px;border:1px solid #dfe3e8;border-radius:5px;overflow:hidden}.vision-flow code{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085;word-break:break-all;text-align:center}.matrix{padding:6px;background:#fff;border:1px solid #e5e7eb;border-radius:8px}.weight-row{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}.weight-row button{border:1px solid #e0e3e8;background:#fff;border-radius:5px;padding:4px;display:flex;flex-direction:column;font-size:8px;color:#667085}.weight-row button.active{background:#ece7f7;border-color:#a895cf}.infer-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;width:100%}.infer-grid>div{padding:7px;border:1px solid #e3e6eb;border-radius:6px;background:#fff}.infer-grid span,.ablation span{font-size:8px;color:#858e9b;display:block}.infer-grid b,.ablation b{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}.ablation{width:100%;padding:8px;border-radius:7px;background:#f3f0fa}.ablation h4{font-size:9px;margin:0 0 6px;color:#6b58a0}.ablation>div{display:grid;grid-template-columns:1fr auto 1fr auto;gap:4px;margin-top:4px}.truth-toggle{border:1px solid #dfe3e8;border-radius:6px;background:#fff;padding:6px 8px;font-size:9px;color:#667085;cursor:pointer}.truth{display:grid;grid-template-columns:1fr 1fr;gap:3px;font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085}
.patch-read{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085}
@media(max-width:980px){.vision-detail-wide{padding:14px}.vision-flow{display:flex;flex-direction:column}.arrow{transform:rotate(90deg);height:24px}.vision-flow article{width:100%}.detail-head h2{font-size:18px}}
</style>
