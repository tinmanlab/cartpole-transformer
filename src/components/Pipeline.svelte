<!--
Compact overview using visualization primitives adapted from
poloclub/transformer-explainer @ bfe50afba10b9b560b84143ee1107d977defa74f.
Detailed calculations live in TransformerDetail.svelte.
-->
<script>
  import * as d3 from 'd3';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  import UpstreamSankeyFlow from '../upstream/SankeyFlow.svelte';

  export let result;
  export let controllerForce = 0;
  export let selectedToken = 7;
  export let selectedRow = 7;
  export let selectedCol = 7;
  export let expandedStage = null;
  export let onSelectToken = () => {};
  export let onSelectAttention = () => {};
  export let onExpandedStageChange = () => {};

  $: n = result.rawTokens.length;
  $: last = n - 1;
  $: selectedToken = Math.min(Math.max(0, selectedToken), last);
  $: qSelected = result.q[selectedToken];
  $: kSelected = result.k[selectedToken];
  $: vSelected = result.v[selectedToken];
  $: contextSelected = result.perTokenContext?.[selectedRow] || result.context;
  $: hiddenSelected = result.hidden?.[selectedToken] || result.tokens[selectedToken];
  $: isLearned = result.modelType === 'learned-tiny-transformer';
  $: redrawKey = [selectedToken,selectedRow,selectedCol,expandedStage,result.modelType].join('|');

  $: weightColor = d3.scaleSequential(d3.interpolatePurples)
    .domain([0, Math.max(.001,d3.max(result.weights.flat()) || 1)]);

  function selectToken(i) {
    onSelectToken(i);
    onSelectAttention(i,i);
  }

  function hoverCell(event,d) {
    onSelectAttention(d.rowIndex,d.colIndex);
  }

  $: pathMap = {
    embeddingToQkv: [{
      from: '.embedding-token.selected .token-vector',
      to: '.qkv-token.selected .qkv-bundle',
      type: 'stroke',
      gradientId: 'gray-blue',
      opacity: .82,
      curve: 32,
      strokeWidth: 1.5
    }],
    qkvToAttention: [{
      from: '.qkv-token.selected .qkv-bundle',
      to: '.attention-overview .matrix-target',
      type: 'stroke',
      gradientId: 'blue-purple',
      opacity: .76,
      curve: 36,
      strokeWidth: 1.5
    }],
    attentionToBlock: [{
      from: '.attention-overview .context-target',
      to: '.block-overview .hidden-target',
      type: 'stroke',
      gradientId: 'purple-purple',
      opacity: .8,
      curve: 34,
      strokeWidth: 1.5
    }],
    blockToAction: [{
      from: '.block-overview .hidden-target',
      to: '.action-overview .force-target',
      type: 'stroke',
      gradientId: 'purple-purple',
      opacity: .82,
      curve: 28,
      strokeWidth: 1.5
    }]
  };
</script>

<section class="pipeline-shell resize-watch">
  <div class="panel-head">
    <div>
      <strong>Transformer · 1 block · 1 head</strong>
      <small>단계를 클릭하면 아래에서 live tensor 계산을 크게 펼칩니다</small>
    </div>
    <span>{isLearned ? 'LEARNED' : 'TOY'} · {controllerForce >= 0 ? '→' : '←'} {Math.abs(controllerForce).toFixed(2)} N</span>
  </div>

  <div class="overview resize-watch">
    <UpstreamSankeyFlow {pathMap} {redrawKey}/>

    <button type="button" class:expanded={expandedStage==='embedding'} class="stage embedding-overview" on:click={() => onExpandedStageChange(expandedStage==='embedding'?null:'embedding')}>
      <div class="stage-head"><b>1</b><span>Embedding</span><small>state → 8D token</small></div>
      <div class="token-column">
        {#each result.tokens as token,i}
          <div class:selected={i===selectedToken} class="embedding-token token-row" on:mouseenter={() => selectToken(i)} on:focus={() => selectToken(i)}>
            <span class="time-label">{i===last?'t':'t−'+(last-i)}</span>
            <div class="token-vector vector"><UpstreamVectorCanvas data={token} colorScale="gray" active={i===selectedToken}/></div>
            <span class="dim-label">8D</span>
          </div>
        {/each}
      </div>
    </button>

    <button type="button" class:expanded={expandedStage==='qkv'} class="stage qkv-overview" on:click={() => onExpandedStageChange(expandedStage==='qkv'?null:'qkv')}>
      <div class="stage-head"><b>2</b><span>Q · K · V</span><small>learned projections</small></div>
      <div class="qkv-labels"><span class="q">Q</span><span class="k">K</span><span class="v">V</span></div>
      <div class="token-column">
        {#each result.q as q,i}
          <div class:selected={i===selectedToken} class="qkv-token qkv-row" on:mouseenter={() => selectToken(i)} on:focus={() => selectToken(i)}>
            <span class="time-label">{i===last?'t':'−'+(last-i)}</span>
            <div class="qkv-bundle">
              <div class="vector mini"><UpstreamVectorCanvas data={q} colorScale="blue" active={i===selectedToken}/></div>
              <div class="vector mini"><UpstreamVectorCanvas data={result.k[i]} colorScale="red" active={i===selectedToken}/></div>
              <div class="vector mini"><UpstreamVectorCanvas data={result.v[i]} colorScale="green" active={i===selectedToken}/></div>
            </div>
          </div>
        {/each}
      </div>
    </button>

    <button type="button" class:expanded={expandedStage==='attention'} class="stage attention-overview" on:click={() => onExpandedStageChange(expandedStage==='attention'?null:'attention')}>
      <div class="stage-head"><b>3</b><span>Self Attention</span><small>weights × V → context</small></div>
      <div class="attention-core">
        <div class="matrix-target overview-matrix">
          <UpstreamMatrixSvg
            data={result.weights}
            cellHeight={17}
            cellWidth={17}
            rowGap={2}
            colGap={2}
            shape="rect"
            colorScale={(v)=>weightColor(v)}
            highlightRow={selectedRow}
            highlightCol={selectedCol}
            onMouseOverCell={hoverCell}
            showTooltip={(e,v)=>(v*100).toFixed(1)+'%'}
          />
        </div>
        <div class="attention-mix">
          <span>×</span>
          <div class="vector attention-vector"><UpstreamVectorCanvas data={vSelected} colorScale="green" active={true}/></div>
          <span>→</span>
          <div class="context-target vector attention-vector context-vector"><UpstreamVectorCanvas data={contextSelected} colorScale="purple" active={true}/></div>
        </div>
      </div>
      <div class="attention-read"><span>Q {selectedRow===last?'t':'t−'+(last-selectedRow)} → K {selectedCol===last?'t':'t−'+(last-selectedCol)}</span><strong>{(result.weights[selectedRow][selectedCol]*100).toFixed(1)}%</strong></div>
    </button>

    <button type="button" class:expanded={expandedStage==='block'} class="stage block-overview" on:click={() => onExpandedStageChange(expandedStage==='block'?null:'block')}>
      <div class="stage-head"><b>4</b><span>Transformer block</span><small>residual + MLP</small></div>
      <div class="block-live">
        <div class="block-op">context</div>
        <span>→</span>
        <div class="block-op">Wₒ + residual</div>
        <span>→</span>
        <div class="block-op">LN · GELU MLP</div>
        <span>→</span>
        <div class="hidden-target vector hidden-vector"><UpstreamVectorCanvas data={hiddenSelected} colorScale="blue" active={true}/></div>
      </div>
      <code>{hiddenSelected.slice(0,3).map(v=>v.toFixed(2)).join(' ')}</code>
    </button>

    <button type="button" class:expanded={expandedStage==='action'} class="stage action-overview" on:click={() => onExpandedStageChange(expandedStage==='action'?null:'action')}>
      <div class="stage-head"><b>5</b><span>Action</span><small>final t only</small></div>
      <div class="force-target force-visual">{controllerForce>=0?'→':'←'}</div>
      <strong class="force-value">{Math.abs(controllerForce).toFixed(2)} N</strong>
      <small>tanh(score) × 10</small>
    </button>
  </div>
</section>

<style>
.pipeline-shell{height:auto;min-height:500px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:12px;display:flex;flex-direction:column;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:0 3px 10px;border-bottom:1px solid #edf0f3;gap:12px;flex-wrap:wrap}.panel-head>div{display:flex;flex-direction:column}.panel-head strong{font-size:14px}.panel-head small{font-size:14px;color:#697386;white-space:normal}.panel-head>span{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7257b5;white-space:nowrap}
.overview{position:relative;flex:1;display:grid;grid-template-columns:.9fr 1fr 1.55fr 1.25fr .65fr;gap:18px;align-items:center;padding:24px 6px 12px;min-height:0}
.stage{appearance:none;border:0;background:transparent;padding:0;position:relative;z-index:2;min-width:0;text-align:center;cursor:pointer;font:inherit;transition:transform .18s,opacity .18s}.stage:hover,.stage:focus-visible{transform:translateY(-2px)}.stage.expanded{outline:2px solid #ded7f0;outline-offset:6px;border-radius:8px}
.stage-head{display:grid;grid-template-columns:22px 1fr;column-gap:7px;align-items:center;margin-bottom:9px;text-align:left}.stage-head>b{grid-row:1/3;width:22px;height:22px;line-height:22px;border-radius:50%;background:#293548;color:#fff;font-size:10px;text-align:center}.stage-head>span{font-size:14px;font-weight:700;color:#4b5563}.stage-head>small{font-size:14px;color:#697386}
.token-column{display:flex;flex-direction:column;gap:3px}.token-row,.qkv-row{height:32px;display:flex;align-items:center;justify-content:center;gap:5px;border-radius:6px;padding:2px 4px}.token-row.selected,.qkv-row.selected{background:#f1eefb}.time-label{width:27px;text-align:right;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7b8492}.dim-label{font-size:10px;color:#8e96a2}.vector{position:relative;width:22px;height:28px;border:1px solid #e0e3e8;border-radius:4px;overflow:hidden;background:white}.qkv-bundle{height:28px;display:grid;grid-template-columns:repeat(3,19px);gap:3px}.vector.mini{width:19px;height:28px}.qkv-labels{display:grid;grid-template-columns:repeat(3,19px);gap:3px;justify-content:center;margin:0 0 5px 31px;font-size:10px;font-weight:700}.q{color:#3971e8}.k{color:#d95757}.v{color:#34966a}
.attention-core{display:grid;grid-template-columns:auto 44px;gap:9px;align-items:center;justify-content:center}.overview-matrix{display:flex;justify-content:center;padding:9px;border:1px solid #e5e7eb;background:#fbfbfd;border-radius:9px}.attention-mix{display:grid;grid-template-columns:9px 23px 9px 25px;gap:4px;align-items:center;font-size:11px;color:#7b8492}.attention-vector{width:23px;height:58px}.context-vector{outline:2px solid #ded7f2}.attention-read{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:7px;margin-top:8px;font-size:14px;color:#596273}.attention-read strong{color:#7359b7;font-size:14px}
.block-live{display:flex;align-items:center;justify-content:center;gap:5px;flex-wrap:wrap}.block-live>span{font-size:10px;color:#8270b0}.block-op{padding:6px 7px;border:1px solid #e3e6eb;border-radius:5px;background:#fafbfc;font-size:10px;color:#667085}.hidden-vector{height:72px;width:23px}.block-overview>code{display:block;margin-top:7px;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#697386}
.force-visual{font-size:48px;color:#8b5cf6;line-height:.85}.force-value{display:block;font:14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6d52aa;margin-top:8px}.action-overview>small{font-size:14px;color:#697386}
/* stage-head/panel-head/attention-read/force-value stay >=14px at every
   width per the P0 design contract; the narrower 5-column overview below 1220px
   wraps these labels onto a second line instead of shrinking them back down */
@media(max-width:1220px){.overview{gap:11px;grid-template-columns:.85fr .95fr 1.45fr 1.16fr .62fr}}
@media(max-width:980px){
  .pipeline-shell{height:auto;min-height:0;overflow:visible}
  .pipeline-shell .upstream-sankey{display:none}
  .overview{display:flex;flex-direction:column;gap:22px;padding:20px 8px 12px}
  .stage{width:100%;max-width:520px;padding:8px 12px;border:1px solid #edf0f3;border-radius:10px;background:#fff}
  .stage:not(:last-child)::after{content:'↓';display:block;margin:10px auto -18px;color:#8b5cf6;font-size:18px;font-weight:700}
  .token-column{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px}
  .qkv-labels{margin-left:0}
  .token-row,.qkv-row{height:34px}
  .attention-core{grid-template-columns:auto 48px}
  .block-live{gap:7px}
}
</style>
