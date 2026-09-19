<!--
Cart-Pole adapter around visualization primitives adapted from
poloclub/transformer-explainer @ bfe50afba10b9b560b84143ee1107d977defa74f
-->
<script>
  import { tick } from 'svelte';
  import * as d3 from 'd3';
  import { gsap } from 'gsap';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  import UpstreamSankeyFlow from '../upstream/SankeyFlow.svelte';
  import UpstreamAttentionExpansion from '../upstream/AttentionExpansion.svelte';
  import { NORMALIZATION_SCALE } from '../lib/attention.js';

  export let history = [];
  export let result;
  export let controllerForce = 0;
  export let selectedToken = 7;
  export let onSelectToken = () => {};

  let selectedRow = 7;
  let selectedCol = 7;
  let expandedStage = null;
  let detailEl;

  const names = ['x','ẋ','θ','θ̇'];
  const units = ['m','m/s','rad','rad/s'];

  $: n = result.rawTokens.length;
  $: last = n - 1;
  $: selectedToken = Math.min(Math.max(0, selectedToken), last);
  $: rawSelected = result.rawTokens[selectedToken];
  $: normalizedSelected = result.normalizedTokens[selectedToken];
  $: tokenSelected = result.tokens[selectedToken];
  $: stateEmbeddingSelected = result.stateEmbeddings?.[selectedToken] || normalizedSelected;
  $: posEmbeddingSelected = result.positionEmbeddings?.[selectedToken] || new Array(tokenSelected.length).fill(0);
  $: norm1Selected = result.norm1?.[selectedToken] || tokenSelected;
  $: qSelected = result.q[selectedToken];
  $: kSelected = result.k[selectedToken];
  $: vSelected = result.v[selectedToken];
  $: contextSelected = result.perTokenContext?.[selectedToken] || result.context;
  $: attendedSelected = result.attended?.[selectedToken] || contextSelected;
  $: residual1Selected = result.residual1?.[selectedToken] || tokenSelected;
  $: norm2Selected = result.norm2?.[selectedToken] || residual1Selected;
  $: mlpUpSelected = result.mlpUp?.[selectedToken] || norm2Selected;
  $: mlpSelected = result.mlp?.[selectedToken] || norm2Selected;
  $: hiddenSelected = result.hidden?.[selectedToken] || residual1Selected;
  $: isLearned = result.modelType === 'learned-tiny-transformer';
  $: redrawKey = [selectedToken, expandedStage, result.modelType].join('|');

  $: weightColor = d3.scaleSequential(d3.interpolatePurples)
    .domain([0, Math.max(.001,d3.max(result.weights.flat()) || 1)]);
  const paramColor = d3.scaleDiverging()
    .domain([-1,0,1])
    .range(['#ef9a9a','#ffffff','#93b4ee'])
    .clamp(true);

  function hoverCell(event,d) {
    selectedRow = d.rowIndex;
    selectedCol = d.colIndex;
    onSelectToken(d.colIndex);
  }

  async function toggleStage(stage) {
    expandedStage = expandedStage === stage ? null : stage;
    await tick();
    if (detailEl && expandedStage) {
      const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      if (!reduce) gsap.fromTo(detailEl,{opacity:0,y:8},{opacity:1,y:0,duration:.28,ease:'power2.out'});
    }
  }

  function selectToken(i) {
    onSelectToken(i);
    selectedRow = i;
    selectedCol = i;
  }

  $: pathMap = {
    embedding: [
      {
        from: '.embedding-token .token-vector',
        to: '.qkv-token .qkv-bundle',
        gradientId: 'gray-blue',
        opacity: .16,
        curve: 42,
        arrow: false
      },
      {
        from: '.embedding-token.selected .token-vector',
        to: '.qkv-token.selected .qkv-bundle',
        type: 'stroke',
        gradientId: 'gray-blue',
        opacity: .95,
        curve: 42
      }
    ],
    query: [{
      from: '.qkv-token.selected .q-slot',
      to: '.attention-overview .matrix-target',
      type: 'stroke',
      gradientId: 'blue-purple',
      opacity: .9,
      curve: 58
    }],
    key: [{
      from: '.qkv-token.selected .k-slot',
      to: '.attention-overview .matrix-target',
      type: 'stroke',
      gradientId: 'blue-mix',
      opacity: .8,
      curve: 62
    }],
    value: [{
      from: '.qkv-token.selected .v-slot',
      to: '.block-overview .attn-out-target',
      type: 'stroke',
      gradientId: 'green-purple',
      opacity: .72,
      curve: 72
    }],
    attentionOut: [{
      from: '.attention-overview .matrix-target',
      to: '.block-overview .attn-out-target',
      type: 'stroke',
      gradientId: 'blue-purple',
      opacity: .82,
      curve: 50
    }],
    action: [{
      from: '.block-overview .hidden-target',
      to: '.action-overview .force-target',
      type: 'stroke',
      gradientId: 'purple-purple',
      opacity: .9,
      curve: 48
    }]
  };

  const stageInfo = {
    embedding: '상태 4개를 정규화하고 learned 4→8 projection과 position vector를 더해 token을 만듭니다.',
    qkv: 'LayerNorm 뒤 같은 token을 서로 다른 WQ, WK, WV에 통과시켜 세 역할로 분리합니다.',
    attention: 'Q와 K를 비교하고 미래를 mask한 뒤 softmax로 V를 얼마나 가져올지 결정합니다.',
    block: 'Attention output을 residual로 더하고, LayerNorm→MLP(GELU)→residual을 지나 hidden state를 만듭니다.',
    action: '마지막 hidden token만 action head가 읽고 tanh를 거쳐 실제 ±10 N force를 냅니다.'
  };
</script>

<section class:has-detail={!!expandedStage} class="pipeline-shell resize-watch">
  <div class="panel-head">
    <div>
      <strong>Transformer · 1 block · 1 head</strong>
      <small>단계를 클릭하면 실제 live 계산을 펼칩니다</small>
    </div>
    <span>{isLearned ? 'LEARNED' : 'TOY FALLBACK'} · {controllerForce >= 0 ? '→' : '←'} {Math.abs(controllerForce).toFixed(1)} N</span>
  </div>

  <div class:dimmed={!!expandedStage} class="overview resize-watch">
    <UpstreamSankeyFlow {pathMap} {redrawKey}/>

    <button type="button" class:expanded={expandedStage==='embedding'} class="stage embedding-overview" on:click={() => toggleStage('embedding')}>
      <div class="stage-head"><b>1</b><span>Embedding</span><small>state → token</small></div>
      <div class="token-column">
        {#each result.tokens as token,i}
          <div class:selected={i===selectedToken} class="embedding-token token-row" on:mouseenter={() => selectToken(i)} on:focus={() => selectToken(i)}>
            <span class="time-label">{i===last?'t':'t−'+(last-i)}</span>
            <div class="token-vector vector"><UpstreamVectorCanvas data={token} colorScale="gray" active={i===selectedToken}/></div>
            <span class="dim-label">{token.length}D</span>
          </div>
        {/each}
      </div>
    </button>

    <button type="button" class:expanded={expandedStage==='qkv'} class="stage qkv-overview" on:click={() => toggleStage('qkv')}>
      <div class="stage-head"><b>2</b><span>Q · K · V</span><small>three learned views</small></div>
      <div class="qkv-labels"><span class="q">Q</span><span class="k">K</span><span class="v">V</span></div>
      <div class="token-column">
        {#each result.q as q,i}
          <div class:selected={i===selectedToken} class="qkv-token qkv-row" on:mouseenter={() => selectToken(i)} on:focus={() => selectToken(i)}>
            <span class="time-label">{i===last?'t':'−'+(last-i)}</span>
            <div class="qkv-bundle">
              <div class="q-slot vector mini"><UpstreamVectorCanvas data={q} colorScale="blue" active={i===selectedToken}/></div>
              <div class="k-slot vector mini"><UpstreamVectorCanvas data={result.k[i]} colorScale="red" active={i===selectedToken}/></div>
              <div class="v-slot vector mini"><UpstreamVectorCanvas data={result.v[i]} colorScale="green" active={i===selectedToken}/></div>
            </div>
          </div>
        {/each}
      </div>
    </button>

    <button type="button" class:expanded={expandedStage==='attention'} class="stage attention-overview" on:click={() => toggleStage('attention')}>
      <div class="stage-head"><b>3</b><span>Self Attention</span><small>QK → mask → softmax</small></div>
      <div class="matrix-target overview-matrix">
        <UpstreamMatrixSvg
          data={result.weights}
          cellHeight={15}
          cellWidth={15}
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
      <div class="attention-read">
        <span>Query</span><b>{selectedRow===last?'t':'t−'+(last-selectedRow)}</b>
        <span>looks at</span><b>{selectedCol===last?'t':'t−'+(last-selectedCol)}</b>
        <strong>{(result.weights[selectedRow][selectedCol]*100).toFixed(1)}%</strong>
      </div>
    </button>

    <button type="button" class:expanded={expandedStage==='block'} class="stage block-overview" on:click={() => toggleStage('block')}>
      <div class="stage-head"><b>4</b><span>Block output</span><small>residual + MLP</small></div>
      <div class="block-chain">
        <div><small>Attn out</small><div class="attn-out-target vector"><UpstreamVectorCanvas data={attendedSelected} colorScale="purple" active={true}/></div></div>
        <span>+</span>
        <div><small>Residual</small><div class="vector"><UpstreamVectorCanvas data={residual1Selected} colorScale="blue" active={true}/></div></div>
        <span>→</span>
        <div><small>MLP</small><div class="vector"><UpstreamVectorCanvas data={mlpSelected} colorScale="purple" active={true}/></div></div>
        <span>+</span>
        <div><small>Hidden</small><div class="hidden-target vector"><UpstreamVectorCanvas data={hiddenSelected} colorScale="blue" active={true}/></div></div>
      </div>
    </button>

    <button type="button" class:expanded={expandedStage==='action'} class="stage action-overview" on:click={() => toggleStage('action')}>
      <div class="stage-head"><b>5</b><span>Action head</span><small>last token only</small></div>
      <div class="force-target force-visual">{controllerForce>=0?'→':'←'}</div>
      <strong class="force-value">{Math.abs(controllerForce).toFixed(1)} N</strong>
      <small>tanh(score) × 10</small>
    </button>
  </div>

  {#if expandedStage}
    <div class="detail-panel" bind:this={detailEl}>
      <div class="detail-head">
        <div><b>{expandedStage === 'embedding' ? 'Embedding' : expandedStage === 'qkv' ? 'Q · K · V' : expandedStage === 'attention' ? 'Self Attention' : expandedStage === 'block' ? 'Residual + MLP' : 'Action head'}</b><span>{stageInfo[expandedStage]}</span></div>
        <button type="button" on:click={() => toggleStage(expandedStage)} aria-label="close detail">×</button>
      </div>

      {#if expandedStage === 'embedding'}
        <div class="calc-line embedding-detail">
          <div class="calc-item raw">
            <b>Raw state · {selectedToken===last?'t':'t−'+(last-selectedToken)}</b>
            <div class="number-vector">{#each rawSelected as value,i}<span><small>{names[i]}</small>{value.toFixed(i<2?2:3)}</span>{/each}</div>
          </div>
          <div class="math-arrow"><span>÷ scale</span>→</div>
          <div class="calc-item">
            <b>Normalized 4D</b>
            <div class="number-vector">{#each normalizedSelected as value}<span>{value.toFixed(3)}</span>{/each}</div>
          </div>
          <div class="math-arrow"><span>Linear 4→8</span>→</div>
          <div class="calc-item">
            <b>State embedding</b>
            <div class="wide-vector"><UpstreamVectorCanvas data={stateEmbeddingSelected} colorScale="blue" active={true}/></div>
          </div>
          <div class="math-symbol">+</div>
          <div class="calc-item">
            <b>Position {selectedToken}</b>
            <div class="wide-vector"><UpstreamVectorCanvas data={posEmbeddingSelected} colorScale="red" active={true}/></div>
          </div>
          <div class="math-symbol">=</div>
          <div class="calc-item">
            <b>Token {tokenSelected.length}D</b>
            <div class="wide-vector final"><UpstreamVectorCanvas data={tokenSelected} colorScale="gray" active={true}/></div>
          </div>
        </div>
        {#if isLearned && result.modelWeights?.embed}
          <div class="weight-peek"><span>learned embedding weight · 8×4</span><UpstreamMatrixSvg data={result.modelWeights.embed.weight} cellHeight={8} cellWidth={8} rowGap={1} colGap={1} shape="rect" colorScale={(v)=>paramColor(v)} showTooltip={(e,v)=>v.toFixed(3)}/></div>
        {/if}
      {:else if expandedStage === 'qkv'}
        <div class="qkv-detail">
          <div class="calc-item source-token"><b>LayerNorm(token)</b><div class="wide-vector"><UpstreamVectorCanvas data={norm1Selected} colorScale="gray" active={true}/></div></div>
          {#each [
            ['Q','blue',result.modelWeights?.q?.weight,qSelected,'무엇을 찾을까?'],
            ['K','red',result.modelWeights?.k?.weight,kSelected,'나는 어떤 정보인가?'],
            ['V','green',result.modelWeights?.v?.weight,vSelected,'실제로 가져갈 내용']
          ] as item}
            <div class="projection-card">
              <b class={item[1]}>{item[0]}</b>
              <small>{item[4]}</small>
              {#if item[2]}<UpstreamMatrixSvg data={item[2]} cellHeight={7} cellWidth={7} rowGap={1} colGap={1} shape="rect" colorScale={(v)=>paramColor(v)} showTooltip={(e,v)=>v.toFixed(3)}/>{/if}
              <span>× token →</span>
              <div class="projection-vector"><UpstreamVectorCanvas data={item[3]} colorScale={item[1]} active={true}/></div>
            </div>
          {/each}
        </div>
      {:else if expandedStage === 'attention'}
        <UpstreamAttentionExpansion
          scores={result.scores}
          masked={result.raw}
          weights={result.weights}
          {selectedRow}
          {selectedCol}
          onSelect={(r,c)=>{selectedRow=r;selectedCol=c;onSelectToken(c);}}
        />
      {:else if expandedStage === 'block'}
        <div class="block-detail">
          <div class="block-step"><b>Attention context</b><div class="wide-vector"><UpstreamVectorCanvas data={contextSelected} colorScale="purple" active={true}/></div><small>Σ attention·V</small></div>
          <span>→ Wₒ →</span>
          <div class="block-step"><b>Attention out</b><div class="wide-vector"><UpstreamVectorCanvas data={attendedSelected} colorScale="purple" active={true}/></div></div>
          <span>+ token →</span>
          <div class="block-step"><b>Residual 1</b><div class="wide-vector"><UpstreamVectorCanvas data={residual1Selected} colorScale="blue" active={true}/></div></div>
          <span>→ LN →</span>
          <div class="block-step"><b>Norm 2</b><div class="wide-vector"><UpstreamVectorCanvas data={norm2Selected} colorScale="gray" active={true}/></div></div>
          <span>→ Linear + GELU →</span>
          <div class="block-step"><b>FFN 16D</b><div class="wide-vector ffn"><UpstreamVectorCanvas data={mlpUpSelected} colorScale="purple" active={true}/></div></div>
          <span>→ Linear → + →</span>
          <div class="block-step"><b>Hidden 8D</b><div class="wide-vector final"><UpstreamVectorCanvas data={hiddenSelected} colorScale="blue" active={true}/></div></div>
        </div>
      {:else if expandedStage === 'action'}
        <div class="action-detail">
          <div class="block-step"><b>Final hidden · t</b><div class="wide-vector"><UpstreamVectorCanvas data={result.hidden?.[last] || hiddenSelected} colorScale="blue" active={true}/></div></div>
          <span>×</span>
          {#if result.modelWeights?.action}
            <div class="weight-peek"><span>action weight · 1×{result.modelWeights.action.weight[0].length}</span><UpstreamMatrixSvg data={result.modelWeights.action.weight} cellHeight={14} cellWidth={14} rowGap={1} colGap={2} shape="rect" colorScale={(v)=>paramColor(v)} showTooltip={(e,v)=>v.toFixed(3)}/></div>
          {/if}
          <span>→ score {result.actionScore.toFixed(3)} → tanh × 10 →</span>
          <div class="action-result">{controllerForce>=0?'RIGHT':'LEFT'} <b>{Math.abs(controllerForce).toFixed(2)} N</b></div>
        </div>
      {/if}
    </div>
  {/if}
</section>

<style>
.pipeline-shell{height:100%;min-height:450px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:11px;display:flex;flex-direction:column;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:0 2px 8px;border-bottom:1px solid #edf0f3;gap:10px}.panel-head>div{display:flex;flex-direction:column}.panel-head strong{font-size:12px}.panel-head small{font-size:7px;color:#9299a5}.panel-head>span{font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7257b5;white-space:nowrap}
.overview{position:relative;flex:1;display:grid;grid-template-columns:.85fr 1fr 1.3fr 1.2fr .65fr;gap:17px;align-items:center;padding:25px 4px 10px;min-height:365px;transition:opacity .2s}.pipeline-shell.has-detail .overview{min-height:240px;max-height:255px}.overview.dimmed .stage:not(.expanded){opacity:.38}
.stage{appearance:none;border:0;background:transparent;padding:0;position:relative;z-index:2;min-width:0;text-align:center;cursor:pointer;transition:opacity .18s,transform .18s}.stage:hover,.stage:focus-visible{transform:translateY(-2px)}.stage.expanded{opacity:1!important}.stage-head{display:grid;grid-template-columns:18px 1fr;column-gap:5px;align-items:center;margin-bottom:7px;text-align:left}.stage-head>b{grid-row:1/3;width:18px;height:18px;line-height:18px;border-radius:50%;background:#293548;color:#fff;font-size:8px;text-align:center}.stage-head>span{font-size:9px;font-weight:700;color:#4b5563}.stage-head>small{font-size:6px;color:#9299a5}
.token-column{display:flex;flex-direction:column;gap:3px}.token-row{display:flex;align-items:center;justify-content:center;gap:4px;border-radius:5px;padding:1px 3px}.token-row.selected{background:#f1eefb}.time-label{width:18px;text-align:right;font:7px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7b8492}.dim-label{font-size:6px;color:#a0a6b0}.vector{position:relative;width:18px;height:25px;border:1px solid #e0e3e8;border-radius:3px;overflow:hidden;background:white}.qkv-bundle{display:grid;grid-template-columns:repeat(3,15px);gap:2px}.vector.mini{width:15px;height:25px}.qkv-labels{display:grid;grid-template-columns:repeat(3,15px);gap:2px;justify-content:center;margin:0 0 3px 22px;font-size:7px;font-weight:700}.q{color:#3971e8}.k{color:#d95757}.v{color:#34966a}
.overview-matrix{display:flex;justify-content:center;padding:8px;border:1px solid #e5e7eb;background:#fbfbfd;border-radius:8px}.attention-read{display:grid;grid-template-columns:auto auto;gap:1px 4px;justify-content:center;margin-top:5px;font-size:6px;color:#8a93a2}.attention-read b{color:#5c6675}.attention-read strong{grid-column:1/3;color:#7359b7;font-size:9px}
.block-chain{display:grid;grid-template-columns:1fr 8px 1fr 8px 1fr 8px 1fr;gap:3px;align-items:center}.block-chain>span{font-size:8px;color:#8f96a1}.block-chain>div{display:flex;flex-direction:column;align-items:center;gap:2px}.block-chain small{font-size:5px;color:#8b93a1}.block-chain .vector{height:54px;width:17px}
.force-visual{font-size:42px;color:#8b5cf6;line-height:.85}.force-value{display:block;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6d52aa;margin-top:7px}.action-overview>small{font-size:6px;color:#9299a5}
.detail-panel{position:relative;z-index:4;border-top:1px solid #e3e7ed;background:#fcfcfe;padding:10px 12px 12px;border-radius:0 0 11px 11px;overflow-x:auto}.detail-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:10px}.detail-head>div{display:flex;flex-direction:column;gap:2px}.detail-head b{font-size:11px}.detail-head span{font-size:8px;color:#7b8492;line-height:1.4}.detail-head button{width:26px;height:26px;border:1px solid #dfe3e8;border-radius:7px;background:white;color:#667085;cursor:pointer}
.calc-line{display:flex;align-items:center;justify-content:center;gap:8px;min-width:760px}.calc-item{display:flex;flex-direction:column;align-items:center;gap:5px}.calc-item>b,.block-step>b{font-size:8px;color:#596273}.number-vector{display:flex;gap:3px}.number-vector span{min-width:34px;padding:5px 4px;border:1px solid #e2e5ea;border-radius:5px;background:white;font:7px ui-monospace,SFMono-Regular,Menlo,monospace}.number-vector small{display:block;font-size:6px;color:#9198a4}.wide-vector{position:relative;width:28px;height:82px;border:1px solid #dfe3e8;border-radius:4px;overflow:hidden;background:white}.wide-vector.final{outline:2px solid #dad3ee}.wide-vector.ffn{height:98px}.math-arrow{display:flex;flex-direction:column;align-items:center;color:#7d6bb2;font-size:14px}.math-arrow span{font-size:6px;color:#8d94a0}.math-symbol{font-size:16px;color:#7d6bb2}.weight-peek{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:9px}.weight-peek>span{font-size:7px;color:#7b8492}
.qkv-detail{display:grid;grid-template-columns:.8fr repeat(3,1fr);gap:10px;align-items:center;min-width:720px}.source-token{justify-self:center}.projection-card{border-left:1px solid #e3e6eb;padding-left:10px;display:grid;grid-template-columns:25px 1fr 25px;grid-template-rows:auto auto 1fr;gap:3px 6px;align-items:center}.projection-card>b{grid-row:1/3;font-size:16px}.projection-card>small{font-size:7px;color:#747d8b}.projection-card>span{font-size:6px;color:#9299a5}.projection-vector{position:relative;width:22px;height:70px;border:1px solid #dfe3e8;border-radius:4px;overflow:hidden;grid-row:1/4;grid-column:3}.block-detail{display:flex;align-items:center;justify-content:center;gap:8px;min-width:850px}.block-detail>span,.action-detail>span{font-size:7px;color:#8472b5;white-space:nowrap}.block-step{display:flex;flex-direction:column;align-items:center;gap:4px}.block-step>small{font-size:6px;color:#9299a5}.action-detail{display:flex;align-items:center;justify-content:center;gap:12px;min-width:620px}.action-result{padding:9px 12px;border-radius:8px;background:#f1eef9;color:#6c55a5;font-size:9px}.action-result b{font-size:12px}
@media(max-width:1220px){.overview{gap:10px;grid-template-columns:.82fr .95fr 1.25fr 1.15fr .62fr}.block-chain{gap:1px}}
@media(max-width:980px){.pipeline-shell{overflow-x:auto}.overview{min-width:780px}.detail-panel{min-width:780px}}
</style>
