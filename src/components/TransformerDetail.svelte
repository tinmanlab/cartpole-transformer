<!--
Full-width Transformer detail drawer.
Uses visualization primitives adapted from poloclub/transformer-explainer
@ bfe50afba10b9b560b84143ee1107d977defa74f.
-->
<script>
  import * as d3 from 'd3';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  import UpstreamAttentionExpansion from '../upstream/AttentionExpansion.svelte';
  import AttentionCellTrace from './AttentionCellTrace.svelte';

  export let result;
  export let controllerForce = 0;
  export let selectedToken = 7;
  export let selectedRow = 7;
  export let selectedCol = 7;
  export let expandedStage = null;
  export let onClose = () => {};
  export let onSelectToken = () => {};
  export let onSelectAttention = () => {};

  const names = ['x','ẋ','θ','θ̇'];
  const stageInfo = {
    embedding: '실제 simulator state를 normalize한 뒤 learned 4→8 projection과 position vector를 더합니다.',
    qkv: 'LayerNorm 뒤 동일한 token을 실제 learned WQ, WK, WV에 통과시켜 세 역할로 분리합니다.',
    attention: '실제 Q·K score에 causal mask와 softmax를 적용하고, 그 weight로 실제 V를 섞습니다.',
    block: '실제 attention output projection, residual, LayerNorm, GELU MLP, 두 번째 residual을 순서대로 보여줍니다.',
    action: '실제 final hidden(t)을 learned action head가 읽고 tanh를 거쳐 ±10 N force를 냅니다.'
  };

  const paramColor = d3.scaleDiverging()
    .domain([-1,0,1])
    .range(['#ef9a9a','#ffffff','#93b4ee'])
    .clamp(true);

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

  function liveText(values, count=4) {
    return values.slice(0,count).map(v=>Number(v).toFixed(3)).join(' ');
  }
</script>

{#if expandedStage}
<section class="transformer-detail-wide">
  <header class="detail-head">
    <div>
      <div class="eyebrow">LIVE TENSOR DETAIL · {selectedToken===last?'t':'t−'+(last-selectedToken)}</div>
      <h2>{expandedStage === 'embedding' ? 'Embedding' : expandedStage === 'qkv' ? 'Q · K · V' : expandedStage === 'attention' ? 'Self Attention' : expandedStage === 'block' ? 'Residual + MLP' : 'Action head'}</h2>
      <p>{stageInfo[expandedStage]}</p>
    </div>
    <button type="button" on:click={onClose} aria-label="close Transformer detail">×</button>
  </header>

  {#if expandedStage === 'embedding'}
    <div class="calc-line embedding-detail">
      <div class="calc-item raw">
        <b>Raw state</b>
        <div class="number-vector">{#each rawSelected as value,i}<span><small>{names[i]}</small>{value.toFixed(i<2?2:3)}</span>{/each}</div>
      </div>
      <div class="math-arrow"><span>normalize</span>→</div>
      <div class="calc-item">
        <b>Normalized 4D</b>
        <div class="number-vector">{#each normalizedSelected as value}<span>{value.toFixed(3)}</span>{/each}</div>
      </div>
      <div class="math-arrow"><span>Linear 4→8</span>→</div>
      <div class="calc-item">
        <b>State embedding</b>
        <div class="wide-vector"><UpstreamVectorCanvas data={stateEmbeddingSelected} colorScale="blue" active={true}/></div>
        <code>{liveText(stateEmbeddingSelected)}</code>
      </div>
      <div class="math-symbol">+</div>
      <div class="calc-item">
        <b>Position {selectedToken}</b>
        <div class="wide-vector"><UpstreamVectorCanvas data={posEmbeddingSelected} colorScale="red" active={true}/></div>
        <code>{liveText(posEmbeddingSelected)}</code>
      </div>
      <div class="math-symbol">=</div>
      <div class="calc-item">
        <b>Token {tokenSelected.length}D</b>
        <div class="wide-vector final"><UpstreamVectorCanvas data={tokenSelected} colorScale="gray" active={true}/></div>
        <code>{liveText(tokenSelected)}</code>
      </div>
    </div>
    {#if isLearned && result.modelWeights?.embed}
      <div class="weight-peek"><span>learned embedding weight · 8×4</span><UpstreamMatrixSvg data={result.modelWeights.embed.weight} cellHeight={13} cellWidth={13} rowGap={2} colGap={2} shape="rect" colorScale={(v)=>paramColor(v)} showTooltip={(e,v)=>v.toFixed(3)}/></div>
    {/if}
  {:else if expandedStage === 'qkv'}
    <div class="qkv-detail">
      <div class="calc-item source-token">
        <b>LayerNorm(token)</b>
        <div class="wide-vector"><UpstreamVectorCanvas data={norm1Selected} colorScale="gray" active={true}/></div>
        <code>{liveText(norm1Selected)}</code>
      </div>
      {#each [
        ['Q','blue',result.modelWeights?.q?.weight,qSelected,'무엇을 찾을까?'],
        ['K','red',result.modelWeights?.k?.weight,kSelected,'나는 어떤 정보인가?'],
        ['V','green',result.modelWeights?.v?.weight,vSelected,'실제로 가져갈 내용']
      ] as item}
        <article class="projection-card">
          <div class="projection-title"><b class={item[1]}>{item[0]}</b><span>{item[4]}</span></div>
          {#if item[2]}<UpstreamMatrixSvg data={item[2]} cellHeight={11} cellWidth={11} rowGap={1.5} colGap={1.5} shape="rect" colorScale={(v)=>paramColor(v)} showTooltip={(e,v)=>v.toFixed(3)}/>{/if}
          <div class="projection-op">× LN(token) →</div>
          <div class="projection-vector"><UpstreamVectorCanvas data={item[3]} colorScale={item[1]} active={true}/></div>
          <code>{liveText(item[3])}</code>
        </article>
      {/each}
    </div>
  {:else if expandedStage === 'attention'}
    <div class="attention-wide">
      <UpstreamAttentionExpansion
        scores={result.scores}
        masked={result.raw}
        weights={result.weights}
        {selectedRow}
        {selectedCol}
        onSelect={(r,c)=>onSelectAttention(r,c)}
      />
      <AttentionCellTrace
        {result}
        {selectedRow}
        {selectedCol}
        onSelect={(r,c)=>onSelectAttention(r,c)}
      />
    </div>
  {:else if expandedStage === 'block'}
    <div class="block-detail">
      <div class="block-step"><b>Context</b><div class="wide-vector"><UpstreamVectorCanvas data={contextSelected} colorScale="purple" active={true}/></div><small>Σ attention·V</small><code>{liveText(contextSelected)}</code></div>
      <span>→ Wₒ →</span>
      <div class="block-step"><b>Attention out</b><div class="wide-vector"><UpstreamVectorCanvas data={attendedSelected} colorScale="purple" active={true}/></div><code>{liveText(attendedSelected)}</code></div>
      <span>+ token →</span>
      <div class="block-step"><b>Residual 1</b><div class="wide-vector"><UpstreamVectorCanvas data={residual1Selected} colorScale="blue" active={true}/></div><code>{liveText(residual1Selected)}</code></div>
      <span>→ LN →</span>
      <div class="block-step"><b>Norm 2</b><div class="wide-vector"><UpstreamVectorCanvas data={norm2Selected} colorScale="gray" active={true}/></div><code>{liveText(norm2Selected)}</code></div>
      <span>→ Linear + GELU →</span>
      <div class="block-step"><b>FFN 16D</b><div class="wide-vector ffn"><UpstreamVectorCanvas data={mlpUpSelected} colorScale="purple" active={true}/></div><code>{liveText(mlpUpSelected)}</code></div>
      <span>→ Linear + residual →</span>
      <div class="block-step"><b>Hidden 8D</b><div class="wide-vector final"><UpstreamVectorCanvas data={hiddenSelected} colorScale="blue" active={true}/></div><code>{liveText(hiddenSelected)}</code></div>
    </div>
  {:else if expandedStage === 'action'}
    <div class="action-detail">
      <div class="block-step"><b>Final hidden · t</b><div class="wide-vector"><UpstreamVectorCanvas data={result.hidden?.[last] || hiddenSelected} colorScale="blue" active={true}/></div><code>{liveText(result.hidden?.[last] || hiddenSelected)}</code></div>
      <span>×</span>
      {#if result.modelWeights?.action}
        <div class="weight-peek"><span>action weight · 1×{result.modelWeights.action.weight[0].length}</span><UpstreamMatrixSvg data={result.modelWeights.action.weight} cellHeight={18} cellWidth={18} rowGap={1} colGap={2} shape="rect" colorScale={(v)=>paramColor(v)} showTooltip={(e,v)=>v.toFixed(3)}/></div>
      {/if}
      <span>→ score <code>{result.actionScore.toFixed(6)}</code> → tanh × 10 →</span>
      <div class="action-result">{controllerForce>=0?'RIGHT':'LEFT'} <b>{Math.abs(controllerForce).toFixed(2)} N</b></div>
    </div>
  {/if}
</section>
{/if}

<style>
.transformer-detail-wide{margin-top:10px;background:#fff;border:1px solid #e1e5ea;border-radius:16px;padding:18px 22px 22px;overflow:hidden}
.detail-head{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;padding-bottom:14px;border-bottom:1px solid #edf0f3;margin-bottom:18px}.detail-head>div{max-width:960px}.eyebrow{font-size:9px;letter-spacing:.08em;color:#8978b3}.detail-head h2{font-size:20px;margin:2px 0 5px}.detail-head p{font-size:12px;color:#667085;line-height:1.55;margin:0}.detail-head button{width:36px;height:36px;border:1px solid #dfe3e8;border-radius:9px;background:#fff;color:#667085;font-size:20px;cursor:pointer}
.calc-line{display:flex;align-items:center;justify-content:center;gap:16px;min-width:900px;padding:8px 0}.calc-item{display:flex;flex-direction:column;align-items:center;gap:8px}.calc-item>b,.block-step>b{font-size:11px;color:#596273}.calc-item>code,.block-step>code,.projection-card>code{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085;white-space:nowrap}.number-vector{display:flex;gap:5px}.number-vector span{min-width:48px;padding:8px 7px;border:1px solid #e2e5ea;border-radius:7px;background:#fbfcfd;font:10px ui-monospace,SFMono-Regular,Menlo,monospace}.number-vector small{display:block;font:9px Inter,ui-sans-serif,system-ui;color:#8e96a2}.wide-vector{position:relative;width:38px;height:112px;border:1px solid #dfe3e8;border-radius:5px;overflow:hidden;background:white}.wide-vector.final{outline:2px solid #d8d0ef}.wide-vector.ffn{height:138px}.math-arrow{display:flex;flex-direction:column;align-items:center;color:#7965ad;font-size:22px}.math-arrow span{font-size:10px;color:#78818f}.math-symbol{font-size:22px;color:#7965ad}.weight-peek{display:flex;align-items:center;justify-content:center;gap:12px}.weight-peek>span{font-size:11px;color:#687386}
.qkv-detail{display:grid;grid-template-columns:.72fr repeat(3,1fr);gap:20px;align-items:center}.source-token{justify-self:center}.projection-card{min-width:0;border-left:1px solid #e4e7ec;padding-left:18px;display:grid;grid-template-columns:auto 1fr;grid-template-rows:auto auto auto;gap:10px;align-items:center}.projection-title{display:flex;align-items:baseline;gap:8px;grid-column:1/3}.projection-title b{font-size:24px}.projection-title span{font-size:11px;color:#667085}.projection-op{font-size:10px;color:#78818f}.projection-vector{position:relative;width:34px;height:110px;border:1px solid #dfe3e8;border-radius:5px;overflow:hidden;grid-row:2/4;grid-column:2}.q{color:#3971e8}.red{color:#d95757}.green{color:#34966a}.blue{color:#3971e8}
.attention-wide{padding:10px 2vw 14px;overflow-x:auto}.block-detail{display:flex;align-items:center;justify-content:center;gap:14px;min-width:1080px;padding:8px 0}.block-detail>span,.action-detail>span{font-size:11px;color:#7662ac;white-space:nowrap}.block-step{display:flex;flex-direction:column;align-items:center;gap:7px}.block-step>small{font-size:10px;color:#7d8593}.action-detail{display:flex;align-items:center;justify-content:center;gap:18px;min-width:760px;padding:18px 0}.action-result{padding:13px 17px;border-radius:9px;background:#f1eef9;color:#6c55a5;font-size:12px}.action-result b{font-size:17px}
@media(max-width:980px){.transformer-detail-wide{padding:15px 14px}.qkv-detail{grid-template-columns:1fr 1fr;min-width:720px}.source-token{grid-row:1/3}.attention-wide{overflow-x:visible;padding-left:0;padding-right:0}.block-detail,.action-detail,.calc-line{overflow-x:auto}.detail-head h2{font-size:18px}}
</style>
