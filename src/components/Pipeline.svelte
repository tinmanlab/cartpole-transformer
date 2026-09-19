<!--
Cart-Pole adapter around visualization primitives adapted from
poloclub/transformer-explainer @ bfe50afba10b9b560b84143ee1107d977defa74f
-->
<script>
  import * as d3 from 'd3';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  import UpstreamSankeyFlow from '../upstream/SankeyFlow.svelte';
  import { NORMALIZATION_SCALE } from '../lib/attention.js';

  export let history = [];
  export let result;
  export let controllerForce = 0;

  let selectedRow = 7;
  let selectedCol = 7;

  const names = ['x','ẋ','θ','θ̇'];
  const units = ['m','m/s','rad','rad/s'];

  $: last = result.rawTokens.length - 1;
  $: rawNow = result.rawTokens[last];
  $: normalizedNow = result.normalizedTokens[last];
  $: tokenNow = result.tokens[last];
  $: qNow = result.q[last];
  $: kNow = result.k[last];
  $: vNow = result.v[last];
  $: isLearned = result.modelType === 'learned-tiny-transformer';
  $: redrawKey = history.map(v=>v.join(',')).join('|') + controllerForce.toFixed(2) + result.modelType;

  $: scoreExtent = d3.extent(result.scores.flat().filter(Number.isFinite));
  $: scoreColor = d3.scaleLinear()
    .domain([scoreExtent[0] || -1, 0, scoreExtent[1] || 1])
    .range(['#eef2ff','#ffffff','#7c3aed'])
    .clamp(true);
  $: weightColor = d3.scaleSequential(d3.interpolatePurples)
    .domain([0, Math.max(.001,d3.max(result.weights.flat()) || 1)]);

  function hoverCell(event,d) {
    selectedRow = d.rowIndex;
    selectedCol = d.colIndex;
  }

  $: pathMap = {
    encode: [{
      from: '.stage-raw .anchor',
      to: '.stage-encode .anchor',
      type: 'stroke',
      gradientId: 'gray-blue',
      opacity: .9,
      curve: 45
    }],
    project: [{
      from: '.stage-encode .anchor',
      to: '.stage-qkv .anchor',
      type: 'stroke',
      gradientId: 'gray-blue',
      opacity: .9,
      curve: 45
    }],
    compare: [{
      from: '.stage-qkv .qk-anchor',
      to: '.stage-attention .score-anchor',
      type: 'stroke',
      gradientId: 'blue-mix',
      opacity: .9,
      curve: 45
    }],
    weighted: [{
      from: '.stage-attention .softmax-anchor',
      to: '.stage-weighted .weight-anchor',
      type: 'stroke',
      gradientId: 'blue-purple',
      opacity: .9,
      curve: 45
    }],
    value: [{
      from: '.stage-qkv .v-anchor',
      to: '.stage-weighted .value-anchor',
      type: 'stroke',
      gradientId: 'green-purple',
      opacity: .78,
      curve: 65
    }],
    action: [{
      from: '.stage-weighted .context-anchor',
      to: '.stage-force .anchor',
      type: 'stroke',
      gradientId: 'purple-purple',
      opacity: .95,
      curve: 45
    }]
  };
</script>

<section class="pipeline-shell resize-watch">
  <div class="panel-head">
    <strong>Transformer dataflow</strong>
    <span>{isLearned ? 'LEARNED' : 'TOY FALLBACK'} · {controllerForce >= 0 ? '→' : '←'} {Math.abs(controllerForce).toFixed(1)} N</span>
  </div>

  <div class="flow-area resize-watch">
    <UpstreamSankeyFlow {pathMap} {redrawKey}/>

    <section class="stage stage-raw">
      <div class="step-no">1</div>
      <h3>Raw state</h3>
      <div class="anchor node-box raw-values">
        {#each rawNow as value,i}
          <div><b>{names[i]}</b><strong>{value.toFixed(i<2?2:3)}</strong><small>{units[i]}</small></div>
        {/each}
      </div>
      <p>from simulation</p>
    </section>

    <section class="stage stage-encode">
      <div class="step-no">2</div>
      <h3>Encode</h3>
      <div class="anchor node-box encoder-box">
        <div class="encode-step">
          <b>normalize</b>
          <div class="equation">÷ [{NORMALIZATION_SCALE.join(', ')}]</div>
          <div class="vector compact"><UpstreamVectorCanvas data={normalizedNow} colorScale="gray" active={true}/></div>
        </div>
        <div class="internal-arrow">→</div>
        <div class="encode-step">
          <b>{isLearned ? 'Linear 4→8 + pos' : 'identity'}</b>
          <div class="vector encoded"><UpstreamVectorCanvas data={tokenNow} colorScale={isLearned ? 'blue' : 'gray'} active={true}/></div>
          <small>{tokenNow.length}D token</small>
        </div>
      </div>
      <p>{isLearned ? 'learned state embedding + position' : 'scale-only fallback'}</p>
    </section>

    <section class="stage stage-qkv">
      <div class="step-no">3</div>
      <h3>Q · K · V</h3>
      <div class="anchor node-box qkv-box">
        <div class="qk-anchor qkv-item"><b class="q">Q</b><div class="vector tall"><UpstreamVectorCanvas data={qNow} colorScale="blue" active={true}/></div><small>WQ·token</small></div>
        <div class="qk-anchor qkv-item"><b class="k">K</b><div class="vector tall"><UpstreamVectorCanvas data={kNow} colorScale="red" active={true}/></div><small>WK·token</small></div>
        <div class="v-anchor qkv-item"><b class="v">V</b><div class="vector tall"><UpstreamVectorCanvas data={vNow} colorScale="green" active={true}/></div><small>WV·token</small></div>
      </div>
      <p>applied to all 8 time tokens</p>
    </section>

    <section class="stage stage-attention">
      <div class="step-no">4</div>
      <h3>Attention</h3>
      <div class="node-box attention-chain">
        <div class="score-anchor mini-matrix">
          <b>QKᵀ / √d</b>
          <UpstreamMatrixSvg
            data={result.scores}
            cellHeight={8}
            cellWidth={8}
            rowGap={1.5}
            colGap={1.5}
            shape="circle"
            colorScale={(v)=>scoreColor(v)}
            highlightRow={selectedRow}
            highlightCol={selectedCol}
            onMouseOverCell={hoverCell}
            showTooltip={(e,v)=>Number.isFinite(v)?v.toFixed(2):'—'}
          />
        </div>
        <div class="internal-arrow">→</div>
        <div class="mini-matrix">
          <b>causal mask</b>
          <UpstreamMatrixSvg
            data={result.raw}
            cellHeight={8}
            cellWidth={8}
            rowGap={1.5}
            colGap={1.5}
            shape="rect"
            colorScale={(v)=>scoreColor(v)}
            highlightRow={selectedRow}
            highlightCol={selectedCol}
            onMouseOverCell={hoverCell}
            showTooltip={(e,v)=>Number.isFinite(v)?v.toFixed(2):'masked'}
          />
        </div>
        <div class="internal-arrow">→</div>
        <div class="softmax-anchor mini-matrix">
          <b>softmax</b>
          <UpstreamMatrixSvg
            data={result.weights}
            cellHeight={8}
            cellWidth={8}
            rowGap={1.5}
            colGap={1.5}
            shape="rect"
            colorScale={(v)=>weightColor(v)}
            highlightRow={selectedRow}
            highlightCol={selectedCol}
            onMouseOverCell={hoverCell}
            showTooltip={(e,v)=>(v*100).toFixed(1)+'%'}
          />
        </div>
      </div>
      <p>score → mask → attention weights</p>
    </section>

    <section class="stage stage-weighted">
      <div class="step-no">5</div>
      <h3>Weighted V</h3>
      <div class="node-box weighted-box">
        <div class="weight-anchor weight-chip"><b>a</b><small>softmax weights</small></div>
        <div class="merge">×</div>
        <div class="value-anchor value-source"><b>V</b><div class="vector context-source"><UpstreamVectorCanvas data={vNow} colorScale="green" active={true}/></div></div>
        <div class="merge">→</div>
        <div class="context-anchor context"><b>Σ(a·V)</b><div class="vector context-out"><UpstreamVectorCanvas data={result.context} colorScale="purple" active={true}/></div></div>
      </div>
      <p>attention chooses how much V survives</p>
    </section>

    <section class="stage stage-force">
      <div class="step-no">6</div>
      <h3>Force</h3>
      <div class="anchor node-box force-box">
        <div class="force-arrow">{controllerForce>=0?'→':'←'}</div>
        <strong>{Math.abs(controllerForce).toFixed(1)} N</strong>
      </div>
      <p>back to physics</p>
    </section>
  </div>
</section>

<style>
.pipeline-shell{height:100%;min-height:430px;background:#fff;border:1px solid #e2e5ea;border-radius:16px;padding:12px;display:flex;flex-direction:column;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:0 2px 8px;border-bottom:1px solid #edf0f3}
.panel-head strong{font-size:13px}.panel-head span{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7257b5}
.flow-area{position:relative;flex:1;display:grid;grid-template-columns:.72fr 1.15fr .78fr 2.08fr 1.12fr .58fr;gap:17px;align-items:center;padding:29px 4px 10px;min-height:365px}
.stage{position:relative;z-index:2;min-width:0;text-align:center;align-self:center}.stage h3{font-size:10px;margin:0 0 7px;color:#5f6877;white-space:nowrap}.step-no{position:absolute;top:-22px;left:50%;transform:translateX(-50%);width:18px;height:18px;line-height:18px;border-radius:50%;background:#283548;color:white;font-size:9px;font-weight:700}.stage p{font-size:7px;color:#8a93a2;line-height:1.3;margin:5px -5px 0}.node-box{position:relative;background:#fbfcfd;border:1px solid #e3e7ed;border-radius:9px;padding:6px;min-height:106px;display:flex;align-items:center;justify-content:center}.raw-values{display:grid;gap:4px;align-content:center}.raw-values div{display:grid;grid-template-columns:18px 1fr 25px;gap:3px;align-items:baseline;text-align:left}.raw-values b{font-size:8px}.raw-values strong{font:8px ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right}.raw-values small{font-size:6px;color:#9299a5}
.encoder-box{display:grid;grid-template-columns:1fr 12px 1fr;gap:4px}.encode-step{display:flex;flex-direction:column;align-items:center;gap:4px;min-width:0}.encode-step>b{font-size:7px;color:#667085;white-space:nowrap}.encode-step small{font-size:6px;color:#8b93a1}.equation{font:5.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7a8491;white-space:nowrap}.internal-arrow{align-self:center;font-size:13px;color:#8b5cf6;font-weight:700}.vector{position:relative;width:22px;height:54px;border:1px solid #d9dde5;border-radius:4px;overflow:hidden}.vector.compact{height:45px}.vector.encoded{height:64px}.vector.tall{width:18px;height:68px}.qkv-box{display:flex;gap:6px}.qkv-item{display:flex;flex-direction:column;align-items:center;gap:3px}.qkv-item>b{font-size:9px}.qkv-item small{font-size:5.5px;color:#8b93a1}.q{color:#3971e8}.k{color:#d95757}.v{color:#34966a}
.attention-chain{display:grid;grid-template-columns:1fr 12px 1fr 12px 1fr;gap:3px;min-height:120px}.mini-matrix{display:flex;flex-direction:column;align-items:center;gap:4px;min-width:0}.mini-matrix>b{font-size:6px;color:#687283;white-space:nowrap}.weighted-box{display:grid;grid-template-columns:30px 9px 23px 9px 28px;gap:3px;align-items:center;min-height:112px}.weight-chip{display:flex;flex-direction:column;gap:3px;align-items:center}.weight-chip>b{font-size:16px;color:#7c3aed}.weight-chip small{font-size:5px;color:#8b93a1}.merge{font-size:10px;color:#8b93a1}.value-source,.context{display:flex;flex-direction:column;align-items:center;gap:4px}.value-source>b,.context>b{font-size:7px;color:#687283}.context-source{width:18px;height:54px}.context-out{width:22px;height:62px}.force-box{flex-direction:column;gap:7px}.force-arrow{font-size:38px;color:#8b5cf6;line-height:.8}.force-box strong{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6d52aa}
@media(max-width:1220px){.flow-area{gap:11px;grid-template-columns:.68fr 1.08fr .75fr 2fr 1.06fr .56fr}.node-box{padding:4px}}
@media(max-width:980px){.pipeline-shell{overflow-x:auto}.flow-area{min-width:810px}}
</style>
