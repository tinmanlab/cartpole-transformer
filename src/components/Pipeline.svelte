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
  $: tokenNow = result.tokens[last];
  $: qNow = result.q[last];
  $: kNow = result.k[last];
  $: vNow = result.v[last];
  $: redrawKey = history.map(v=>v.join(',')).join('|') + controllerForce.toFixed(2);

  $: scoreExtent = d3.extent(result.raw.flat().filter(Number.isFinite));
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
      to: '.stage-score .anchor',
      type: 'stroke',
      gradientId: 'blue-mix',
      opacity: .9,
      curve: 45
    }],
    softmax: [{
      from: '.stage-score .anchor',
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
      opacity: .8,
      curve: 60
    }],
    action: [{
      from: '.stage-weighted .anchor',
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
    <span>{controllerForce >= 0 ? '→' : '←'} {Math.abs(controllerForce).toFixed(1)} N</span>
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
      <p>simulation output</p>
    </section>

    <section class="stage stage-encode">
      <div class="step-no">2</div>
      <h3>Normalize</h3>
      <div class="anchor node-box encoded-box">
        <div class="equation">÷ [{NORMALIZATION_SCALE.join(', ')}]</div>
        <div class="vector"><UpstreamVectorCanvas data={tokenNow} colorScale="gray" active={true}/></div>
        <code>[{tokenNow.map(v=>v.toFixed(2)).join(', ')}]</code>
      </div>
      <p>this is the current “encoder”</p>
    </section>

    <section class="stage stage-qkv">
      <div class="step-no">3</div>
      <h3>Q · K · V</h3>
      <div class="anchor node-box qkv-box">
        <div class="qk-anchor qkv-item"><b class="q">Q</b><div class="vector tall"><UpstreamVectorCanvas data={qNow} colorScale="blue" active={true}/></div><small>WQ·token</small></div>
        <div class="qk-anchor qkv-item"><b class="k">K</b><div class="vector tall"><UpstreamVectorCanvas data={kNow} colorScale="red" active={true}/></div><small>WK·token</small></div>
        <div class="v-anchor qkv-item"><b class="v">V</b><div class="vector tall"><UpstreamVectorCanvas data={vNow} colorScale="green" active={true}/></div><small>WV·token</small></div>
      </div>
      <p>same for all 8 time tokens</p>
    </section>

    <section class="stage stage-score">
      <div class="step-no">4</div>
      <h3>Compare</h3>
      <div class="anchor node-box matrix-box">
        <b>Q·Kᵀ / √d + mask</b>
        <UpstreamMatrixSvg
          data={result.raw}
          cellHeight={11}
          cellWidth={11}
          rowGap={2}
          colGap={2}
          shape="circle"
          colorScale={(v)=>scoreColor(v)}
          highlightRow={selectedRow}
          highlightCol={selectedCol}
          onMouseOverCell={hoverCell}
          showTooltip={(e,v)=>Number.isFinite(v)?v.toFixed(2):'masked'}
        />
      </div>
      <p>circle size grid = who matches whom</p>
    </section>

    <section class="stage stage-weighted">
      <div class="step-no">5</div>
      <h3>Softmax + V</h3>
      <div class="node-box weighted-box">
        <div class="weight-anchor weights">
          <b>attention weights</b>
          <UpstreamMatrixSvg
            data={result.weights}
            cellHeight={11}
            cellWidth={11}
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
        <div class="merge">×</div>
        <div class="value-anchor value-source"><b>V</b><div class="vector context-source"><UpstreamVectorCanvas data={vNow} colorScale="green" active={true}/></div></div>
        <div class="merge">→</div>
        <div class="anchor context"><b>Σ(a·V)</b><div class="vector context-out"><UpstreamVectorCanvas data={result.context} colorScale="purple" active={true}/></div></div>
      </div>
      <p>weights choose how much V to carry forward</p>
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
.panel-head strong{font-size:13px}.panel-head span{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7257b5}
.flow-area{position:relative;flex:1;display:grid;grid-template-columns:.78fr .92fr 1fr 1.2fr 1.55fr .72fr;gap:19px;align-items:center;padding:28px 5px 12px;min-height:365px}
.stage{position:relative;z-index:2;min-width:0;text-align:center;align-self:center}.stage h3{font-size:10px;margin:0 0 7px;color:#5f6877;white-space:nowrap}.step-no{position:absolute;top:-22px;left:50%;transform:translateX(-50%);width:18px;height:18px;line-height:18px;border-radius:50%;background:#283548;color:white;font-size:9px;font-weight:700}.stage p{font-size:8px;color:#8a93a2;line-height:1.3;margin:6px -4px 0}.node-box{position:relative;background:#fbfcfd;border:1px solid #e3e7ed;border-radius:9px;padding:7px;min-height:106px;display:flex;align-items:center;justify-content:center}.raw-values{display:grid;gap:4px;align-content:center}.raw-values div{display:grid;grid-template-columns:18px 1fr 27px;gap:3px;align-items:baseline;text-align:left}.raw-values b{font-size:9px}.raw-values strong{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right}.raw-values small{font-size:7px;color:#9299a5}
.encoded-box{flex-direction:column;gap:6px}.equation{font:7px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7a8491;white-space:nowrap}.encoded-box code{font-size:7px;color:#667085;white-space:nowrap}.vector{position:relative;width:24px;height:56px;border:1px solid #d9dde5;border-radius:4px;overflow:hidden}.vector.tall{width:20px;height:68px}.qkv-box{display:flex;gap:7px}.qkv-item{display:flex;flex-direction:column;align-items:center;gap:3px}.qkv-item>b{font-size:10px}.qkv-item small{font-size:6px;color:#8b93a1}.q{color:#3971e8}.k{color:#d95757}.v{color:#34966a}
.matrix-box{flex-direction:column;gap:5px}.matrix-box>b,.weights>b,.value-source>b,.context>b{font-size:7px;color:#687283}.weighted-box{display:grid;grid-template-columns:auto 10px 24px 10px 28px;gap:4px;align-items:center;min-height:116px}.weights{display:flex;flex-direction:column;gap:4px}.merge{font-size:11px;color:#8b93a1}.value-source,.context{display:flex;flex-direction:column;align-items:center;gap:4px}.context-source{width:18px;height:54px}.context-out{width:22px;height:62px}.force-box{flex-direction:column;gap:7px}.force-arrow{font-size:42px;color:#8b5cf6;line-height:.8}.force-box strong{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6d52aa}
@media(max-width:1180px){.flow-area{gap:12px;grid-template-columns:.76fr .86fr .95fr 1.08fr 1.45fr .7fr}.node-box{padding:5px}}
@media(max-width:920px){.pipeline-shell{overflow-x:auto}.flow-area{min-width:720px}}
</style>
