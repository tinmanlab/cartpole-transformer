<!--
Cart-Pole-specific arithmetic trace built from live tensors.
Rendering primitives are adapted from Polo Club Transformer Explainer.
-->
<script>
  import { tick } from 'svelte';
  import { gsap } from 'gsap';
  import * as d3 from 'd3';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';

  export let result;
  export let selectedRow = 0;
  export let selectedCol = 0;
  export let onSelect = () => {};
  export let source = 'live';
  export let lockQuery = false;
  export let highlightDim = -1;
  export let onSelectDim = () => {};

  let root;
  let lastKey = '';

  $: n = result.q.length;
  $: last = n - 1;
  $: q = result.q[selectedRow];
  $: k = result.k[selectedCol];
  $: v = result.v[selectedCol];
  $: products = result.qkProducts[selectedRow][selectedCol];
  $: dotSum = products.reduce((sum, x) => sum + x, 0);
  $: scale = Math.sqrt(q.length);
  $: score = result.scores[selectedRow][selectedCol];
  $: hasFixedBias = Array.isArray(result.scoreBias);
  $: preBiasScore = hasFixedBias ? result.preBiasScores[selectedRow][selectedCol] : score;
  $: biasValue = hasFixedBias ? result.scoreBias[selectedRow][selectedCol] : 0;
  $: maskedScore = result.raw[selectedRow][selectedCol];
  $: masked = !Number.isFinite(maskedScore);
  $: rowMax = result.softmaxMax[selectedRow];
  $: shifted = result.softmaxShifted[selectedRow][selectedCol];
  $: expValue = result.softmaxExp[selectedRow][selectedCol];
  $: denominator = result.softmaxDenominators[selectedRow];
  $: weight = result.weights[selectedRow][selectedCol];
  $: contribution = result.weightedValueContributions[selectedRow][selectedCol];
  $: context = result.perTokenContext[selectedRow];
  $: traceKey = selectedRow + '|' + selectedCol;

  const productExtent = () => {
    const max = Math.max(1e-6, ...products.map(Math.abs));
    return d3.scaleDiverging().domain([-max,0,max]).range(['#f2a3a3','#ffffff','#9bb8ef']).clamp(true);
  };

  function label(i) {
    return i === last ? 't' : 't−' + (last - i);
  }

  async function animateTrace() {
    if (!root || traceKey === lastKey) return;
    lastKey = traceKey;
    await tick();
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return;
    gsap.fromTo(
      root.querySelectorAll('.trace-stage'),
      { opacity: .28, y: 5 },
      { opacity: 1, y: 0, duration: .24, stagger: .055, ease: 'power2.out' }
    );
  }

  $: if (root) { traceKey; animateTrace(); }
</script>

<section
  class="attention-cell-trace"
  bind:this={root}
  data-row={selectedRow}
  data-col={selectedCol}
  data-dot-sum={dotSum}
  data-scale={scale}
  data-score={score}
  data-masked={masked}
  data-exp={expValue}
  data-denominator={denominator}
  data-weight={weight}
  data-v0={v[0]}
  data-contribution0={contribution[0]}
  data-source={source}
  data-has-fixed-bias={hasFixedBias}
  data-pre-bias-score={preBiasScore}
  data-bias-value={biasValue}
  data-lock-query={lockQuery}
  data-highlight-dim={highlightDim}
>
  <header class="trace-head">
    <div>
      <div class="eyebrow">SELECTED ATTENTION CELL · ACTUAL {source === 'frozen' ? 'FROZEN' : 'LIVE'} ARITHMETIC{hasFixedBias ? ' · TOY FALLBACK (fixed weights, not learned)' : ''}</div>
      <h3>Q {label(selectedRow)} × K {label(selectedCol)}</h3>
      <p>matrix 한 칸이 어떤 계산을 뜻하는지 이 경로만 따라가면 됩니다.</p>
    </div>
    <div class:masked class="weight-badge">
      <small>{masked ? 'CAUSALLY MASKED' : 'ATTENTION WEIGHT'}</small>
      <strong>{masked ? '0%' : (weight*100).toFixed(2)+'%'}</strong>
    </div>
  </header>

  <div class="selector-row">
    <span>Query{lockQuery ? ' (locked)' : ''}</span>
    <div class="selector-buttons">
      {#each Array(n) as _,i}
        <button type="button" class="trace-query-button" data-index={i} class:active={i===selectedRow} disabled={lockQuery} on:click={() => onSelect(i, selectedCol)}>{label(i)}</button>
      {/each}
    </div>
    <span>Key</span>
    <div class="selector-buttons">
      {#each Array(n) as _,i}
        <button type="button" class="trace-key-button" data-index={i} class:active={i===selectedCol} class:future={i>selectedRow} on:click={() => onSelect(selectedRow, i)}>{label(i)}</button>
      {/each}
    </div>
  </div>

  <div class="trace-flow">
    <article class="trace-stage qk-stage">
      <div class="step">A</div>
      <h4>Q와 K 원소별 곱</h4>
      <div class="qk-vectors">
        <div><b>Q {label(selectedRow)}</b><div class="trace-vector"><UpstreamVectorCanvas data={q} colorScale="blue" active={true}/></div></div>
        <span>×</span>
        <div><b>K {label(selectedCol)}</b><div class="trace-vector"><UpstreamVectorCanvas data={k} colorScale="red" active={true}/></div></div>
      </div>
      <div class="product-grid">
        {#each products as product,i}
          <button type="button" class="product-dim" class:selected={i===highlightDim} data-dim={i} on:click={() => onSelectDim(i)}>
            <small>d{i}</small>
            <code>{q[i].toFixed(3)} × {k[i].toFixed(3)}</code>
            <b>{product.toFixed(3)}</b>
          </button>
        {/each}
      </div>
      <div class="product-matrix">
        <UpstreamMatrixSvg
          data={[products]}
          cellHeight={22}
          cellWidth={22}
          rowGap={1}
          colGap={3}
          shape="rect"
          colorScale={(v)=>productExtent()(v)}
          showTooltip={(e,v)=>v.toFixed(4)}
        />
      </div>
    </article>

    <div class="trace-arrow">→</div>

    <article class="trace-stage score-stage">
      <div class="step">B</div>
      <h4>합하고 √d로 나눔{hasFixedBias ? ' + 고정 recency prior' : ''}</h4>
      <div class="equation">
        <span>Σ(QᵢKᵢ)</span>
        <strong>{dotSum.toFixed(5)}</strong>
        <span>÷ √{q.length}</span>
        <strong>{scale.toFixed(5)}</strong>
        <span>= {hasFixedBias ? 'pre-bias score' : 'score'}</span>
        <strong>{preBiasScore.toFixed(5)}</strong>
        {#if hasFixedBias}
          <span>+ 고정 bias ({result.recencyBiasPerStep.toFixed(2)} × key index {selectedCol})</span>
          <strong>{biasValue.toFixed(5)}</strong>
          <span>= score</span>
          <strong>{score.toFixed(5)}</strong>
        {/if}
      </div>
      {#if hasFixedBias}
        <p>이 toy fallback은 학습된 attention이 아닙니다 — dot-product score에 학습되지 않은 고정 recency bias(나중 key일수록 커짐)를 더한 값입니다.</p>
      {:else}
        <p>이 값이 Q·Kᵀ matrix의 선택한 한 칸입니다.</p>
      {/if}
    </article>

    <div class="trace-arrow">→</div>

    <article class:masked class="trace-stage mask-stage">
      <div class="step">C</div>
      <h4>Causal mask</h4>
      {#if masked}
        <div class="masked-block">
          <strong>{label(selectedCol)} is future of {label(selectedRow)}</strong>
          <code>{score.toFixed(5)} → −∞</code>
          <span>따라서 softmax weight = 0</span>
        </div>
      {:else}
        <div class="pass-block">
          <strong>허용된 과거/현재 token</strong>
          <code>{score.toFixed(5)} 그대로 통과</code>
        </div>
      {/if}
    </article>

    <div class="trace-arrow">→</div>

    <article class="trace-stage softmax-stage">
      <div class="step">D</div>
      <h4>Stable softmax</h4>
      <div class="equation compact">
        <span>row max</span><strong>{rowMax.toFixed(5)}</strong>
        <span>score − max</span><strong>{shifted.toFixed(5)}</strong>
        <span>exp(·)</span><strong>{expValue.toExponential(3)}</strong>
        <span>Σ exp</span><strong>{denominator.toFixed(5)}</strong>
        <span>weight</span><strong>{(weight*100).toFixed(3)}%</strong>
      </div>
      <div class="softmax-row">
        {#each result.weights[selectedRow] as w,i}
          <button type="button" class="trace-softmax-button" data-index={i} class:active={i===selectedCol} on:click={() => onSelect(selectedRow,i)}>
            <span>{label(i)}</span><b>{(w*100).toFixed(1)}%</b>
          </button>
        {/each}
      </div>
    </article>

    <div class="trace-arrow">→</div>

    <article class="trace-stage value-stage">
      <div class="step">E</div>
      <h4>weight × V</h4>
      <div class="value-flow">
        <div><b>V {label(selectedCol)}</b><div class="trace-vector"><UpstreamVectorCanvas data={v} colorScale="green" active={true}/></div></div>
        <span>× {(weight*100).toFixed(2)}%</span>
        <div><b>이 Key의 기여량</b><div class="trace-vector contribution"><UpstreamVectorCanvas data={contribution} colorScale="purple" active={true}/></div></div>
        <span>Σ all keys →</span>
        <div><b>Context · Q {label(selectedRow)}</b><div class="trace-vector context"><UpstreamVectorCanvas data={context} colorScale="purple" active={true}/></div></div>
      </div>
      <div class="numeric-row">
        <code>V = [{v.map(x=>x.toFixed(2)).join(', ')}]</code>
        <code>contribution = [{contribution.map(x=>x.toFixed(3)).join(', ')}]</code>
        <code>context = [{context.map(x=>x.toFixed(3)).join(', ')}]</code>
      </div>
      {#if highlightDim >= 0}
        <div class="numeric-row dim-highlight" data-highlighted-dim={highlightDim} data-highlighted-v={v[highlightDim]} data-highlighted-contribution={contribution[highlightDim]} data-highlighted-context={context[highlightDim]}>
          <code>selected dim d{highlightDim} → V[d{highlightDim}]={v[highlightDim]?.toFixed(3)} · contribution[d{highlightDim}]={contribution[highlightDim]?.toFixed(3)} · context[d{highlightDim}]={context[highlightDim]?.toFixed(3)}</code>
        </div>
      {/if}
    </article>
  </div>
</section>

<style>
.attention-cell-trace{margin-top:22px;padding-top:18px;border-top:1px solid #e2e6ec}
.trace-head{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:14px}.eyebrow{font-size:10px;letter-spacing:.08em;color:#8170ae}.trace-head h3{font-size:17px;margin:2px 0 3px}.trace-head p{margin:0;font-size:12px;color:#6d7684}.weight-badge{min-width:126px;padding:9px 12px;border-radius:9px;background:#f2eff9;text-align:right}.weight-badge small{display:block;font-size:10px;color:#806cae}.weight-badge strong{font:17px ui-monospace,SFMono-Regular,Menlo,monospace;color:#674e9f}.weight-badge.masked{background:#f4f5f7}.weight-badge.masked strong{color:#7b8492}
.selector-row{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:7px 10px;align-items:center;margin-bottom:18px}.selector-row>span{font-size:10px;font-weight:700;color:#687386}.selector-buttons{display:flex;gap:4px;flex-wrap:wrap}.selector-buttons button,.softmax-row button{border:1px solid #dfe3e8;background:#fff;border-radius:6px;min-height:30px;padding:4px 7px;font-size:10px;color:#667085;cursor:pointer}.selector-buttons button.active,.softmax-row button.active{background:#ece7f7;border-color:#a895cf;color:#5e4894}.selector-buttons button.future{border-style:dashed;color:#a0a6b0}
.trace-flow{display:flex;align-items:stretch;gap:9px;min-width:1260px}.trace-stage{position:relative;flex:1;min-width:0;padding:12px;border:1px solid #e3e7ed;border-radius:10px;background:#fbfcfd}.trace-stage h4{font-size:12px;margin:0 0 10px;color:#505a69}.step{position:absolute;top:8px;right:8px;width:19px;height:19px;border-radius:50%;background:#293548;color:#fff;text-align:center;line-height:19px;font-size:10px;font-weight:700}.trace-arrow{display:flex;align-items:center;color:#826db7;font-size:18px}
.qk-stage{flex:1.5}.qk-vectors,.value-flow{display:flex;align-items:center;justify-content:center;gap:8px}.qk-vectors>div,.value-flow>div{display:flex;flex-direction:column;align-items:center;gap:4px}.qk-vectors b,.value-flow b{font-size:10px;color:#667085}.trace-vector{position:relative;width:25px;height:72px;border:1px solid #dde2e8;border-radius:4px;overflow:hidden;background:#fff}.trace-vector.contribution,.trace-vector.context{outline:2px solid #ded6f1}.product-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:10px}.product-grid .product-dim{appearance:none;font:inherit;text-align:left;padding:5px;min-width:44px;min-height:44px;border-radius:5px;background:#fff;border:1px solid #e8eaee;cursor:pointer}.product-grid .product-dim.selected{border-color:#a895cf;background:#f2eff9}.product-grid small{display:block;font-size:10px;color:#9299a5}.product-grid code{display:block;font-size:10px;color:#697386;white-space:nowrap}.product-grid b{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}.product-matrix{display:flex;justify-content:center;margin-top:8px}
.trace-query-button:disabled{opacity:.4;cursor:not-allowed}
.equation{display:grid;grid-template-columns:1fr auto;gap:6px 10px;align-items:baseline}.equation span{font-size:10px;color:#7b8492}.equation strong{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}.score-stage p{font-size:10px;line-height:1.4;color:#7b8492;margin:10px 0 0}.mask-stage.masked{background:#f7f7f8}.masked-block,.pass-block{display:flex;flex-direction:column;gap:7px}.masked-block strong,.pass-block strong{font-size:11px;color:#596273}.masked-block code,.pass-block code{font-size:10px}.masked-block span{font-size:10px;color:#7b8492}
.softmax-stage{flex:1.25}.softmax-row{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-top:10px}.softmax-row button{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3px}.softmax-row button span{font-size:10px}.softmax-row button b{font-size:10px}
.value-stage{flex:1.7}.value-flow>span{font-size:10px;color:#7d8593;white-space:nowrap}.numeric-row{display:flex;flex-direction:column;gap:3px;margin-top:10px}.numeric-row code{font-size:10px;color:#667085;white-space:normal;overflow:visible;text-overflow:clip;line-height:1.35;overflow-wrap:anywhere}.numeric-row.dim-highlight{margin-top:6px;padding:6px 8px;border-radius:6px;background:#f2eff9;border:1px solid #ded6f1}.numeric-row.dim-highlight code{color:#5e4894;font-size:14px;line-height:1.5}
@media(max-width:1100px){.attention-cell-trace{overflow:visible}.selector-row{grid-template-columns:auto 1fr}.trace-head{min-width:0}.trace-flow{min-width:0;display:flex;flex-direction:column;gap:8px}.trace-arrow{justify-content:center;transform:rotate(90deg);height:24px}.trace-stage{width:100%}.product-grid{grid-template-columns:repeat(2,1fr)}.qk-vectors,.value-flow{gap:12px}.numeric-row code{font-size:10px}.weight-badge{min-width:108px}}
</style>
