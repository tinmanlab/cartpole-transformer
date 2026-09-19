<!--
Adapted from poloclub/transformer-explainer
src/components/AttentionMatrix.svelte
Commit: bfe50afba10b9b560b84143ee1107d977defa74f
MIT License, Copyright (c) 2022 Polo Club of Data Science
-->
<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import { gsap } from 'gsap';
  import MatrixSvg from './MatrixSvg.svelte';

  export let scores = [];
  export let masked = [];
  export let weights = [];
  export let selectedRow = 0;
  export let selectedCol = 0;
  export let onSelect = () => {};

  let root;
  let qkEl;
  let maskEl;
  let softmaxEl;

  $: finiteScores = scores.flat().filter(Number.isFinite);
  $: extent = d3.extent(finiteScores);
  $: scoreColor = d3.scaleLinear()
    .domain([extent[0] || -1, 0, extent[1] || 1])
    .range(['#eef2ff', '#ffffff', '#6d5ab6'])
    .clamp(true);
  $: maxWeight = Math.max(.001, d3.max(weights.flat()) || 1);
  $: weightColor = d3.scaleSequential(d3.interpolatePurples).domain([0, maxWeight]);

  function hover(event, d) {
    onSelect(d.rowIndex, d.colIndex);
  }

  function play() {
    if (!root) return;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return;
    const tl = gsap.timeline();
    tl.fromTo(qkEl, { opacity: .25, y: 5 }, { opacity: 1, y: 0, duration: .28 })
      .fromTo(maskEl, { opacity: .15, x: -8 }, { opacity: 1, x: 0, duration: .28 }, '>-0.05')
      .fromTo(softmaxEl, { opacity: .15, x: -8 }, { opacity: 1, x: 0, duration: .28 }, '>-0.05');

    root.querySelectorAll('.calc-arrow').forEach((arrow, i) => {
      const length = arrow.getTotalLength?.() || 40;
      gsap.fromTo(arrow,
        { strokeDasharray: length, strokeDashoffset: length },
        { strokeDashoffset: 0, duration: .35, delay: .18 + i * .22, ease: 'power2.out' }
      );
    });
  }

  onMount(() => {
    play();
  });
</script>

<div class="attention-expansion" bind:this={root}>
  <div class="calc" bind:this={qkEl}>
    <div class="calc-title"><b>1</b><span>Dot product</span><code>QKᵀ / √d</code></div>
    <MatrixSvg
      data={scores}
      cellHeight={18}
      cellWidth={18}
      rowGap={3}
      colGap={3}
      shape="circle"
      colorScale={(v)=>scoreColor(v)}
      highlightRow={selectedRow}
      highlightCol={selectedCol}
      onMouseOverCell={hover}
      showTooltip={(e,v)=>Number.isFinite(v)?v.toFixed(3):'—'}
    />
    <p>Query와 Key가 얼마나 비슷한지 모든 시간쌍을 비교합니다.</p>
  </div>

  <svg class="arrow-svg" viewBox="0 0 48 30" aria-hidden="true">
    <path class="calc-arrow" d="M4 15 H40 M33 8 L40 15 L33 22"/>
  </svg>

  <div class="calc" bind:this={maskEl}>
    <div class="calc-title"><b>2</b><span>Causal mask</span><code>future → −∞</code></div>
    <MatrixSvg
      data={masked}
      cellHeight={18}
      cellWidth={18}
      rowGap={3}
      colGap={3}
      shape="rect"
      colorScale={(v)=>scoreColor(v)}
      highlightRow={selectedRow}
      highlightCol={selectedCol}
      onMouseOverCell={hover}
      showTooltip={(e,v)=>Number.isFinite(v)?v.toFixed(3):'masked'}
    />
    <p>현재 시점은 미래 token을 볼 수 없습니다. 위쪽 삼각형이 차단됩니다.</p>
  </div>

  <svg class="arrow-svg" viewBox="0 0 48 30" aria-hidden="true">
    <path class="calc-arrow" d="M4 15 H40 M33 8 L40 15 L33 22"/>
  </svg>

  <div class="calc" bind:this={softmaxEl}>
    <div class="calc-title"><b>3</b><span>Softmax</span><code>Σ a = 1</code></div>
    <MatrixSvg
      data={weights}
      cellHeight={18}
      cellWidth={18}
      rowGap={3}
      colGap={3}
      shape="rect"
      colorScale={(v)=>weightColor(v)}
      highlightRow={selectedRow}
      highlightCol={selectedCol}
      onMouseOverCell={hover}
      showTooltip={(e,v)=>(v*100).toFixed(2)+'%'}
    />
    <p>점수를 양수 비율로 바꿉니다. 마지막 row가 지금 force를 만드는 attention입니다.</p>
  </div>
</div>

<style>
.attention-expansion{display:grid;grid-template-columns:1fr 48px 1fr 48px 1fr;gap:7px;align-items:center}
.calc{display:flex;flex-direction:column;align-items:center;gap:7px;min-width:0}
.calc-title{display:grid;grid-template-columns:20px 1fr;column-gap:6px;align-items:center;width:100%;max-width:180px}
.calc-title b{grid-row:1/3;width:20px;height:20px;line-height:20px;text-align:center;border-radius:50%;background:#293548;color:#fff;font-size:9px}
.calc-title span{font-size:10px;font-weight:700;color:#4b5563}.calc-title code{font-size:8px;color:#7c6ab5}
.calc p{margin:0;max-width:190px;font-size:8px;line-height:1.4;color:#7b8492;text-align:center}
.arrow-svg{width:40px;height:28px;overflow:visible}
.calc-arrow{fill:none;stroke:#8b5cf6;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
@media(max-width:850px){.attention-expansion{min-width:720px}}
</style>
