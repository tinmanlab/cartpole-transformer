<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  export let data = [];
  export let kind = 'state';
  export let active = false;
  export let label = '';

  let canvas;
  let ro;
  const palettes = {
    state: ['#eef0f3', '#6b7280'],
    q: ['#eef3ff', '#4f74ff'],
    k: ['#fff0f0', '#e76363'],
    v: ['#eefbf4', '#3fa574'],
    out: ['#f5f0ff', '#8b66d8']
  };

  function draw() {
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const w = Math.max(18, rect.width);
    const h = Math.max(48, rect.height);
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = w * ratio;
    canvas.height = h * ratio;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const [a, b] = palettes[kind] || palettes.state;
    const color = d3.scaleLinear().domain([-1, 1]).range([a, b]).clamp(true);
    const n = Math.max(1, data.length);
    const cellH = h / n;
    data.forEach((value, i) => {
      ctx.fillStyle = color(Math.max(-1, Math.min(1, value)));
      ctx.fillRect(0, i * cellH, w, Math.ceil(cellH));
    });
  }

  onMount(() => {
    ro = new ResizeObserver(draw);
    if (canvas?.parentElement) ro.observe(canvas.parentElement);
    draw();
    return () => ro?.disconnect();
  });

  $: data, active, kind, canvas && draw();
</script>

<div class:active class="strip-wrap" title={label}>
  <canvas bind:this={canvas}></canvas>
  {#if label}<span>{label}</span>{/if}
</div>

<style>
.strip-wrap{position:relative;width:26px;height:74px;border:1px solid #d9dde5;border-radius:6px;overflow:hidden;background:white;transition:transform .15s,border-color .15s}
.strip-wrap.active{transform:scale(1.06);border-color:#6978d6;box-shadow:0 0 0 2px rgba(105,120,214,.12)}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
span{position:absolute;bottom:2px;left:50%;transform:translateX(-50%);font-size:9px;color:#1f2937;background:rgba(255,255,255,.78);padding:0 2px;border-radius:3px}
</style>
