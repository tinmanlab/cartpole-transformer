<!--
Adapted from poloclub/transformer-explainer
src/components/common/VectorCanvas.svelte
Commit: bfe50afba10b9b560b84143ee1107d977defa74f
MIT License, Copyright (c) 2022 Polo Club of Data Science
-->
<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  export let active = false;
  export let data = [];
  export let colorScale = 'gray';

  let canvas;
  let resizeObserver;

  const palettes = {
    gray: ['#f3f4f6','#9ca3af'],
    blue: ['#eff6ff','#60a5fa'],
    red: ['#fef2f2','#f87171'],
    green: ['#f0fdf4','#4ade80'],
    purple: ['#faf5ff','#a78bfa']
  };

  $: color = typeof colorScale === 'function'
    ? colorScale
    : d3.interpolate(...(palettes[colorScale] || palettes.gray));

  function drawCanvas() {
    const ctx = canvas?.getContext('2d');
    const parent = canvas?.parentElement;
    if (!ctx || !parent) return;

    const width = parent.clientWidth;
    const height = parent.clientHeight;
    const pixelRatio = 4;

    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.clearRect(0,0,canvas.width,canvas.height);

    const safe = data.length ? data : [0];
    for (let y = 0; y < height; y++) {
      const idx = Math.min(safe.length - 1, Math.floor(y / Math.max(1,height) * safe.length));
      const value = Math.max(-1, Math.min(1, safe[idx] ?? 0));
      ctx.fillStyle = color((value + 1) / 2);
      ctx.fillRect(0, y * pixelRatio, width * pixelRatio, pixelRatio);
    }
  }

  onMount(() => {
    resizeObserver = new ResizeObserver(drawCanvas);
    if (canvas?.parentElement) resizeObserver.observe(canvas.parentElement);
    drawCanvas();
    return () => resizeObserver?.disconnect();
  });

  $: if (data && canvas && color) drawCanvas();
</script>

<canvas class:active bind:this={canvas}></canvas>

<style>
canvas{position:absolute;inset:0;display:block;width:100%;height:100%;overflow:hidden;opacity:.72;transition:opacity .2s}
canvas.active,canvas:hover{opacity:1}
</style>
