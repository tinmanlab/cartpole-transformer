<script>
  import { onMount } from 'svelte';
  export let frame = [];
  export let size = 32;
  export let label = '';
  export let active = false;
  export let pixelScale = 4;

  let canvas;
  let ro;

  function draw() {
    if (!canvas || !frame?.length) return;
    const ctx = canvas.getContext('2d');
    const ratio = window.devicePixelRatio || 1;
    const cssSize = size * pixelScale;
    canvas.width = cssSize * ratio;
    canvas.height = cssSize * ratio;
    canvas.style.width = cssSize + 'px';
    canvas.style.height = cssSize + 'px';
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for (let y=0;y<size;y++) {
      for (let x=0;x<size;x++) {
        const v=Math.max(0,Math.min(1,frame[y*size+x] ?? 0));
        const c=Math.round(v*255);
        ctx.fillStyle='rgb('+c+','+c+','+c+')';
        ctx.fillRect(x*pixelScale*ratio,y*pixelScale*ratio,pixelScale*ratio,pixelScale*ratio);
      }
    }
  }

  onMount(()=>{
    ro=new ResizeObserver(draw);
    if(canvas?.parentElement) ro.observe(canvas.parentElement);
    draw();
    return ()=>ro?.disconnect();
  });
  $: if(canvas && frame) draw();
</script>

<div class:active class="vision-frame">
  {#if label}<div class="label">{label}</div>{/if}
  <canvas bind:this={canvas} aria-label={label || 'vision frame'}></canvas>
</div>

<style>
.vision-frame{display:inline-flex;flex-direction:column;align-items:center;gap:4px;padding:5px;border:1px solid #e2e5ea;border-radius:7px;background:#111827}
.vision-frame.active{outline:2px solid #8b5cf6;outline-offset:2px}.label{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cbd5e1}canvas{display:block;image-rendering:pixelated}
</style>
