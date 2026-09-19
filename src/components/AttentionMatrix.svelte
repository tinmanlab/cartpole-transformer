<script>
  import * as d3 from 'd3';
  export let data = [];
  export let mode = 'weights';
  export let selectedRow = null;
  export let selectedCol = null;
  export let onSelect = () => {};

  $: n = data.length;
  $: finite = data.flat().filter(Number.isFinite);
  $: extent = finite.length ? d3.extent(finite) : [0, 1];
  $: scale = mode === 'weights'
    ? d3.scaleSequential(d3.interpolatePurples).domain([0, Math.max(.001, d3.max(finite) || 1)])
    : d3.scaleLinear().domain([extent[0] || -1, 0, extent[1] || 1]).range(['#eef2ff','#ffffff','#7b61c9']).clamp(true);
  const size = 19, gap = 2, pad = 24;
  $: width = pad + n * (size + gap);
  $: height = pad + n * (size + gap);
</script>

<svg viewBox={"0 0 " + width + " " + height} class="matrix" aria-label={mode === 'weights' ? 'softmax attention matrix' : 'query key dot product matrix'}>
  {#each Array(n) as _, i}
    <text x={pad - 7} y={pad + i*(size+gap) + size*.68} text-anchor="end">{i === n-1 ? 't' : 't−'+(n-1-i)}</text>
    <text x={pad + i*(size+gap) + size*.5} y={17} text-anchor="middle">{i === n-1 ? 't' : '−'+(n-1-i)}</text>
  {/each}
  {#each data as row, r}
    {#each row as value, c}
      <g on:mouseenter={() => onSelect(r,c)} on:focus={() => onSelect(r,c)} tabindex="0" role="button" aria-label={"row "+r+" column "+c}>
        <rect
          x={pad + c*(size+gap)}
          y={pad + r*(size+gap)}
          width={size}
          height={size}
          rx="5"
          fill={Number.isFinite(value) ? scale(value) : '#f1f2f4'}
          stroke={(selectedRow===r || selectedCol===c) ? '#4f5d75' : '#e1e4ea'}
          stroke-width={(selectedRow===r && selectedCol===c) ? 2 : 1}
          opacity={Number.isFinite(value) ? 1 : .45}
        />
        {#if Number.isFinite(value)}
          <title>{mode === 'weights' ? (value*100).toFixed(1)+'%' : value.toFixed(2)}</title>
        {/if}
      </g>
    {/each}
  {/each}
</svg>

<style>
.matrix{display:block;max-width:100%;height:auto;overflow:visible}
text{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#7a8392}
g:focus{outline:none}g:focus rect{stroke:#111827;stroke-width:2}
</style>
