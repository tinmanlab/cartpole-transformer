<script>
  import * as d3 from 'd3';
  export let data = [];
  export let mode = 'weights';
  export let selectedRow = null;
  export let selectedCol = null;
  export let onSelect = () => {};
  export let cellSize = 19;
  export let gap = 2;
  export let pad = 24;
  export let showLabels = true;

  $: n = data.length;
  $: finite = data.flat().filter(Number.isFinite);
  $: extent = finite.length ? d3.extent(finite) : [0, 1];
  $: scale = mode === 'weights'
    ? d3.scaleSequential(d3.interpolatePurples).domain([0, Math.max(.001, d3.max(finite) || 1)])
    : d3.scaleLinear().domain([extent[0] || -1, 0, extent[1] || 1]).range(['#eef2ff','#ffffff','#7b61c9']).clamp(true);
  $: labelPad = showLabels ? pad : 3;
  $: width = labelPad + n * (cellSize + gap);
  $: height = labelPad + n * (cellSize + gap);
</script>

<svg viewBox={"0 0 " + width + " " + height} class="matrix" aria-label={mode === 'weights' ? 'softmax attention matrix' : 'query key dot product matrix'}>
  {#if showLabels}
    {#each Array(n) as _, i}
      <text x={labelPad - 6} y={labelPad + i*(cellSize+gap) + cellSize*.68} text-anchor="end">{i === n-1 ? 't' : 't−'+(n-1-i)}</text>
      <text x={labelPad + i*(cellSize+gap) + cellSize*.5} y={15} text-anchor="middle">{i === n-1 ? 't' : '−'+(n-1-i)}</text>
    {/each}
  {/if}
  {#each data as row, r}
    {#each row as value, c}
      <g on:mouseenter={() => onSelect(r,c)} on:focus={() => onSelect(r,c)} tabindex="0" role="button" aria-label={"row "+r+" column "+c}>
        <rect
          x={labelPad + c*(cellSize+gap)}
          y={labelPad + r*(cellSize+gap)}
          width={cellSize}
          height={cellSize}
          rx={Math.max(2,cellSize*.18)}
          fill={Number.isFinite(value) ? scale(value) : '#f1f2f4'}
          stroke={(selectedRow===r || selectedCol===c) ? '#4f5d75' : '#e1e4ea'}
          stroke-width={(selectedRow===r && selectedCol===c) ? 1.7 : .7}
          opacity={Number.isFinite(value) ? 1 : .42}
        />
        {#if Number.isFinite(value)}
          <title>{mode === 'weights' ? (value*100).toFixed(1)+'%' : value.toFixed(2)}</title>
        {/if}
      </g>
    {/each}
  {/each}
</svg>

<style>
.matrix{display:block;width:100%;height:auto;overflow:visible}
text{font:8px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#7a8392}
g:focus{outline:none}g:focus rect{stroke:#111827;stroke-width:1.7}
</style>
