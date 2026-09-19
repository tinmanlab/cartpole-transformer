<script>
  import * as d3 from 'd3';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  export let patches = [];
  export let gridSize = 8;
  export let cellSize = 18;
  export let patchSize = 2;
  export let onSelect = () => {};
  export let selected = null;
  export let mode = 'intensity';

  $: matrix = Array.from({length:gridSize},(_,r)=>patches.slice(r*gridSize,(r+1)*gridSize));
  $: maxAbs = Math.max(1e-6,...patches.map(v=>Math.abs(v)));
  $: color = mode === 'delta'
    ? d3.scaleLinear().domain([-maxAbs,0,maxAbs]).range(['#e88989','#ffffff','#7098df']).clamp(true)
    : d3.scaleSequential(d3.interpolateGreys).domain([0,1]);
  function hover(event,d){ onSelect(d.rowIndex,d.colIndex,d.cell); }
</script>

<div class="patch-grid">
  <div class="axis top">{gridSize} columns</div>
  <UpstreamMatrixSvg
    data={matrix}
    cellHeight={cellSize}
    cellWidth={cellSize}
    rowGap={2}
    colGap={2}
    shape="rect"
    colorScale={(v)=>color(v)}
    highlightRow={selected?.row}
    highlightCol={selected?.col}
    onMouseOverCell={hover}
    showTooltip={(e,v)=>(mode==='delta'?'Δpatch ':'patch mean ')+v.toFixed(3)}
  />
  <div class="caption">{mode==='delta'?'difference from previous sampled frame':'32×32 pixels → '+patchSize+'×'+patchSize+' average patches → '+gridSize+'×'+gridSize+' = '+patches.length+' features'}</div>
</div>

<style>
.patch-grid{display:flex;flex-direction:column;align-items:center;gap:5px}.axis,.caption{font-size:10px;color:#7b8492}.caption{max-width:250px;text-align:center;line-height:1.4}
</style>
