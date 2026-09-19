<script>
  import * as d3 from 'd3';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';
  export let patches = [];
  export let gridSize = 8;
  export let cellSize = 18;
  export let onSelect = () => {};
  export let selected = null;

  $: matrix = Array.from({length:gridSize},(_,r)=>patches.slice(r*gridSize,(r+1)*gridSize));
  const color = d3.scaleSequential(d3.interpolateGreys).domain([0,1]);
  function hover(event,d){ onSelect(d.rowIndex,d.colIndex,d.cell); }
</script>

<div class="patch-grid">
  <div class="axis top">8 columns</div>
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
    showTooltip={(e,v)=>'patch mean '+v.toFixed(3)}
  />
  <div class="caption">32×32 pixels → 4×4 average patches → 8×8 = 64 features</div>
</div>

<style>
.patch-grid{display:flex;flex-direction:column;align-items:center;gap:5px}.axis,.caption{font-size:9px;color:#7b8492}.caption{max-width:250px;text-align:center;line-height:1.4}
</style>
