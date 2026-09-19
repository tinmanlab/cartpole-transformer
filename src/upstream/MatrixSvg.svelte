<!--
Adapted from poloclub/transformer-explainer
src/components/common/MatrixSvg.svelte
Commit: bfe50afba10b9b560b84143ee1107d977defa74f
MIT License, Copyright (c) 2022 Polo Club of Data Science
-->
<script>
  import * as d3 from 'd3';

  export let data = [];
  export let cellHeight = 12;
  export let cellWidth = 12;
  export let rowGap = 2;
  export let colGap = 2;
  export let shape = 'rect';
  export let colorScale;
  export let highlightRow;
  export let highlightCol;
  export let onMouseOverCell;
  export let onMouseOutCell;
  export let showTooltip;

  let svgEl;
  let tooltipData = '';
  let tooltipVisible = false;
  let tooltipX = 0;
  let tooltipY = 0;

  $: rowLen = data.length;
  $: dimension = data[0]?.length || 0;
  $: svgWidth = dimension * cellWidth + Math.max(0,dimension-1)*colGap;
  $: svgHeight = rowLen * cellHeight + Math.max(0,rowLen-1)*rowGap;
  $: matrixColorScale = typeof colorScale === 'function'
    ? colorScale
    : d3.interpolate('white','#8b5cf6');

  function onCellOver(event,d) {
    const el = event.currentTarget;
    onMouseOverCell?.(event,d,el);
    const value = showTooltip?.(event,d.cell,el);
    if (value !== undefined && value !== null) {
      const parent = svgEl.getBoundingClientRect();
      const box = el.getBoundingClientRect();
      tooltipData = value;
      tooltipVisible = true;
      tooltipX = box.left + box.width/2 - parent.left;
      tooltipY = box.top - parent.top - 7;
    }
  }
  function onCellOut(event,d) {
    onMouseOutCell?.(event,d,event.currentTarget);
    tooltipVisible = false;
  }

  function drawMatrixSvg() {
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    const rows = svg.selectAll('g.g-row')
      .data(data)
      .join('g')
      .attr('class',(d,i)=>'g-row g-row-'+i+' row-'+i)
      .attr('transform',(d,i)=>'translate(0,'+(i*cellHeight+i*rowGap)+')');

    const join = shape === 'circle'
      ? rows.selectAll('circle.cell')
      : rows.selectAll('rect.cell');

    const cells = join
      .data((d,rowIndex)=>d.map((cell,colIndex)=>({cell,rowIndex,colIndex})))
      .join(shape)
      .attr('class',d=>'cell row-'+d.rowIndex+' col-'+d.colIndex)
      .on('mouseenter',onCellOver)
      .on('mouseleave',onCellOut)
      .attr('fill',(d,i)=>Number.isFinite(d.cell) ? matrixColorScale(d.cell,i) : '#e5e7eb');

    if (shape === 'circle') {
      cells
        .attr('cx',(d,i)=>cellWidth/2+i*cellWidth+i*colGap)
        .attr('cy',cellHeight/2)
        .attr('r',Math.min(cellWidth,cellHeight)/2)
        .attr('stroke','#e5e7eb');
    } else {
      cells
        .attr('x',(d,i)=>i*cellWidth+i*colGap)
        .attr('y',0)
        .attr('width',cellWidth)
        .attr('height',cellHeight)
        .attr('rx',2);
    }

    cells.attr('opacity',d=>{
      if (highlightRow === undefined && highlightCol === undefined) return 1;
      if (highlightRow !== undefined && highlightCol !== undefined)
        return d.rowIndex===highlightRow && d.colIndex===highlightCol ? 1 : .16;
      if (highlightRow !== undefined) return d.rowIndex===highlightRow ? 1 : .16;
      return d.colIndex===highlightCol ? 1 : .16;
    });
  }

  $: if (data && svgEl) drawMatrixSvg();
</script>

<div class="matrix-wrap">
  <svg bind:this={svgEl} width={svgWidth} height={svgHeight}></svg>
  {#if tooltipVisible}
    <div class="tip" style={"left:"+tooltipX+"px;top:"+tooltipY+"px"}>{tooltipData}</div>
  {/if}
</div>

<style>
.matrix-wrap{position:relative;display:inline-block;line-height:0}
.tip{position:absolute;transform:translate(-50%,-100%);z-index:5;background:white;border:1px solid #d1d5db;border-radius:4px;padding:4px 6px;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563;white-space:nowrap;pointer-events:none;box-shadow:0 3px 9px rgba(0,0,0,.08)}
</style>
