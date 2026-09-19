<!--
Adapted from poloclub/transformer-explainer
src/components/Sankey.svelte
Commit: bfe50afba10b9b560b84143ee1107d977defa74f
MIT License, Copyright (c) 2022 Polo Club of Data Science
-->
<script>
  import * as d3 from 'd3';
  import { onMount, tick } from 'svelte';

  export let pathMap = {};
  export let redrawKey = 0;

  let svgEl;
  let resizeObserver;

  const gradientMap = {
    'gray-blue': {0:'#d1d5db',100:'#93c5fd'},
    'blue-mix': {0:'#93c5fd',50:'#fca5a5',100:'#86efac'},
    'blue-purple': {0:'#93c5fd',100:'#c4b5fd'},
    'purple-purple': {0:'#c4b5fd',100:'#c4b5fd'},
    'green-purple': {0:'#86efac',100:'#c4b5fd'}
  };

  function createGradients() {
    const svg = d3.select(svgEl);
    svg.select('defs').remove();
    const defs = svg.append('defs');

    Object.keys(gradientMap).forEach(key => {
      const stops = gradientMap[key];
      const grad = defs.append('linearGradient')
        .attr('id',key).attr('x1','0%').attr('y1','0%').attr('x2','100%').attr('y2','0%');
      Object.keys(stops).forEach(stop => {
        grad.append('stop').attr('offset',stop+'%').attr('stop-color',stops[stop]);
      });
    });

    defs.append('marker')
      .attr('id','arrow-head')
      .attr('viewBox','0 0 10 10')
      .attr('refX',9).attr('refY',5)
      .attr('markerWidth',6).attr('markerHeight',6)
      .attr('orient','auto-start-reverse')
      .append('path').attr('d','M 0 0 L 10 5 L 0 10 z').attr('fill','#8b5cf6');
  }

  const pathAdjustor = (source,target,curve) => {
    const distance = target.left - source.right;
    const maxDistance = 100;
    const curveOffset = distance > maxDistance ? curve : curve * (distance / maxDistance);
    return {curveOffset};
  };

  const centerPathGenerator = (source,target,curve) => {
    const parent = svgEl.getBoundingClientRect();
    const sy = source.top - parent.top + source.height/2;
    const ty = target.top - parent.top + target.height/2;
    const sx = source.right - parent.left;
    const tx = target.left - parent.left;
    const {curveOffset} = pathAdjustor(
      {right:sx},{left:tx},curve
    );
    return 'M '+sx+','+sy+' C '+(sx+curveOffset)+','+sy+' '+(tx-curveOffset)+','+ty+' '+tx+','+ty;
  };

  const defaultPathGenerator = (source,target,curve) => {
    const parent = svgEl.getBoundingClientRect();
    const s={left:source.left-parent.left,right:source.right-parent.left,top:source.top-parent.top,bottom:source.bottom-parent.top};
    const t={left:target.left-parent.left,right:target.right-parent.left,top:target.top-parent.top,bottom:target.bottom-parent.top};
    const {curveOffset}=pathAdjustor(s,t,curve);
    return 'M '+s.right+','+s.top+
      ' C '+(s.right+curveOffset)+','+s.top+' '+(t.left-curveOffset)+','+t.top+' '+t.left+','+t.top+
      ' L '+t.left+','+t.bottom+
      ' C '+(t.left-curveOffset)+','+t.bottom+' '+(s.right+curveOffset)+','+s.bottom+' '+s.right+','+s.bottom+' Z';
  };

  async function drawPath() {
    await tick();
    if (!svgEl) return;
    createGradients();
    const svg=d3.select(svgEl);
    const names=Object.keys(pathMap);
    const groups=svg.selectAll('g.path-group').data(names).join('g').attr('class',d=>'path-group '+d);

    groups.selectAll('path.sankey-path')
      .data(name=>{
        const items=pathMap[name]||[];
        return items.flatMap(item=>{
          const sources=[...document.querySelectorAll(item.from)];
          const targets=[...document.querySelectorAll(item.to)];
          return sources.map((src,i)=>{
            const source=src?.getBoundingClientRect();
            const target=targets[Math.min(i,targets.length-1)]?.getBoundingClientRect();
            const generator=item.pathGenerator || (item.type==='stroke'?centerPathGenerator:defaultPathGenerator);
            return {
              path:source&&target?generator(source,target,item.curve||70):'',
              fill:item.type==='stroke'?'none':(item.gradientId?'url(#'+item.gradientId+')':item.fill),
              stroke:item.type==='stroke'?(item.gradientId?'url(#'+item.gradientId+')':item.fill):'none',
              opacity:item.opacity ?? .55,
              arrow:item.arrow !== false && item.type==='stroke'
            };
          });
        });
      })
      .join('path')
      .attr('class','sankey-path')
      .attr('d',d=>d.path)
      .attr('fill',d=>d.fill)
      .attr('stroke',d=>d.stroke)
      .attr('stroke-width',d=>d.fill==='none'?3:0)
      .attr('opacity',d=>d.opacity)
      .attr('marker-end',d=>d.arrow?'url(#arrow-head)':null);
  }

  onMount(()=>{
    resizeObserver=new ResizeObserver(drawPath);
    document.querySelectorAll('.resize-watch').forEach(el=>resizeObserver.observe(el));
    window.addEventListener('resize',drawPath);
    drawPath();
    return ()=>{resizeObserver?.disconnect();window.removeEventListener('resize',drawPath);};
  });

  $: redrawKey, pathMap, svgEl && drawPath();
</script>

<svg bind:this={svgEl} class="upstream-sankey" aria-hidden="true"></svg>

<style>
.upstream-sankey{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;z-index:1}
</style>
