<script>
  import * as d3 from 'd3';
  import UpstreamVectorCanvas from '../upstream/VectorCanvas.svelte';
  import UpstreamMatrixSvg from '../upstream/MatrixSvg.svelte';

  export let result;
  export let model;
  export let scenario='clean';
  export let onScenarioChange=()=>{};
  export let selectedToken=15;
  export let onSelectToken=()=>{};
  export let groundTruthState=null;
  export let onClose=()=>{};
  export let frameIntervalMs=60;

  $: lastTime=model?.sequence_length-1 || 7;
  $: selectedMeta=result.tokenMeta[selectedToken];
  $: latestState=result.finalStateIndex;
  $: latestVision=result.finalVisionIndex;
  $: latestStateToState = result.weights[latestState].filter((_,i)=>result.tokenMeta[i].modality==='state').reduce((a,b)=>a+b,0);
  $: latestStateToVision = result.weights[latestState].filter((_,i)=>result.tokenMeta[i].modality==='vision').reduce((a,b)=>a+b,0);
  $: latestVisionToState = result.weights[latestVision].filter((_,i)=>result.tokenMeta[i].modality==='state').reduce((a,b)=>a+b,0);
  $: latestVisionToVision = result.weights[latestVision].filter((_,i)=>result.tokenMeta[i].modality==='vision').reduce((a,b)=>a+b,0);
  $: weightColor=d3.scaleSequential(d3.interpolatePurples).domain([0,Math.max(.001,d3.max(result.weights.flat())||1)]);

  const scenarios=[
    ['clean','Clean'],['noisy-state','Noisy state'],['missing-state','State missing'],['partial-vision','Partial vision'],['missing-vision','Vision missing']
  ];

  function timeLabel(t){return t===lastTime?'t':'−'+Math.round((lastTime-t)*frameIntervalMs)+'ms';}
  function metric(name,key='mean_steps'){return model?.closed_loop?.[name]?.[key];}
</script>

<section class="fusion-detail-wide">
  <header class="detail-head">
    <div>
      <div class="eyebrow">STATE + VISION · SIMPLE TYPED-TOKEN FUSION</div>
      <h2>같은 시간의 두 modality를 하나의 self-attention 안에서 비교</h2>
      <p>Cross-attention block을 추가하지 않았습니다. 각 시점의 State token과 Vision token이 같은 time embedding을 공유하고 modality/type embedding으로 구분됩니다.</p>
    </div>
    <button type="button" on:click={onClose} aria-label="close fusion detail">×</button>
  </header>

  <div class="scenario-bar">
    {#each scenarios as s}
      <button type="button" class:active={scenario===s[0]} on:click={()=>onScenarioChange(s[0])}>{s[1]}</button>
    {/each}
  </div>

  <div class="token-pairs">
    {#each Array(model.sequence_length) as _,t}
      <div class="time-pair">
        <span class="time">{timeLabel(t)}</span>
        <button type="button" class:active={selectedToken===t*2} on:click={()=>onSelectToken(t*2)}>
          <b>State</b><div class="vec"><UpstreamVectorCanvas data={result.tokens[t*2]} colorScale="blue" active={selectedToken===t*2}/></div>
        </button>
        <button type="button" class:active={selectedToken===t*2+1} on:click={()=>onSelectToken(t*2+1)}>
          <b>Vision</b><div class="vec"><UpstreamVectorCanvas data={result.tokens[t*2+1]} colorScale="green" active={selectedToken===t*2+1}/></div>
        </button>
      </div>
    {/each}
  </div>

  <div class="fusion-grid">
    <article class="attention-card">
      <h3>16×16 typed-token attention</h3>
      <div class="matrix">
        <UpstreamMatrixSvg data={result.weights} cellHeight={17} cellWidth={17} rowGap={2} colGap={2}
          shape="rect" colorScale={(v)=>weightColor(v)}
          highlightRow={selectedToken} highlightCol={selectedToken}
          onMouseOverCell={(e,d)=>onSelectToken(d.colIndex)}
          showTooltip={(e,v)=>(v*100).toFixed(2)+'%'}/>
      </div>
      <div class="legend"><span class="s">S</span> state token <span class="v">V</span> vision token · future timestamps are masked, same-time S↔V is allowed.</div>
    </article>

    <article class="cross-card">
      <h3>마지막 시점의 modality attention</h3>
      <div class="bars">
        <div><span>State query → all State keys</span><b>{(latestStateToState*100).toFixed(1)}%</b><i style={"width:"+(latestStateToState*100)+"%"}></i></div>
        <div><span>State query → all Vision keys</span><b>{(latestStateToVision*100).toFixed(1)}%</b><i style={"width:"+(latestStateToVision*100)+"%"}></i></div>
        <div><span>Vision query → all State keys</span><b>{(latestVisionToState*100).toFixed(1)}%</b><i style={"width:"+(latestVisionToState*100)+"%"}></i></div>
        <div><span>Vision query → all Vision keys</span><b>{(latestVisionToVision*100).toFixed(1)}%</b><i style={"width:"+(latestVisionToVision*100)+"%"}></i></div>
      </div>
      <div class="same-time">same-time S(t)→V(t) <b>{(result.weights[latestState][latestVision]*100).toFixed(2)}%</b> · V(t)→S(t) <b>{(result.weights[latestVision][latestState]*100).toFixed(2)}%</b></div>
    </article>

    <article class="state-card">
      <h3>Fusion inferred state</h3>
      <div class="state-grid">
        {#each [['x',0,'m'],['ẋ',1,'m/s'],['θ',2,'rad'],['θ̇',3,'rad/s']] as item}
          <div><span>{item[0]}</span><b>{result.inferredState[item[1]].toFixed(3)} {item[2]}</b>{#if groundTruthState}<small>truth {groundTruthState[item[1]].toFixed(3)}</small>{/if}</div>
        {/each}
      </div>
    </article>
  </div>

  {#if model?.closed_loop}
    <section class="ablation">
      <h3>Degradation ablations · same trained fusion model</h3>
      <div class="ablation-grid">
        <div><span>Clean fusion</span><b>{metric('clean').toFixed(0)}/500</b></div>
        <div><span>State only (vision missing)</span><b>{metric('state_only').toFixed(0)}/500</b></div>
        <div><span>Vision only (state missing)</span><b>{metric('vision_only').toFixed(0)}/500</b></div>
        <div class="negative"><span>Noisy state only</span><b>{metric('noisy_state').toFixed(0)}/500</b></div>
        <div class="negative"><span>Noisy state + vision</span><b>{metric('noisy_state_plus_vision').toFixed(0)}/500</b></div>
        <div><span>Partial vision only</span><b>{metric('partial_vision').toFixed(0)}/500</b></div>
        <div class="positive"><span>Partial vision + state</span><b>{metric('partial_vision_plus_state').toFixed(0)}/500</b></div>
      </div>
      <p><b>Negative result retained:</b> 이 simple fusion은 biased noisy-state에서 average state-estimation error는 줄였지만 closed-loop control은 오히려 악화되었습니다. 반대로 partial vision은 state가 완전히 보완합니다. 이 결과 때문에 cross-attention/gating을 자동으로 추가하지 않습니다.</p>
    </section>
  {/if}
</section>

<style>
.fusion-detail-wide{margin-top:10px;background:#fff;border:1px solid #e1e5ea;border-radius:16px;padding:18px 22px 22px;overflow:hidden}
.detail-head{display:flex;justify-content:space-between;gap:20px;padding-bottom:14px;border-bottom:1px solid #edf0f3}.eyebrow{font-size:10px;letter-spacing:.08em;color:#7d6aaa}.detail-head h2{font-size:20px;margin:2px 0 5px}.detail-head p{font-size:12px;color:#667085;margin:0;line-height:1.5}.detail-head button{width:36px;height:36px;border:1px solid #dfe3e8;border-radius:9px;background:#fff;font-size:20px;color:#667085;cursor:pointer}
.scenario-bar{display:flex;gap:5px;flex-wrap:wrap;padding:12px 0}.scenario-bar button{border:1px solid #dfe3e8;background:#fff;border-radius:6px;padding:6px 9px;font-size:9px;color:#687386;cursor:pointer}.scenario-bar button.active{background:#efeaf8;border-color:#a895cf;color:#604b95}
.token-pairs{display:grid;grid-template-columns:repeat(8,1fr);gap:6px;padding:4px 0 16px}.time-pair{display:flex;flex-direction:column;align-items:center;gap:4px}.time-pair>.time{font:8px ui-monospace,SFMono-Regular,Menlo,monospace;color:#7b8492}.time-pair button{display:flex;align-items:center;gap:4px;border:1px solid #e2e5ea;background:#fff;border-radius:6px;padding:4px;cursor:pointer}.time-pair button.active{background:#f1eef9;border-color:#aa9ad0}.time-pair b{font-size:7px;color:#687386}.vec{position:relative;width:18px;height:34px;border:1px solid #e0e3e8;border-radius:3px;overflow:hidden}
.fusion-grid{display:grid;grid-template-columns:1.25fr 1fr .8fr;gap:12px}.fusion-grid article{border:1px solid #e3e7ed;border-radius:10px;background:#fbfcfd;padding:14px}.fusion-grid h3,.ablation h3{font-size:12px;margin:0 0 10px;color:#505a69}.matrix{display:flex;justify-content:center}.legend{margin-top:8px;font-size:9px;color:#737d8b;text-align:center}.legend .s{color:#3c67b2;font-weight:800}.legend .v{color:#30805a;font-weight:800;margin-left:8px}
.bars{display:flex;flex-direction:column;gap:8px}.bars>div{position:relative;display:grid;grid-template-columns:1fr auto;gap:6px;padding-bottom:5px;border-bottom:3px solid #eceef2}.bars span{font-size:9px;color:#687386}.bars b{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:#604b95}.bars i{position:absolute;left:0;bottom:-3px;height:3px;background:#9d89c9;max-width:100%}.same-time{margin-top:10px;font-size:9px;color:#737d8b}.same-time b{color:#604b95}
.state-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.state-grid>div{padding:8px;border:1px solid #e3e6eb;border-radius:7px;background:#fff}.state-grid span,.state-grid small{display:block;font-size:8px;color:#86909d}.state-grid b{display:block;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563;margin:2px 0}
.ablation{margin-top:14px;padding:14px;border:1px solid #e3e7ed;border-radius:10px;background:#fcfcfd}.ablation-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.ablation-grid>div{padding:8px;border-radius:7px;background:#f7f8fa;border:1px solid #e6e9ed}.ablation-grid span{display:block;font-size:9px;color:#6d7684}.ablation-grid b{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4b5563}.ablation-grid .positive{background:#eff8f3}.ablation-grid .negative{background:#fff6ed}.ablation p{font-size:10px;line-height:1.5;color:#667085;margin:10px 0 0}
@media(max-width:980px){.fusion-detail-wide{padding:14px}.token-pairs{grid-template-columns:repeat(4,1fr)}.fusion-grid{grid-template-columns:1fr}.ablation-grid{grid-template-columns:1fr 1fr}.detail-head h2{font-size:18px}}
</style>
