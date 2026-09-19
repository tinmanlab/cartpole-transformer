import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const outDir = path.resolve('qa-output');
fs.mkdirSync(outDir, { recursive: true });
const baseURL = process.env.QA_URL || 'http://127.0.0.1:4173/cartpole-transformer/';

const report = {
  generatedAt: new Date().toISOString(),
  baseURL,
  errors: [],
  warnings: [],
  views: {},
  interactions: {},
};

const pushError = (msg) => report.errors.push(msg);
const pushWarning = (msg) => report.warnings.push(msg);

function rectData(r) {
  return r ? {
    x: Math.round(r.x), y: Math.round(r.y),
    width: Math.round(r.width), height: Math.round(r.height),
    right: Math.round(r.x + r.width), bottom: Math.round(r.y + r.height)
  } : null;
}

async function inspectView(page, name) {
  const data = await page.evaluate(() => {
    const pick = (sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom };
    };
    const stageSelectors = ['.embedding-overview','.qkv-overview','.attention-overview','.block-overview','.action-overview'];
    const stages = Object.fromEntries(stageSelectors.map(s => [s, pick(s)]));
    const overlaps = [];
    for (let i=0;i<stageSelectors.length;i++) {
      for (let j=i+1;j<stageSelectors.length;j++) {
        const a=stages[stageSelectors[i]], b=stages[stageSelectors[j]];
        if (!a || !b) continue;
        const ix=Math.max(0,Math.min(a.right,b.right)-Math.max(a.x,b.x));
        const iy=Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.y,b.y));
        if (ix*iy > 4) overlaps.push([stageSelectors[i],stageSelectors[j],Math.round(ix*iy)]);
      }
    }
    const paths = [...document.querySelectorAll('path.sankey-path')].map((p,i) => {
      let length = 0;
      try { length = p.getTotalLength(); } catch {}
      const r=p.getBoundingClientRect();
      return {i,length,rect:{x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom}};
    });
    const detail = document.querySelector('.transformer-detail-wide');
    const detailFonts = detail ? [...detail.querySelectorAll('b,span,small,p,code')].map(el=>parseFloat(getComputedStyle(el).fontSize)).filter(Number.isFinite) : [];
    return {
      viewport:{width:innerWidth,height:innerHeight},
      document:{scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight},
      lab:pick('.lab-grid'),
      sim:pick('.sim-card'),
      pipeline:pick('.pipeline-shell'),
      overview:pick('.overview'),
      detail:pick('.transformer-detail-wide'),
      stages,
      overlaps,
      paths,
      detailMinFont:detailFonts.length?Math.min(...detailFonts):null,
      learnedText:document.querySelector('.topbar span')?.textContent?.trim() || '',
      bodyText:(document.body.innerText || '').slice(0,5000)
    };
  });
  const clean = JSON.parse(JSON.stringify(data));
  for (const key of ['lab','sim','pipeline','overview','detail']) clean[key]=rectData(data[key]);
  for (const [k,v] of Object.entries(clean.stages)) clean.stages[k]=rectData(v);
  report.views[name]=clean;

  if (data.document.scrollWidth > data.viewport.width + 2) pushWarning(name+': page-level horizontal overflow '+data.document.scrollWidth+' > '+data.viewport.width);
  if (name === 'mobile-overview' && data.overview && data.overview.right > data.viewport.width + 2) pushError('mobile-overview: transformer overview escapes viewport');
  if (data.overlaps.length) pushWarning(name+': stage overlaps '+JSON.stringify(data.overlaps));
  if (data.paths.some(p=>!Number.isFinite(p.length)||p.length<8)) pushWarning(name+': zero/short Sankey path detected');
  if (data.detailMinFont !== null && data.detailMinFont < 9.5) pushError(name+': detail text too small, min '+data.detailMinFont+'px');
  if (name === 'desktop-overview' && data.paths.length > 10) pushError('desktop-overview: too many visible Sankey paths ('+data.paths.length+')');
  if (name === 'desktop-attention' && data.detail && data.detail.width < 1200) pushError('desktop-attention: full-width detail too narrow ('+Math.round(data.detail.width)+'px)');
  if (name === 'desktop-attention' && data.detail && data.detail.height < 260) pushError('desktop-attention: detail panel too short ('+Math.round(data.detail.height)+'px)');
  if (name === 'desktop-attention' && data.overview) {
    for (const [sel,rect] of Object.entries(data.stages)) {
      if (!rect) continue;
      if (rect.top < data.overview.y - 3 || rect.bottom > data.overview.bottom + 3) {
        pushError('desktop-attention: '+sel+' escapes overview bounds');
      }
    }
  }
  return data;
}

async function waitLearned(page) {
  await page.waitForFunction(() => {
    const t=document.querySelector('.topbar span')?.textContent || '';
    return t.includes('learned') || t.includes('fallback');
  }, null, { timeout: 15000 });
}

async function runDesktop(browser) {
  const page = await browser.newPage({ viewport:{width:1440,height:1000}, deviceScaleFactor:1 });
  const consoleErrors=[];
  page.on('console', msg=>{ if(msg.type()==='error') consoleErrors.push(msg.text()); });
  page.on('pageerror', err=>consoleErrors.push(String(err)));

  await page.goto(baseURL, { waitUntil:'networkidle', timeout:30000 });
  await waitLearned(page);
  await page.waitForTimeout(700);

  const t0=await page.locator('.time').innerText();
  const canvas0=await page.locator('.embedding-overview canvas').first().evaluate(el=>el.toDataURL());
  const matrix0=await page.locator('.attention-overview svg').first().evaluate(el=>el.outerHTML);
  await page.waitForTimeout(500);
  const t1=await page.locator('.time').innerText();
  const canvas1=await page.locator('.embedding-overview canvas').first().evaluate(el=>el.toDataURL());
  const matrix1=await page.locator('.attention-overview svg').first().evaluate(el=>el.outerHTML);
  report.interactions.live = { t0,t1, canvasChanged:canvas0!==canvas1, matrixChanged:matrix0!==matrix1 };
  if (t0===t1) pushError('simulation clock did not advance');
  if (canvas0===canvas1) pushWarning('embedding canvas did not visibly change over 500 ms');
  if (matrix0===matrix1) pushWarning('attention matrix DOM did not change over 500 ms');

  const overviewData=await inspectView(page,'desktop-overview');
  const bridgeCount=await page.locator('.lab-grid > .upstream-sankey path.sankey-path').count();
  report.interactions.bridge={count:bridgeCount};
  if (bridgeCount !== 0) pushError('cross-panel simulation→Transformer Sankey bridge should be absent, found '+bridgeCount);
  await page.screenshot({path:path.join(outDir,'desktop-overview.jpg'),type:'jpeg',quality:80,fullPage:true});

  const stageCases = [
    ['embedding','.embedding-overview','Embedding'],
    ['qkv','.qkv-overview','Q · K · V'],
    ['attention','.attention-overview','Self Attention'],
    ['block','.block-overview','Residual + MLP'],
    ['action','.action-overview','Action head'],
  ];
  report.interactions.expansions={};
  for (const [name,sel,needle] of stageCases) {
    await page.locator(sel).click();
    await page.waitForTimeout(250);
    const count=await page.locator('.transformer-detail-wide').count();
    const txt=count?await page.locator('.transformer-detail-wide').innerText():'';
    let liveChanged=null;
    if(count){
      const fp0=await page.locator('.transformer-detail-wide').evaluate(el=>[...el.querySelectorAll('canvas')].map(c=>c.toDataURL()).join('|')+'#'+[...el.querySelectorAll('svg')].map(s=>s.innerHTML).join('|')+'#'+el.innerText);
      await page.waitForTimeout(320);
      const fp1=await page.locator('.transformer-detail-wide').evaluate(el=>[...el.querySelectorAll('canvas')].map(c=>c.toDataURL()).join('|')+'#'+[...el.querySelectorAll('svg')].map(s=>s.innerHTML).join('|')+'#'+el.innerText);
      liveChanged=fp0!==fp1;
    }
    report.interactions.expansions[name]={detailCount:count,containsExpected:txt.includes(needle.split(' ')[0]),textLength:txt.length,liveChanged};
    if(!count) pushError(name+': click did not create detail panel');
    if(count && txt.length<30) pushError(name+': detail panel appears empty/too sparse');
    if(count && liveChanged===false) pushWarning(name+': expanded visualization did not change over 320 ms');
    const shotName='desktop-'+name;
    if(name==='attention') {
      await inspectView(page,'desktop-attention');

      const trace=page.locator('.attention-cell-trace');
      const traceCount=await trace.count();
      if(traceCount!==1) {
        pushError('attention trace missing or duplicated: '+traceCount);
      } else {
        const qButtons=trace.locator('.trace-query-button');
        const kButtons=trace.locator('.trace-key-button');
        const countButtons=await qButtons.count();
        const allowedRow=Math.max(0,countButtons-1);
        const allowedCol=Math.max(0,countButtons-3);

        await qButtons.nth(allowedRow).click();
        await kButtons.nth(allowedCol).click();
        await page.waitForTimeout(120);

        const allowed=await trace.evaluate(el=>{
          const d=el.dataset;
          return {
            row:Number(d.row),
            col:Number(d.col),
            dotSum:Number(d.dotSum),
            scale:Number(d.scale),
            score:Number(d.score),
            masked:d.masked==='true',
            exp:Number(d.exp),
            denominator:Number(d.denominator),
            weight:Number(d.weight),
            v0:Number(d.v0),
            contribution0:Number(d.contribution0)
          };
        });
        const close=(a,b,eps=1e-8)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=eps;
        if(!close(allowed.dotSum/allowed.scale,allowed.score)) pushError('attention trace score arithmetic mismatch');
        if(!close(allowed.exp/allowed.denominator,allowed.weight)) pushError('attention trace softmax arithmetic mismatch');
        if(!close(allowed.weight*allowed.v0,allowed.contribution0)) pushError('attention trace weighted-V arithmetic mismatch');
        if(allowed.masked) pushError('allowed attention trace unexpectedly masked');

        let maskedCase=null;
        if(countButtons>=6) {
          await qButtons.nth(2).click();
          await kButtons.nth(5).click();
          await page.waitForTimeout(80);
          maskedCase=await trace.evaluate(el=>({
            row:Number(el.dataset.row),
            col:Number(el.dataset.col),
            masked:el.dataset.masked==='true',
            weight:Number(el.dataset.weight)
          }));
          if(!maskedCase.masked) pushError('future attention trace did not report causal mask');
          if(Math.abs(maskedCase.weight)>1e-12) pushError('future attention trace weight is not zero');
        }

        report.interactions.attentionTrace={allowed,masked:maskedCase};

        await qButtons.nth(allowedRow).click();
        await kButtons.nth(allowedCol).click();
        await page.waitForTimeout(80);
        await trace.screenshot({path:path.join(outDir,'desktop-attention-trace.jpg'),type:'jpeg',quality:86});
      }
    }
    await page.screenshot({path:path.join(outDir,shotName+'.jpg'),type:'jpeg',quality:82,fullPage:true});
    // close by clicking same stage
    await page.locator(sel).click();
    await page.waitForTimeout(100);
  }

  // linked token selection
  const ghostBefore=await page.locator('.selected-history-label').count();
  await page.locator('.embedding-token').nth(2).hover();
  await page.waitForTimeout(120);
  const selectedText=await page.locator('.state-readout .selected-time').innerText();
  const ghostAfter=await page.locator('.selected-history-label').count();
  report.interactions.tokenLink={ghostBefore,ghostAfter,selectedText};
  if(!selectedText.includes('t−')) pushWarning('historical token selection did not surface in simulation readout');
  if(ghostAfter<1) pushWarning('selected historical ghost pose not visible');

  // disturbance while policy continues
  const xdot0=await page.locator('.state-readout span').nth(1).innerText();
  const push=page.getByRole('button',{name:'Push →'});
  await push.dispatchEvent('pointerdown');
  await page.waitForTimeout(300);
  await push.dispatchEvent('pointerup');
  await page.waitForTimeout(100);
  const xdot1=await page.locator('.state-readout span').nth(1).innerText();
  report.interactions.push={before:xdot0,after:xdot1,changed:xdot0!==xdot1};
  if(xdot0===xdot1) pushWarning('push interaction did not change visible xdot');

  // Optional pixels-only mode becomes mandatory once the trained artifact is present.
  const visionButton=page.getByRole('button',{name:'Vision'});
  const visionEnabled=await visionButton.isEnabled().catch(()=>false);
  report.interactions.visionMode={available:visionEnabled};

  if(visionEnabled){
    await visionButton.click();
    await page.getByRole('button',{name:'Reset'}).click();
    await page.waitForTimeout(450);

    const pipelineCount=await page.locator('.vision-pipeline').count();
    const hiddenStateCount=await page.locator('.vision-hidden-state').count();
    const stateReadoutCount=await page.locator('.state-readout').count();
    const frameCount=await page.locator('.vision-stage-frame .vision-frame').count();
    const patchGridCount=await page.locator('.vision-stage-patch .patch-grid').count();
    const patchCellCount=await page.locator('.vision-stage-patch .cell').count();

    if(pipelineCount!==1) pushError('vision mode: pipeline missing');
    if(hiddenStateCount!==1) pushError('vision mode: hidden-state label missing');
    if(stateReadoutCount!==0) pushError('vision mode: explicit state readout leaked into pixels-only mode');
    if(frameCount!==8) pushError('vision mode: expected 8 sampled frames, found '+frameCount);
    if(patchGridCount!==2) pushError('vision mode: expected patch and delta grids, found '+patchGridCount);
    if(patchCellCount!==512) pushError('vision mode: expected 512 visible patch+delta cells, found '+patchCellCount);

    const frameLabels=await page.locator('.vision-stage-frame .label').allInnerTexts();
    if(!frameLabels.some(x=>x.includes('420ms')) || !frameLabels.some(x=>x==='t')) {
      pushError('vision mode: frame labels do not expose 420ms temporal span');
    }

    const visionOutput0=await page.locator('.vision-stage-output').innerText();
    await page.waitForTimeout(350);
    const pushVision=page.getByRole('button',{name:'Push →'});
    await pushVision.dispatchEvent('pointerdown');
    await page.waitForTimeout(320);
    await pushVision.dispatchEvent('pointerup');
    await page.waitForTimeout(160);
    const visionOutput1=await page.locator('.vision-stage-output').innerText();
    if(visionOutput0===visionOutput1) pushWarning('vision mode: inferred motion/action did not visibly change after disturbance');

    await page.screenshot({path:path.join(outDir,'desktop-vision-overview.jpg'),type:'jpeg',quality:82,fullPage:true});

    // Re-enter a fresh stable visual episode before capturing the explanatory detail.
    await page.getByRole('button',{name:'Reset'}).click();
    await page.waitForTimeout(500);
    const pushForDetail=page.getByRole('button',{name:'Push →'});
    await pushForDetail.dispatchEvent('pointerdown');
    await page.waitForTimeout(140);
    await pushForDetail.dispatchEvent('pointerup');
    await page.waitForTimeout(100);

    await page.locator('.vision-stage-attention').click();
    await page.waitForTimeout(180);
    const visionDetailCount=await page.locator('.vision-detail-wide').count();
    const ablationCount=await page.locator('.vision-detail-wide .ablation').count();
    const truthCountBefore=await page.locator('.vision-detail-wide .truth').count();
    if(visionDetailCount!==1) pushError('vision mode: full-width detail missing');
    if(ablationCount!==1) pushError('vision mode: latest-frame ablation missing');
    if(truthCountBefore!==0) pushError('vision mode: ground truth should be hidden by default');

    await page.screenshot({path:path.join(outDir,'desktop-vision-detail.jpg'),type:'jpeg',quality:84,fullPage:true});

    const reveal=page.getByRole('button',{name:/Reveal ground-truth reference/});
    await reveal.click();
    await page.waitForTimeout(80);
    const truthCountAfter=await page.locator('.vision-detail-wide .truth').count();
    if(truthCountAfter!==1) pushError('vision mode: ground-truth teaching reference cannot be revealed');

    report.interactions.visionMode={
      available:true,pipelineCount,hiddenStateCount,stateReadoutCount,frameCount,patchGridCount,patchCellCount,
      outputChangedAfterPush:visionOutput0!==visionOutput1,
      detailCount:visionDetailCount,ablationCount,truthHiddenByDefault:truthCountBefore===0,truthRevealable:truthCountAfter===1
    };
    await page.screenshot({path:path.join(outDir,'desktop-vision-detail-truth.jpg'),type:'jpeg',quality:84,fullPage:true});
  }

  // Optional typed-token fusion mode becomes mandatory once its trained artifact is present.
  const fusionButton=page.getByRole('button',{name:'Fusion'});
  const fusionEnabled=await fusionButton.isEnabled().catch(()=>false);
  report.interactions.fusionMode={available:fusionEnabled};

  if(fusionEnabled){
    const timeBeforeFusion=parseFloat((await page.locator('.time').innerText()).replace(' s',''));
    await fusionButton.click();
    await page.waitForTimeout(260);

    const fusionModeAttr=await page.locator('main').getAttribute('data-observation-mode');
    const fusionPipelineCount=await page.locator('.fusion-pipeline').count();
    const fusionPairCount=await page.locator('.fusion-stage-observation .pair').count();
    const fusionTokenCount=await page.locator('.fusion-stage-token .token').count();
    const fusionAttentionCells=await page.locator('.fusion-stage-attention .cell').count();
    const fusionScenarioButtons=await page.locator('.fusion-pipeline .scenario-bar button').count();
    const fusionStateReadout=await page.locator('.state-readout').count();
    const timeAfterFusion=parseFloat((await page.locator('.time').innerText()).replace(' s',''));

    if(fusionModeAttr!=='fusion') pushError('fusion mode: main observation mode did not switch to fusion');
    if(fusionPipelineCount!==1) pushError('fusion mode: pipeline missing');
    if(fusionPairCount!==8) pushError('fusion mode: expected 8 aligned timestamp pairs, found '+fusionPairCount);
    if(fusionTokenCount!==16) pushError('fusion mode: expected 16 typed tokens, found '+fusionTokenCount);
    if(fusionAttentionCells!==256) pushError('fusion mode: expected 16x16 attention matrix, found '+fusionAttentionCells+' cells');
    if(fusionScenarioButtons!==5) pushError('fusion mode: expected 5 degradation controls, found '+fusionScenarioButtons);
    if(fusionStateReadout!==1) pushError('fusion mode: explicit state readout should be visible');
    if(!(timeAfterFusion>=timeBeforeFusion)) pushError('fusion mode: mode switch reset or rewound the episode clock');

    await page.screenshot({path:path.join(outDir,'desktop-fusion-overview.jpg'),type:'jpeg',quality:82,fullPage:true});

    await page.getByRole('button',{name:'State missing'}).click();
    await page.waitForTimeout(80);
    const missingStateOff=await page.locator('.fusion-stage-observation .state-chip.off').count();
    if(missingStateOff!==8) pushError('fusion mode: state-missing control did not disable 8 state observations');

    await page.getByRole('button',{name:'Partial vision'}).click();
    await page.waitForTimeout(80);
    const partialVision=await page.locator('.fusion-stage-observation .vision-chip.partial').count();
    if(partialVision!==8) pushError('fusion mode: partial-vision control did not mark 8 vision observations');

    await page.getByRole('button',{name:'Vision missing'}).click();
    await page.waitForTimeout(80);
    const missingVisionOff=await page.locator('.fusion-stage-observation .vision-chip.off').count();
    if(missingVisionOff!==8) pushError('fusion mode: vision-missing control did not disable 8 vision observations');

    await page.getByRole('button',{name:'Noisy state'}).click();
    await page.waitForTimeout(80);
    const noisyActive=await page.getByRole('button',{name:'Noisy state'}).evaluate(el=>el.classList.contains('active'));
    if(!noisyActive) pushError('fusion mode: noisy-state scenario did not activate');

    await page.getByRole('button',{name:'Clean'}).click();
    await page.waitForTimeout(80);

    await page.locator('.fusion-stage-attention').click();
    await page.waitForTimeout(160);
    const fusionDetailCount=await page.locator('.fusion-detail-wide').count();
    const fusionDetailCells=await page.locator('.fusion-detail-wide .attention-card .cell').count();
    const fusionAblations=await page.locator('.fusion-detail-wide .ablation-grid > div').count();
    const negativeResultText=await page.locator('.fusion-detail-wide .ablation').innerText().catch(()=>'');
    if(fusionDetailCount!==1) pushError('fusion mode: detail drawer missing');
    if(fusionDetailCells!==256) pushError('fusion mode: detail attention matrix is not 16x16');
    if(fusionAblations!==7) pushError('fusion mode: expected 7 ablation results, found '+fusionAblations);
    if(!negativeResultText.includes('Negative result retained')) pushError('fusion mode: noisy-state negative result is not explicitly retained');

    report.interactions.fusionMode={
      available:true,modeAttr:fusionModeAttr,pipelineCount:fusionPipelineCount,pairCount:fusionPairCount,
      tokenCount:fusionTokenCount,attentionCells:fusionAttentionCells,scenarioButtons:fusionScenarioButtons,
      stateReadoutCount:fusionStateReadout,sameEpisode:timeAfterFusion>=timeBeforeFusion,
      missingStateOff,partialVision,missingVisionOff,noisyActive,
      detailCount:fusionDetailCount,detailAttentionCells:fusionDetailCells,ablationCount:fusionAblations
    };
    await page.screenshot({path:path.join(outDir,'desktop-fusion-detail.jpg'),type:'jpeg',quality:84,fullPage:true});
  }

  report.interactions.consoleErrors=consoleErrors;
  if(consoleErrors.length) pushError('browser console/page errors: '+consoleErrors.join(' | '));
  await page.close();
}

async function runMobile(browser) {
  const page=await browser.newPage({viewport:{width:390,height:844},deviceScaleFactor:1});
  await page.goto(baseURL,{waitUntil:'networkidle',timeout:30000});
  await waitLearned(page);
  await page.waitForTimeout(500);
  const mobile=await inspectView(page,'mobile-overview');
  const mobileSankeyDisplay=await page.locator('.pipeline-shell .upstream-sankey').first().evaluate(el=>getComputedStyle(el).display).catch(()=>null);
  report.interactions.mobileSankeyDisplay=mobileSankeyDisplay;
  if(mobileSankeyDisplay && mobileSankeyDisplay!=='none') pushError('mobile-overview: internal Sankey must be hidden, display='+mobileSankeyDisplay);
  await page.screenshot({path:path.join(outDir,'mobile-overview.jpg'),type:'jpeg',quality:78,fullPage:true});

  await page.locator('.attention-overview').click();
  await page.waitForTimeout(180);
  const mobileDetailCount=await page.locator('.transformer-detail-wide').count();
  if(mobileDetailCount!==1) pushError('mobile-attention: detail drawer missing');
  const mobileAttention=await inspectView(page,'mobile-attention');
  if(mobileAttention.document.scrollWidth > mobileAttention.viewport.width + 2) {
    pushError('mobile-attention: page-level horizontal overflow '+mobileAttention.document.scrollWidth+' > '+mobileAttention.viewport.width);
  }
  const traceScroller=await page.locator('.attention-cell-trace').evaluate(el=>({
    scrollWidth:el.scrollWidth,
    clientWidth:el.clientWidth,
    overflowX:getComputedStyle(el).overflowX
  })).catch(()=>null);
  const traceFlowDirection=await page.locator('.attention-cell-trace .trace-flow').evaluate(el=>getComputedStyle(el).flexDirection).catch(()=>null);
  const attentionExpansionDirection=await page.locator('.attention-expansion').evaluate(el=>getComputedStyle(el).flexDirection).catch(()=>null);
  report.interactions.mobileAttention={detailCount:mobileDetailCount,traceScroller,traceFlowDirection,attentionExpansionDirection};
  if(traceScroller && traceScroller.scrollWidth>traceScroller.clientWidth+2) {
    pushError('mobile-attention: attention trace still requires horizontal scrolling');
  }
  if(traceFlowDirection!=='column') pushError('mobile-attention: arithmetic trace is not vertically stacked');
  if(attentionExpansionDirection!=='column') pushError('mobile-attention: QK/mask/softmax expansion is not vertically stacked');
  await page.screenshot({path:path.join(outDir,'mobile-attention.jpg'),type:'jpeg',quality:80,fullPage:true});

  // Close state detail, then exercise the pixels-only mobile flow.
  const closeState=page.getByRole('button',{name:'close Transformer detail'});
  if(await closeState.count()) await closeState.click();
  const mobileVisionButton=page.getByRole('button',{name:'Vision'});
  const mobileVisionEnabled=await mobileVisionButton.isEnabled().catch(()=>false);
  if(mobileVisionEnabled){
    await mobileVisionButton.click();
    await page.getByRole('button',{name:'Reset'}).click();
    await page.waitForTimeout(350);

    const mobileVisionStateReadout=await page.locator('.state-readout').count();
    const mobileVisionHidden=await page.locator('.vision-hidden-state').count();
    const mobileVisionFrames=await page.locator('.vision-stage-frame .vision-frame').count();
    const mobileVisionGrids=await page.locator('.vision-stage-patch .patch-grid').count();
    const mobileVisionSankeyDisplay=await page.locator('.vision-pipeline .upstream-sankey').first().evaluate(el=>getComputedStyle(el).display).catch(()=>null);
    if(mobileVisionStateReadout!==0) pushError('mobile vision: explicit state readout leaked');
    if(mobileVisionHidden!==1) pushError('mobile vision: hidden-state label missing');
    if(mobileVisionFrames!==8) pushError('mobile vision: expected 8 frames, found '+mobileVisionFrames);
    if(mobileVisionGrids!==2) pushError('mobile vision: expected patch and delta grids, found '+mobileVisionGrids);
    if(mobileVisionSankeyDisplay && mobileVisionSankeyDisplay!=='none') pushError('mobile vision: Sankey must be hidden, display='+mobileVisionSankeyDisplay);

    const mobileVisionDoc=await page.evaluate(()=>({w:document.documentElement.scrollWidth,v:innerWidth}));
    if(mobileVisionDoc.w>mobileVisionDoc.v+2) pushError('mobile vision: overview causes page-level horizontal overflow');
    await page.screenshot({path:path.join(outDir,'mobile-vision-overview.jpg'),type:'jpeg',quality:80,fullPage:true});

    await page.locator('.vision-stage-attention').click();
    await page.waitForTimeout(150);
    const mobileVisionDetail=await page.locator('.vision-detail-wide').count();
    if(mobileVisionDetail!==1) pushError('mobile vision: detail drawer missing');
    const mobileVisionDetailDoc=await page.evaluate(()=>({w:document.documentElement.scrollWidth,v:innerWidth}));
    if(mobileVisionDetailDoc.w>mobileVisionDetailDoc.v+2) pushError('mobile vision: detail causes page-level horizontal overflow');
    report.interactions.mobileVision={
      enabled:true,stateReadoutCount:mobileVisionStateReadout,hiddenStateCount:mobileVisionHidden,
      frameCount:mobileVisionFrames,patchGridCount:mobileVisionGrids,detailCount:mobileVisionDetail,
      sankeyDisplay:mobileVisionSankeyDisplay
    };
    await page.screenshot({path:path.join(outDir,'mobile-vision-detail.jpg'),type:'jpeg',quality:80,fullPage:true});
  } else {
    report.interactions.mobileVision={enabled:false};
    pushError('mobile vision: trained artifact exists but Vision button is disabled');
  }

  const mobileFusionButton=page.getByRole('button',{name:'Fusion'});
  const mobileFusionEnabled=await mobileFusionButton.isEnabled().catch(()=>false);
  if(mobileFusionEnabled){
    await mobileFusionButton.click();
    await page.waitForTimeout(220);

    const mobileFusionPipeline=await page.locator('.fusion-pipeline').count();
    const mobileFusionTokens=await page.locator('.fusion-stage-token .token').count();
    const mobileFusionCells=await page.locator('.fusion-stage-attention .cell').count();
    const mobileFusionSankeyDisplay=await page.locator('.fusion-pipeline .upstream-sankey').first().evaluate(el=>getComputedStyle(el).display).catch(()=>null);
    const mobileFusionDoc=await page.evaluate(()=>({w:document.documentElement.scrollWidth,v:innerWidth}));

    if(mobileFusionPipeline!==1) pushError('mobile fusion: pipeline missing');
    if(mobileFusionTokens!==16) pushError('mobile fusion: expected 16 typed tokens, found '+mobileFusionTokens);
    if(mobileFusionCells!==256) pushError('mobile fusion: expected 256 attention cells, found '+mobileFusionCells);
    if(mobileFusionSankeyDisplay && mobileFusionSankeyDisplay!=='none') pushError('mobile fusion: Sankey must be hidden, display='+mobileFusionSankeyDisplay);
    if(mobileFusionDoc.w>mobileFusionDoc.v+2) pushError('mobile fusion: overview causes page-level horizontal overflow');

    await page.screenshot({path:path.join(outDir,'mobile-fusion-overview.jpg'),type:'jpeg',quality:80,fullPage:true});

    await page.locator('.fusion-stage-attention').click();
    await page.waitForTimeout(150);
    const mobileFusionDetail=await page.locator('.fusion-detail-wide').count();
    const mobileFusionDetailDoc=await page.evaluate(()=>({w:document.documentElement.scrollWidth,v:innerWidth}));
    if(mobileFusionDetail!==1) pushError('mobile fusion: detail drawer missing');
    if(mobileFusionDetailDoc.w>mobileFusionDetailDoc.v+2) pushError('mobile fusion: detail causes page-level horizontal overflow');

    report.interactions.mobileFusion={
      enabled:true,pipelineCount:mobileFusionPipeline,tokenCount:mobileFusionTokens,
      attentionCells:mobileFusionCells,detailCount:mobileFusionDetail,sankeyDisplay:mobileFusionSankeyDisplay
    };
    await page.screenshot({path:path.join(outDir,'mobile-fusion-detail.jpg'),type:'jpeg',quality:80,fullPage:true});
  } else {
    report.interactions.mobileFusion={enabled:false};
  }

  await page.close();
}

let browser;
try {
  browser = await chromium.launch({ headless:true });
  await runDesktop(browser);
  await runMobile(browser);
} catch (error) {
  pushError('unhandled visual QA exception: ' + (error?.stack || String(error)));
} finally {
  if (browser) {
    try { await browser.close(); } catch (error) {
      pushWarning('browser close failed: ' + String(error));
    }
  }
  fs.writeFileSync(path.join(outDir,'report.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
}
if (report.errors.length) process.exitCode=1;
