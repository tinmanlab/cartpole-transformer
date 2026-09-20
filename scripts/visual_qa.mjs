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

async function readAtomicSnapshot(page) {
  return page.locator('main').evaluate(el => {
    const parse = value => value ? value.split(',').map(Number) : [];
    return {
      mode:el.dataset.observationMode,
      tick:Number(el.dataset.syncTick),
      sim:parse(el.dataset.simState),
      stateToken:parse(el.dataset.stateToken),
      visionState:parse(el.dataset.visionSampleState),
      actionScore:Number(el.dataset.activeActionScore),
      force:Number(el.dataset.controllerForce)
    };
  });
}

function verifyAtomicSnapshot(snapshot, label) {
  const close = (a,b,eps=1e-10) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a-b) <= eps;
  const arraysClose = (a,b) => a.length===4 && b.length===4 && a.every((v,i)=>close(v,b[i]));

  if(!arraysClose(snapshot.sim,snapshot.stateToken)) {
    pushError(label+': displayed simulator state != latest State raw token');
  }
  if(!arraysClose(snapshot.sim,snapshot.visionState)) {
    pushError(label+': displayed simulator state != newest Vision/Fusion sampled state');
  }
  if(!Number.isFinite(snapshot.actionScore) || !Number.isFinite(snapshot.force)) {
    pushError(label+': active action score/force is non-finite');
  } else {
    const expectedForce=10*Math.tanh(snapshot.actionScore);
    if(!close(expectedForce,snapshot.force,1e-8)) {
      pushError(label+': displayed force does not match current snapshot action score');
    }
  }
}

async function readDecisionTrace(page) {
  return page.locator('.decision-trace').evaluate(el=>{
    const parse=value=>value ? value.split(',').map(Number) : [];
    const d=el.dataset;
    return {
      hasTrace:d.hasTrace==='true',
      mode:d.mode,
      tickFrom:Number(d.tickFrom),
      tickTo:Number(d.tickTo),
      beforeState:parse(d.beforeState),
      appliedActionScore:Number(d.appliedActionScore),
      appliedPolicyForce:Number(d.appliedPolicyForce),
      appliedControl:Number(d.appliedControl),
      disturbance:Number(d.disturbance),
      totalForce:Number(d.totalForce),
      xAcc:Number(d.xAcc),
      thetaAcc:Number(d.thetaAcc),
      dt:Number(d.dt),
      nextState:parse(d.nextState),
      nextActionScore:Number(d.nextActionScore),
      nextPolicyForce:Number(d.nextPolicyForce)
    };
  });
}

function verifyDecisionTrace(trace,before,after,label) {
  const close=(a,b,eps=1e-9)=>Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=eps;
  const arraysClose=(a,b)=>a.length===4&&b.length===4&&a.every((v,i)=>close(v,b[i]));

  if(!trace.hasTrace) pushError(label+': decision trace missing');
  if(trace.mode!==before.mode) pushError(label+': decision trace mode mismatch');
  if(trace.tickFrom!==before.tick || trace.tickTo!==after.tick) pushError(label+': trace ticks do not match Step');
  if(!arraysClose(trace.beforeState,before.sim)) pushError(label+': trace beforeState mismatch');
  if(!arraysClose(trace.nextState,after.sim)) pushError(label+': trace nextState mismatch');
  if(!close(trace.appliedActionScore,before.actionScore)) pushError(label+': applied action score mismatch');
  if(!close(trace.appliedPolicyForce,before.force)) pushError(label+': applied policy force mismatch');
  if(!close(trace.nextActionScore,after.actionScore)) pushError(label+': next action score mismatch');
  if(!close(trace.nextPolicyForce,after.force)) pushError(label+': next policy force mismatch');
  if(!close(trace.totalForce,trace.appliedControl+trace.disturbance)) pushError(label+': policy + disturbance != total plant force');

  if(trace.beforeState.length===4 && trace.nextState.length===4){
    if(!close(trace.nextState[0],trace.beforeState[0]+trace.dt*trace.beforeState[1])) pushError(label+': x Euler integration mismatch');
    if(!close(trace.nextState[1],trace.beforeState[1]+trace.dt*trace.xAcc)) pushError(label+': xDot Euler integration mismatch');
    if(!close(trace.nextState[2],trace.beforeState[2]+trace.dt*trace.beforeState[3])) pushError(label+': theta Euler integration mismatch');
    if(!close(trace.nextState[3],trace.beforeState[3]+trace.dt*trace.thetaAcc)) pushError(label+': thetaDot Euler integration mismatch');
  }
}


async function waitAllModes(page) {
  await page.waitForFunction(() => {
    const wanted=['Vision','Fusion','Compare'];
    return wanted.every(name=>{
      const button=[...document.querySelectorAll('.mode-switch button')].find(b=>b.textContent.trim()===name);
      return button && !button.disabled;
    });
  }, null, { timeout:15000 });
}

async function auditLayout(page, name, { minFont=9.75 } = {}) {
  const data=await page.evaluate(({minFont})=>{
    const visible=el=>{
      const cs=getComputedStyle(el);
      const r=el.getBoundingClientRect();
      return cs.display!=='none' && cs.visibility!=='hidden' && Number(cs.opacity)!==0 && r.width>0 && r.height>0;
    };
    const scrollableAncestor=el=>{
      let p=el.parentElement;
      while(p && p!==document.body){
        const cs=getComputedStyle(p);
        if(['auto','scroll'].includes(cs.overflowX) && p.scrollWidth>p.clientWidth+1) return true;
        p=p.parentElement;
      }
      return false;
    };
    const cls=el=>typeof el.className==='string'?el.className:'';
    const textPreview=el=>{
      const direct=[...el.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent.trim()).filter(Boolean).join(' ');
      return direct.replace(/\s+/g,' ').slice(0,90);
    };

    const tinyText=[];
    const clippedText=[];
    const viewportEscapes=[];
    const smallControls=[];
    const tinySvgText=[];

    // SVG <text> is styled in local user units; when the viewBox (or any
    // ancestor <g transform>) scales down to fit a narrow container, declared
    // font-size no longer equals rendered size on screen. Measure the real
    // screen size via the text element's OWN screen-space transform
    // (getScreenCTM), which folds in every ancestor transform, not just the
    // root <svg>'s. This UI never rotates/skews text, so (c,d) is the
    // vertical (font-size) axis scale factor to use.
    for(const el of [...document.querySelectorAll('svg text')]){
      if(!visible(el)) continue;
      const text=(el.textContent||'').trim();
      if(!text) continue;
      const ctm=el.getScreenCTM();
      if(!ctm) continue;
      const scale=Math.hypot(ctm.c,ctm.d);
      const declared=parseFloat(getComputedStyle(el).fontSize);
      if(!Number.isFinite(declared)) continue;
      const effective=declared*scale;
      if(effective<minFont){
        tinySvgText.push({class:cls(el),text:text.slice(0,60),declared:Number(declared.toFixed(2)),effective:Number(effective.toFixed(2))});
      }
    }

    // Essential readouts and control labels have a stricter, curated 14px
    // floor (design contract), separate from the blanket ~10px minFont floor
    // above that still covers every other compact/eyebrow label in the app.
    const coreFontSelectors='.state-readout span,.action-readout span,.vision-hidden-state,.sim-controls button,'+
      '.steps b,.steps p,.claim,.qkv-key,.panel-head small,.stage-head>span,.stage-head>small,.attention-read,.attention-read strong,.force-value,'+
      '.fd-badge,.fd-stages button,.fd-content p,.fd-values span,.fd-values b,.fd-result-tick,.fd-ba-label,.fd-details summary,.follow-decision-entry,'+
      '.fd-repo-row,.fd-repo-row span,.fd-token-select button,.fd-key-select button,.fd-dim-select button,.fd-selector-label,.fd-weight-bar-label,.fd-weight-bar-value,'+
      '.fd-calc-step b,.fd-calc-step code,.fd-caveat,.fd-vector-details summary,.fd-open-detail';
    const coreTextTooSmall=[];
    for(const el of [...document.querySelectorAll(coreFontSelectors)]){
      if(!visible(el)) continue;
      const fs=parseFloat(getComputedStyle(el).fontSize);
      if(Number.isFinite(fs) && fs<14){
        coreTextTooSmall.push({tag:el.tagName.toLowerCase(),class:cls(el),text:textPreview(el)||el.textContent.trim().slice(0,60),fontSize:Number(fs.toFixed(2))});
      }
    }

    // Primary controls (Pause/Run, Step, Reset, Push) get a stricter 44px
    // touch-target floor, separate from the blanket 28px smallControls floor.
    const primaryControlSelectors='.sim-controls button';
    const undersizedPrimaryControls=[];
    for(const el of [...document.querySelectorAll(primaryControlSelectors)]){
      if(!visible(el)) continue;
      const r=el.getBoundingClientRect();
      if(r.height<44 || r.width<44){
        undersizedPrimaryControls.push({tag:el.tagName.toLowerCase(),class:cls(el),text:el.textContent.trim().slice(0,40),width:Number(r.width.toFixed(1)),height:Number(r.height.toFixed(1))});
      }
    }

    for(const el of [...document.querySelectorAll('body *')]){
      if(el.matches('script,style,noscript,template,svg,svg *,canvas')) continue;
      if(!visible(el)) continue;
      const cs=getComputedStyle(el);
      const r=el.getBoundingClientRect();
      const text=textPreview(el);

      if(text){
        const fs=parseFloat(cs.fontSize);
        if(Number.isFinite(fs) && fs<minFont){
          tinyText.push({tag:el.tagName.toLowerCase(),class:cls(el),text,fontSize:Number(fs.toFixed(2))});
        }

        const clippedX=el.clientWidth>0 && el.scrollWidth>el.clientWidth+1 && ['hidden','clip'].includes(cs.overflowX);
        const clippedY=el.clientHeight>0 && el.scrollHeight>el.clientHeight+1 && ['hidden','clip'].includes(cs.overflowY);
        if(clippedX || clippedY){
          clippedText.push({
            tag:el.tagName.toLowerCase(),class:cls(el),text,
            client:[el.clientWidth,el.clientHeight],scroll:[el.scrollWidth,el.scrollHeight],
            overflow:[cs.overflowX,cs.overflowY]
          });
        }
      }

      if(
        (el.matches('button,select') || (el.matches('input') && el.getAttribute('type')!=='range')) &&
        r.height<28 && !el.closest('.frame-history')
      ){
        smallControls.push({tag:el.tagName.toLowerCase(),class:cls(el),text:text||el.getAttribute('aria-label')||'',height:Number(r.height.toFixed(1))});
      }
    }

    const escapeSelectors='button,article,.stage,.controller-card,.decision-trace,.transformer-detail-wide,.vision-detail-wide,.fusion-detail-wide,.comparison-lab';
    for(const el of [...document.querySelectorAll(escapeSelectors)]){
      if(!visible(el) || scrollableAncestor(el)) continue;
      const r=el.getBoundingClientRect();
      if(r.left<-1 || r.right>innerWidth+1){
        viewportEscapes.push({tag:el.tagName.toLowerCase(),class:cls(el),left:Number(r.left.toFixed(1)),right:Number(r.right.toFixed(1)),width:Number(r.width.toFixed(1))});
      }
    }

    const clippedContainers=[];
    const containerSelectors=['.pipeline-shell','.vision-pipeline','.fusion-pipeline','.comparison-lab','.decision-trace','.transformer-detail-wide','.vision-detail-wide','.fusion-detail-wide'];
    for(const sel of containerSelectors){
      for(const el of [...document.querySelectorAll(sel)]){
        if(!visible(el)) continue;
        const cs=getComputedStyle(el);
        const clipX=['hidden','clip'].includes(cs.overflowX) && el.scrollWidth>el.clientWidth+2;
        const clipY=['hidden','clip'].includes(cs.overflowY) && el.scrollHeight>el.clientHeight+2;
        if(clipX||clipY){
          clippedContainers.push({selector:sel,class:cls(el),client:[el.clientWidth,el.clientHeight],scroll:[el.scrollWidth,el.scrollHeight],overflow:[cs.overflowX,cs.overflowY]});
        }
      }
    }

    const overlaps=[];
    const groups=[
      '.topbar','.sim-controls','.state-readout','.overview','.vision-overview','.fusion-overview',
      '.decision-trace .flow','.steps','.controller-grid','.replay-controls',
      '.fusion-grid','.ablation-grid','.vision-flow','.token-pairs'
    ];
    for(const sel of groups){
      const group=document.querySelector(sel);
      if(!group || !visible(group)) continue;
      const children=[...group.children].filter(el=>{
        if(!visible(el)) return false;
        const pos=getComputedStyle(el).position;
        return pos!=='absolute' && pos!=='fixed' && !el.classList.contains('upstream-sankey');
      });
      for(let i=0;i<children.length;i++){
        const a=children[i].getBoundingClientRect();
        for(let j=i+1;j<children.length;j++){
          const b=children[j].getBoundingClientRect();
          const ix=Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left));
          const iy=Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));
          if(ix*iy>4){
            overlaps.push({
              group:sel,
              a:cls(children[i])||children[i].tagName.toLowerCase(),
              b:cls(children[j])||children[j].tagName.toLowerCase(),
              area:Number((ix*iy).toFixed(1))
            });
          }
        }
      }
    }

    return {
      viewport:{width:innerWidth,height:innerHeight},
      document:{scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight},
      tinyText,
      clippedText,
      tinySvgText,
      coreTextTooSmall,
      undersizedPrimaryControls,
      smallControls,
      viewportEscapes,
      clippedContainers,
      overlaps
    };
  },{minFont});

  report.layouts ??= {};
  report.layouts[name]=data;

  if(data.document.scrollWidth>data.viewport.width+2) pushError(name+': page-level horizontal overflow '+data.document.scrollWidth+' > '+data.viewport.width);
  if(data.tinyText.length) pushError(name+': text below 10px '+JSON.stringify(data.tinyText.slice(0,12)));
  if(data.tinySvgText.length) pushError(name+': SVG text renders below 10px on screen (effective size, not declared) '+JSON.stringify(data.tinySvgText.slice(0,12)));
  if(data.coreTextTooSmall.length) pushError(name+': essential readout/control text below 14px '+JSON.stringify(data.coreTextTooSmall.slice(0,12)));
  if(data.undersizedPrimaryControls.length) pushError(name+': primary control smaller than 44px touch target '+JSON.stringify(data.undersizedPrimaryControls.slice(0,12)));
  if(data.clippedText.length) pushError(name+': clipped visible text '+JSON.stringify(data.clippedText.slice(0,10)));
  if(data.smallControls.length) pushError(name+': controls shorter than 28px '+JSON.stringify(data.smallControls.slice(0,10)));
  if(data.viewportEscapes.length) pushError(name+': elements escape viewport '+JSON.stringify(data.viewportEscapes.slice(0,10)));
  if(data.clippedContainers.length) pushError(name+': clipped major containers '+JSON.stringify(data.clippedContainers.slice(0,8)));
  if(data.overlaps.length) pushError(name+': unintended sibling overlaps '+JSON.stringify(data.overlaps.slice(0,10)));
  return data;
}

async function runResponsiveLayoutAudit(browser, width, label) {
  const page=await browser.newPage({viewport:{width,height:900},deviceScaleFactor:1});
  const errors=[];
  page.on('console',msg=>{if(msg.type()==='error') errors.push(msg.text());});
  page.on('pageerror',err=>errors.push(String(err)));
  await page.goto(baseURL,{waitUntil:'networkidle',timeout:30000});
  await waitLearned(page);
  await waitAllModes(page);
  await page.waitForTimeout(250);

  const screenshot=async name=>{
    if(width===1024 || width===768){
      await page.screenshot({path:path.join(outDir,label+'-'+name+'.jpg'),type:'jpeg',quality:74,fullPage:true});
    }
  };

  const pause=page.getByRole('button',{name:'Pause'});
  if(await pause.count()) await pause.click();

  // Precondition for the essential-text-floor checks on .steps/.qkv-key
  // below: those now live inside a closed-by-default reference <details>
  // (F1 reflow), so open it once up front and leave it open for every
  // auditLayout call in this page session (state/vision/fusion/compare all
  // reuse the same persistent element) rather than losing that coverage.
  const explainSummary=page.locator('.explain-disclosure summary');
  if(await explainSummary.count()) await explainSummary.click();

  await auditLayout(page,label+'-state-overview');

  // F1: the primary guide entry must sit ahead of the secondary Pipeline
  // overview in the DOM (and thus cannot be displaced by it), on every
  // audited width.
  const entryPipelineOrder=await page.evaluate(()=>{
    const entry=document.querySelector('.follow-decision-entry');
    const pipeline=document.querySelector('.pipeline-shell');
    if(!entry || !pipeline) return null;
    return !!(entry.compareDocumentPosition(pipeline) & Node.DOCUMENT_POSITION_FOLLOWING);
  });
  if(!entryPipelineOrder) pushError(label+': follow-decision entry is not positioned before the Pipeline overview in the DOM');

  // Follow-one-decision guide: exercise the self-contained Calculation trace
  // (weight bars, key selector, numeric steps) at every audited width so it
  // never relies on a wide-only layout, and keep a representative screenshot
  // at each width including the narrowest (320) and widest (1440) targets.
  const followBtn=page.getByRole('button',{name:'한 판단 따라가기 · Follow one decision'});
  if(await followBtn.isEnabled().catch(()=>false)){
    await followBtn.click();
    await page.waitForTimeout(100);
    await auditLayout(page,label+'-follow-decision-input');
    await page.getByRole('tab',{name:/Calculation/}).click();
    await page.waitForTimeout(120);
    await auditLayout(page,label+'-follow-decision-calculation');
    const auditAdvancedOpen=await page.locator('.transformer-detail-wide').count();
    if(auditAdvancedOpen!==0) pushError(label+': advanced detail drawer auto-opened on Calculation entry (should stay closed until explicitly opened)');
    const auditVectorDetailsOpen=await page.locator('.follow-decision-guide .fd-vector-details').evaluateAll(els=>els.some(el=>el.open)).catch(()=>false);
    if(auditVectorDetailsOpen) pushError(label+': full vector details are expanded by default in the Calculation screenshot (should be collapsed)');
    await page.screenshot({path:path.join(outDir,label+'-follow-decision.jpg'),type:'jpeg',quality:74,fullPage:true});
    await page.getByRole('tab',{name:/Action/}).click();
    await page.waitForTimeout(120);
    await auditLayout(page,label+'-follow-decision-action');
    const auditActionAdvancedOpen=await page.locator('.transformer-detail-wide').count();
    if(auditActionAdvancedOpen!==0) pushError(label+': advanced detail drawer auto-opened on Action entry (should stay closed until explicitly opened)');
    const auditActionVectorDetailsOpen=await page.locator('.follow-decision-guide .fd-vector-details').evaluateAll(els=>els.some(el=>el.open)).catch(()=>false);
    if(auditActionVectorDetailsOpen) pushError(label+': full action-product details are expanded by default in the Action screenshot (should be collapsed)');

    // Per-tile numeric readout check: each Action tile's actual value text
    // (not just its presence in the DOM/dataset) must render as one complete,
    // single-line numeric token inside its own tile, never clipped, never
    // colliding with its own label, and never bleeding into a neighboring
    // tile — at whatever precision the component actually displays.
    const actionTiles=await page.locator('.follow-decision-guide .fd-action-values span').evaluateAll(spans=>spans.map(span=>{
      const tile=span.getBoundingClientRect();
      const labelEl=span.querySelector('b');
      const valueEl=span.querySelector('output');
      const labelRect=labelEl?labelEl.getBoundingClientRect():null;
      let valueRects=[];
      if(valueEl){
        const range=document.createRange();
        range.selectNodeContents(valueEl);
        valueRects=Array.from(range.getClientRects());
      }
      const cs=valueEl?getComputedStyle(valueEl):null;
      const r=rect=>rect&&{left:rect.left,right:rect.right,top:rect.top,bottom:rect.bottom};
      return {
        labelText:labelEl?labelEl.textContent.trim():'',
        valueText:valueEl?valueEl.textContent.trim():'',
        tile:r(tile),
        label:r(labelRect),
        value:valueRects.map(r),
        fontSize:cs?parseFloat(cs.fontSize):0
      };
    }));
    if(actionTiles.length!==4) pushError(label+': Action stage does not render exactly 4 readout tiles, found '+actionTiles.length);
    const intersects=(a,b)=>a && b && a.left<b.right-0.5 && a.right>b.left+0.5 && a.top<b.bottom-0.5 && a.bottom>b.top+0.5;
    for(const t of actionTiles){
      if(!t.valueText){ pushError(label+': Action tile "'+t.labelText+'" has an empty numeric value'); continue; }
      if(t.value.length!==1){
        pushError(label+': Action tile "'+t.labelText+'" numeric value "'+t.valueText+'" does not render as one line (client rects: '+t.value.length+')');
        continue;
      }
      const v=t.value[0];
      if(t.fontSize<14) pushError(label+': Action tile "'+t.labelText+'" numeric text is smaller than 14px ('+t.fontSize.toFixed(1)+'px)');
      if(v.left<t.tile.left-0.5 || v.right>t.tile.right+0.5 || v.bottom>t.tile.bottom+0.5){
        pushError(label+': Action tile "'+t.labelText+'" numeric value escapes its own tile bounds '+JSON.stringify({tile:t.tile,value:v}));
      }
      if(intersects(v,t.label)){
        pushError(label+': Action tile "'+t.labelText+'" numeric value overlaps its own label');
      }
      for(const other of actionTiles){
        if(other===t) continue;
        const ownTile=other.tile.left>=t.tile.left-0.5 && other.tile.right<=t.tile.right+0.5 && other.tile.top>=t.tile.top-0.5 && other.tile.bottom<=t.tile.bottom+0.5;
        if(!ownTile && intersects(v,other.tile)){
          pushError(label+': Action tile "'+t.labelText+'" numeric value overlaps neighboring tile "'+other.labelText+'"');
        }
      }
    }
    const tileByLabel=text=>actionTiles.find(t=>t.labelText.includes(text));
    const scoreTile=tileByLabel('action score');
    const forceTile=tileByLabel('10·tanh(score)');
    const commandTile=tileByLabel('force command');
    const tickTile=tileByLabel('captured tick');
    if(!scoreTile || !forceTile || !commandTile || !tickTile){
      pushError(label+': Action stage is missing one of the 4 expected readout tiles '+JSON.stringify(actionTiles.map(t=>t.labelText)));
    } else {
      // Derive the expected score/force from the full-precision native
      // arithmetic already exposed on the Action-head detail step
      // (data-action-score-check = actionProductsSum + actionBias, the same
      // un-rounded number the component feeds into forceFromScore), rather
      // than re-deriving force from the tile's own already-rounded 6dp
      // display text — that would fail near a rounding boundary even when
      // both displayed values are individually correct.
      const trueScoreStep=page.locator('.follow-decision-guide .fd-calc-step[data-action-score-check]');
      const trueScoreAttr=(await trueScoreStep.count())?await trueScoreStep.getAttribute('data-action-score-check'):null;
      if(trueScoreAttr!==null){
        const trueScore=Number(trueScoreAttr);
        const expectedScoreText=trueScore.toFixed(6);
        const expectedForceText=(10*Math.tanh(trueScore)).toFixed(2);
        if(scoreTile.valueText!==expectedScoreText){
          pushError(label+': Action tile "action score" shows '+scoreTile.valueText+' but the true native score is '+expectedScoreText);
        }
        if(forceTile.valueText!==expectedForceText){
          pushError(label+': Action tile "10·tanh(score) [N]" shows '+forceTile.valueText+' but the true native score '+expectedScoreText+' implies '+expectedForceText);
        }
        if(commandTile.valueText!==expectedForceText){
          pushError(label+': Action tile "force command [N]" shows '+commandTile.valueText+' but the true native score '+expectedScoreText+' implies '+expectedForceText);
        }
      }
      if(!Number.isFinite(Number(tickTile.valueText))) pushError(label+': Action tile "captured tick" is not a readable number, got '+tickTile.valueText);
      if(!Number.isFinite(Number(scoreTile.valueText))) pushError(label+': Action tile "action score" is not a readable number, got '+scoreTile.valueText);
      if(!Number.isFinite(Number(forceTile.valueText))) pushError(label+': Action tile "10·tanh(score) [N]" is not a readable number, got '+forceTile.valueText);
      if(!Number.isFinite(Number(commandTile.valueText))) pushError(label+': Action tile "force command [N]" is not a readable number, got '+commandTile.valueText);
    }

    await page.screenshot({path:path.join(outDir,label+'-follow-decision-action.jpg'),type:'jpeg',quality:74,fullPage:true});
    const closeGuide=page.getByRole('button',{name:'close follow-one-decision guide'});
    if(await closeGuide.count()) await closeGuide.click();
    await page.waitForTimeout(80);
  }

  for(const [name,sel] of [
    ['embedding','.embedding-overview'],
    ['qkv','.qkv-overview'],
    ['attention','.attention-overview'],
    ['block','.block-overview'],
    ['action','.action-overview']
  ]){
    await page.locator(sel).click();
    await page.waitForTimeout(100);
    await auditLayout(page,label+'-state-'+name+'-detail');
    if(name==='qkv') await screenshot('state-qkv');
    await page.locator(sel).click();
    await page.waitForTimeout(50);
  }

  const step=page.getByRole('button',{name:'Step'});
  if(await step.isEnabled().catch(()=>false)) await step.click();
  await page.waitForTimeout(70);

  // Precondition: DecisionTrace is now a closed-by-default disclosure in
  // State mode (F1 reflow) — open it so the existing font/overflow checks
  // below still exercise its real rendered content instead of silently
  // skipping a hidden subtree.
  const decisionTraceSummary=page.locator('.decision-trace-disclosure summary');
  if(await decisionTraceSummary.count()) await decisionTraceSummary.click();
  await page.waitForTimeout(60);
  await auditLayout(page,label+'-decision-trace');

  await page.getByRole('button',{name:'Vision'}).click();
  await page.waitForTimeout(120);
  await auditLayout(page,label+'-vision-overview');
  await page.locator('.vision-stage-attention').click();
  await page.waitForTimeout(100);
  await auditLayout(page,label+'-vision-detail');
  await screenshot('vision-detail');
  const closeVision=page.getByRole('button',{name:'close vision detail'});
  if(await closeVision.count()) await closeVision.click();

  await page.getByRole('button',{name:'Fusion'}).click();
  await page.waitForTimeout(120);
  await auditLayout(page,label+'-fusion-overview');
  await page.locator('.fusion-stage-attention').click();
  await page.waitForTimeout(100);
  await auditLayout(page,label+'-fusion-detail');
  await screenshot('fusion-detail');
  const closeFusion=page.getByRole('button',{name:'close fusion detail'});
  if(await closeFusion.count()) await closeFusion.click();

  await page.getByRole('button',{name:'Compare'}).click();
  await page.waitForTimeout(120);
  await auditLayout(page,label+'-compare');
  await screenshot('compare');

  if(errors.length) pushError(label+': console/page errors during layout sweep '+errors.join(' | '));
  await page.close();
}

async function runDesktop(browser) {
  const page = await browser.newPage({ viewport:{width:1440,height:1000}, deviceScaleFactor:1 });
  const consoleErrors=[];
  page.on('console', msg=>{ if(msg.type()==='error') consoleErrors.push(msg.text()); });
  page.on('pageerror', err=>consoleErrors.push(String(err)));

  await page.goto(baseURL, { waitUntil:'networkidle', timeout:30000 });
  await waitLearned(page);
  await page.waitForTimeout(700);

  const stateActionReadouts=await page.locator('.action-readout span').count();
  if(stateActionReadouts!==2) pushError('state mode: expected exactly 2 action-readout cells (force+push), found '+stateActionReadouts);

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
        const liveSource=await trace.getAttribute('data-source');
        if(liveSource!=='live') pushError('attention trace: default (outside Follow-one-decision) source is not "live", got '+JSON.stringify(liveSource));
        const liveEyebrow=await trace.locator('.eyebrow').innerText();
        if(!liveEyebrow.includes('LIVE')) pushError('attention trace: default eyebrow does not say LIVE arithmetic, got '+JSON.stringify(liveEyebrow));

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

        // Block/context provenance regression (r != c, already true here:
        // allowedRow=last, allowedCol=last-2): attention output and the
        // following Block transform belong to the Query row -- Key only
        // controls which V contributed. Pipeline's Block-overview vector and
        // the attention-overview context vector must NOT change when only
        // Key changes with Query held fixed; the displayed attention weight
        // MUST change, proving the Key change actually took effect.
        // Sim must be frozen for this: the tick advancing between the
        // "before" and "after" reads would move context/hidden on its own
        // and make the equality assertions meaningless.
        const provenancePause=page.getByRole('button',{name:'Pause'});
        const pausedForProvenance=await provenancePause.count();
        if(pausedForProvenance) await provenancePause.click();
        await page.waitForTimeout(80);
        const blockTextBefore=await page.locator('.block-overview code').innerText();
        const contextCanvasBefore=await page.locator('.attention-overview .context-vector canvas').evaluate(el=>el.toDataURL());
        const weightBefore=await page.locator('.attention-overview .attention-read strong').innerText();
        const altKeyIndex=allowedCol===0?1:0;
        await kButtons.nth(altKeyIndex).click();
        await page.waitForTimeout(120);
        const weightAfter=await page.locator('.attention-overview .attention-read strong').innerText();
        if(weightAfter===weightBefore) pushError('block/context provenance: changing Key alone did not change the displayed attention weight -- test precondition invalid');
        const blockTextAfter=await page.locator('.block-overview code').innerText();
        if(blockTextAfter!==blockTextBefore) {
          pushError('block/context provenance: Pipeline Block-overview vector changed when only Key changed (Query fixed) -- Block must read the Query row, not Key '+JSON.stringify({blockTextBefore,blockTextAfter}));
        }
        const contextCanvasAfter=await page.locator('.attention-overview .context-vector canvas').evaluate(el=>el.toDataURL());
        if(contextCanvasAfter!==contextCanvasBefore) {
          pushError('block/context provenance: attention-overview context vector changed when only Key changed (Query fixed)');
        }

        // Full Block detail header/source must also label the Query row,
        // not Key. Dispatched (not Playwright .click()) so the realistic
        // mouse-travel path from the Key button up to this stage button
        // doesn't cross Pipeline's other hover-to-select elements
        // (embedding/qkv token rows, attention matrix cells) and silently
        // reselect Query/Key as a side effect before we read the header --
        // a real test-precondition hazard, not a source defect: confirmed
        // by isolating the same sequence with only the click method varied.
        await page.evaluate(()=>document.querySelector('.block-overview').click());
        await page.waitForTimeout(150);
        const blockEyebrow=await page.locator('.transformer-detail-wide .detail-head .eyebrow').innerText().catch(()=>'');
        if(!blockEyebrow.split('·').map(s=>s.trim()).includes('t')) {
          pushError('block/context provenance: full Block detail header does not label the Query row ("t"), got '+JSON.stringify(blockEyebrow));
        }
        await page.evaluate(()=>document.querySelector('.attention-overview').click());
        await page.waitForTimeout(150);

        // Restore the pre-check running state so later push/progression
        // tests still see the sim advancing as they expect.
        if(pausedForProvenance) await page.getByRole('button',{name:'Run'}).click();
        await page.waitForTimeout(80);

        // Returning to attention must preserve the original r/c/dim --
        // restore the Key the pre-existing code below expects, exactly like
        // the reset it already performs.
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

  // Atomic snapshot contract: pose, latest model observations and displayed action
  // must describe the same simulation tick.
  const pauseState=page.getByRole('button',{name:'Pause'});
  if(await pauseState.count()) await pauseState.click();
  await page.waitForTimeout(80);
  const stateSync0=await readAtomicSnapshot(page);
  verifyAtomicSnapshot(stateSync0,'state pause');

  const stepState=page.getByRole('button',{name:'Step'});
  const stateStepEnabled=await stepState.isEnabled().catch(()=>false);
  if(!stateStepEnabled) pushError('state pause: Step button is not enabled');

  const tracePush=page.getByRole('button',{name:'Push →'});
  if(stateStepEnabled){
    await tracePush.dispatchEvent('pointerdown');
    await stepState.click();
    await tracePush.dispatchEvent('pointerup');
  }
  await page.waitForTimeout(80);
  const stateSync1=await readAtomicSnapshot(page);
  verifyAtomicSnapshot(stateSync1,'state step');
  if(stateSync1.tick!==stateSync0.tick+1) pushError('state Step did not advance exactly one tick');

  const stateDecisionTrace=await readDecisionTrace(page);
  verifyDecisionTrace(stateDecisionTrace,stateSync0,stateSync1,'state decision trace');
  if(Math.abs(stateDecisionTrace.disturbance-6)>1e-9) pushError('state decision trace: expected +6 N disturbance during Step');

  report.interactions.atomicSync={
    state:{before:stateSync0,after:stateSync1,stepDelta:stateSync1.tick-stateSync0.tick}
  };
  report.interactions.decisionTrace={state:stateDecisionTrace};
  await page.screenshot({path:path.join(outDir,'desktop-decision-trace-closed.jpg'),type:'jpeg',quality:84,fullPage:true});

  // Reference case: open the secondary trace disclosure and capture it
  // expanded too, so both the default-collapsed and expanded states are on
  // record (not just the collapsed default).
  const desktopTraceSummary=page.locator('.decision-trace-disclosure summary');
  if(await desktopTraceSummary.count()) await desktopTraceSummary.click();
  await page.waitForTimeout(80);
  await page.screenshot({path:path.join(outDir,'desktop-decision-trace.jpg'),type:'jpeg',quality:84,fullPage:true});

  // "Follow one decision" guide (State/learned mode only): frozen event
  // snapshot must survive Apply-step without leaking live post-step data
  // back into the earlier stages, and must never apply a second step.
  await page.getByRole('button',{name:'Reset'}).click();
  await page.waitForTimeout(300);

  const readFdValues=async () => (await page.locator('.fd-values span').allInnerTexts()).map(t=>t.trim());

  const followEntry=page.getByRole('button',{name:'한 판단 따라가기 · Follow one decision'});
  await followEntry.click();
  await page.waitForTimeout(120);

  const guideCount0=await page.locator('.follow-decision-guide').count();
  if(guideCount0!==1) pushError('follow-decision guide: entry click did not open exactly one guide');

  // F1: while the guide owns the workspace, Pipeline must be an explicit
  // collapsed overview positioned after the guide (not just after it in the
  // DOM by chance, but also visually below it), and reopenable without
  // recapturing the event.
  const guidePipelineOnOpen=await page.evaluate(()=>{
    const guide=document.querySelector('.follow-decision-guide');
    const disclosure=document.querySelector('.pipeline-disclosure');
    if(!guide || !disclosure) return null;
    return {
      domGuideBeforePipeline: !!(guide.compareDocumentPosition(disclosure) & Node.DOCUMENT_POSITION_FOLLOWING),
      guideTop: guide.getBoundingClientRect().top,
      pipelineTop: disclosure.getBoundingClientRect().top,
      pipelineOpen: disclosure.open
    };
  });
  if(!guidePipelineOnOpen || !guidePipelineOnOpen.domGuideBeforePipeline) pushError('follow-decision guide: Pipeline is not positioned after the guide in the DOM');
  if(!guidePipelineOnOpen || !(guidePipelineOnOpen.guideTop < guidePipelineOnOpen.pipelineTop)) pushError('follow-decision guide: Pipeline does not render below the guide on screen');
  if(!guidePipelineOnOpen || guidePipelineOnOpen.pipelineOpen) pushError('follow-decision guide: Pipeline did not collapse to a closed overview while the guide is open');

  const capturedTickBeforeReopen=await page.locator('.fd-badge').innerText();
  await page.locator('.pipeline-disclosure summary').click();
  await page.waitForTimeout(80);
  const pipelineReopened=await page.locator('.pipeline-disclosure').evaluate(el=>el.open);
  if(!pipelineReopened) pushError('follow-decision guide: Pipeline overview did not reopen on manual toggle while the guide is open');
  const capturedTickAfterReopen=await page.locator('.fd-badge').innerText();
  if(capturedTickAfterReopen!==capturedTickBeforeReopen) pushError('follow-decision guide: manually reopening Pipeline recaptured/changed the frozen event');
  await page.locator('.pipeline-disclosure summary').click();
  await page.waitForTimeout(80);

  const runDisabledOnOpen=await page.getByRole('button',{name:/^(Pause|Run)$/}).isDisabled().catch(()=>false);
  const pushDisabledOnOpen=await page.getByRole('button',{name:'Push →'}).isDisabled().catch(()=>false);
  if(!runDisabledOnOpen || !pushDisabledOnOpen) pushError('follow-decision guide: Run/Push are not disabled while the guide owns the frozen event');

  const badgeText0=await page.locator('.fd-badge').innerText();
  const capturedTickMatch=badgeText0.match(/captured tick (\d+)/);
  const capturedTick=capturedTickMatch?Number(capturedTickMatch[1]):NaN;
  const tickAtOpen=Number(await page.locator('main').getAttribute('data-sync-tick'));
  if(!Number.isFinite(capturedTick) || capturedTick!==tickAtOpen) pushError('follow-decision guide: badge captured tick does not match plant tick at open');

  // F4: real tablist keyboard support (ArrowRight/Home/End) with coherent
  // selected/focus/tabpanel relationships, not just mouse clicks.
  const fdTabs=page.locator('.follow-decision-guide [role="tab"]');
  const fdTabCount=await fdTabs.count();
  await fdTabs.first().focus();
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(80);
  const afterArrowSelected=await fdTabs.nth(1).getAttribute('aria-selected');
  const afterArrowFocused=await fdTabs.nth(1).evaluate(el=>el===document.activeElement);
  if(afterArrowSelected!=='true' || !afterArrowFocused) {
    pushError('follow-decision guide: ArrowRight on tablist did not select+focus the next tab '+JSON.stringify({afterArrowSelected,afterArrowFocused}));
  }
  const panelLabelledBy=await page.locator('.follow-decision-guide [role="tabpanel"]').getAttribute('aria-labelledby');
  if(panelLabelledBy!=='fd-tab-calculation') pushError('follow-decision guide: tabpanel aria-labelledby did not follow the ArrowRight-selected tab, got '+JSON.stringify(panelLabelledBy));
  await page.keyboard.press('End');
  await page.waitForTimeout(80);
  const afterEndSelected=await fdTabs.nth(fdTabCount-1).getAttribute('aria-selected');
  if(afterEndSelected!=='true') pushError('follow-decision guide: End on tablist did not select the last tab');
  await page.keyboard.press('Home');
  await page.waitForTimeout(80);
  const afterHomeSelected=await fdTabs.nth(0).getAttribute('aria-selected');
  if(afterHomeSelected!=='true') pushError('follow-decision guide: Home on tablist did not select the first (Input) tab');

  // F4: compact repo-purpose row, not a new large header.
  const repoRowText=await page.locator('.follow-decision-guide .fd-repo-row').innerText().catch(()=>'');
  if(!/PPO/.test(repoRowText) || !/Transformer/.test(repoRowText) || !/Diffusion/.test(repoRowText)) {
    pushError('follow-decision guide: missing compact repo-purpose row naming PPO/Transformer/DiffusionPolicy, got '+JSON.stringify(repoRowText));
  }

  const entryInputValues=await readFdValues();

  // F2: Input must expose actual history, not just the latest tick — picking
  // an earlier history token changes the readout, and returning to the
  // latest token (the identity used everywhere else) restores it exactly.
  const inputTokenButtons=page.locator('.fd-token-select button');
  const inputTokenCount=await inputTokenButtons.count();
  if(inputTokenCount<2) pushError('follow-decision guide: Input history selector exposes fewer than 2 tokens, got '+inputTokenCount);
  await inputTokenButtons.nth(0).click();
  await page.waitForTimeout(80);
  const earlyInputValues=await readFdValues();
  if(JSON.stringify(earlyInputValues)===JSON.stringify(entryInputValues)) {
    pushWarning('follow-decision guide: selecting the earliest history token produced identical values to the latest token');
  }
  await inputTokenButtons.nth(inputTokenCount-1).click();
  await page.waitForTimeout(80);
  const restoredInputValues=await readFdValues();
  if(JSON.stringify(restoredInputValues)!==JSON.stringify(entryInputValues)) {
    pushError('follow-decision guide: Input history identity did not restore after reselecting the latest token '+JSON.stringify({expected:entryInputValues,got:restoredInputValues}));
  }

  await page.getByRole('tab',{name:/Calculation/}).click();
  await page.waitForTimeout(150);

  // Precondition: entering Calculation must NOT auto-open the large advanced
  // detail drawer — that recreates the exact detached/vertical-overload
  // problem this guide fixes. The guide's own arithmetic below must already
  // be fully readable with the drawer closed.
  const advancedOpenOnCalcEntry=await page.locator('.transformer-detail-wide').count();
  if(advancedOpenOnCalcEntry!==0) pushError('follow-decision guide: Calculation stage auto-opened the advanced detail drawer on entry (should stay closed until explicitly opened)');

  // F1: the guide itself must show a self-contained, readable numeric trace
  // for the selected key/dimension (query fixed to the latest/controller
  // token) — not just prose pointing at the drawer above. Full per-dimension
  // vectors are collapsed by default (native <details>), so the emptiness/
  // change checks below read textContent (unaffected by collapse) rather
  // than innerText (which only reflects rendered/visible text).
  const calcStepCodeText=async () => page.locator('.follow-decision-guide .fd-calc-step > code').evaluateAll(els=>els.map(el=>el.textContent.trim()));
  const fdKeyButtons=page.locator('.follow-decision-guide .fd-key-select button');
  const fdKeyCount=await fdKeyButtons.count();
  if(fdKeyCount<2) pushError('follow-decision guide: Calculation key selector exposes fewer than 2 keys, got '+fdKeyCount);
  const fdDimButtons=page.locator('.follow-decision-guide .fd-dim-select button');
  const fdDimCount=await fdDimButtons.count();
  if(fdDimCount<2) pushError('follow-decision guide: Calculation value-dimension selector exposes fewer than 2 dimensions, got '+fdDimCount);
  const fdBarCount=await page.locator('.follow-decision-guide .fd-weight-bar').count();
  if(fdBarCount!==fdKeyCount) pushError('follow-decision guide: weight-bar count does not match key count '+JSON.stringify({bars:fdBarCount,keys:fdKeyCount}));

  const vectorDetailsOpenByDefault=await page.locator('.follow-decision-guide .fd-vector-details').evaluateAll(els=>els.some(el=>el.open));
  if(vectorDetailsOpenByDefault) pushError('follow-decision guide: full per-dimension vector/product details are expanded by default (should be collapsed native <details>)');

  const calcStepTextsBefore=await calcStepCodeText();
  if(calcStepTextsBefore.some(t=>!t)) pushError('follow-decision guide: a Calculation step shows no numeric value');
  const entryCommandForce=await page.locator('main').getAttribute('data-controller-force');

  await fdKeyButtons.nth(0).click();
  await page.waitForTimeout(80);
  const calcStepTextsAfterKey=await calcStepCodeText();
  if(JSON.stringify(calcStepTextsAfterKey)===JSON.stringify(calcStepTextsBefore) && fdKeyCount>1) {
    pushError('follow-decision guide: selecting a different key token did not change the Calculation numeric trace');
  }
  const badgeAfterKeyChange=await page.locator('.fd-badge').innerText();
  if(badgeAfterKeyChange!==badgeText0) pushError('follow-decision guide: selecting a different key token changed the captured-event badge (frozen identity violated)');

  await fdDimButtons.nth(fdDimCount-1).click();
  await page.waitForTimeout(80);
  const calcStepTextsAfterDim=await calcStepCodeText();
  if(JSON.stringify(calcStepTextsAfterDim)===JSON.stringify(calcStepTextsAfterKey) && fdDimCount>1) {
    pushError('follow-decision guide: selecting a different value dimension did not change the Calculation numeric trace');
  }
  const badgeAfterDimChange=await page.locator('.fd-badge').innerText();
  if(badgeAfterDimChange!==badgeText0) pushError('follow-decision guide: selecting a different value dimension changed the captured-event badge (frozen identity violated)');
  const commandForceAfterSelectors=await page.locator('main').getAttribute('data-controller-force');
  if(Number(commandForceAfterSelectors)!==Number(entryCommandForce)) {
    pushError('follow-decision guide: selecting key/dimension changed the commanded force (frozen identity violated) '+JSON.stringify({entryCommandForce,commandForceAfterSelectors}));
  }

  // Exact-value check: weight * V[key][dim] must equal the displayed
  // contribution[dim] at display precision, read from data-* attributes
  // exposed on the same frozen tensors (no recomputation of the model).
  const contributionStep=page.locator('.follow-decision-guide .fd-calc-step[data-contribution-dim]');
  const {weightAttr,vDimAttr,contribAttr}=await contributionStep.evaluate(el=>({
    weightAttr:Number(el.dataset.weight),
    vDimAttr:Number(el.dataset.vDim),
    contribAttr:Number(el.dataset.contributionDim)
  }));
  if(!Number.isFinite(weightAttr) || !Number.isFinite(vDimAttr) || !Number.isFinite(contribAttr) || Math.abs(weightAttr*vDimAttr-contribAttr)>1e-8) {
    pushError('follow-decision guide: displayed weight*V[dim] does not equal displayed contribution[dim] '+JSON.stringify({weightAttr,vDimAttr,contribAttr}));
  }

  await fdKeyButtons.nth(fdKeyCount-1).click();
  await fdDimButtons.nth(0).click();
  await page.waitForTimeout(80);

  // Explicit full-detail open (Calculation): the guide's own arithmetic
  // above was already fully readable without this — this only tests the
  // advanced-drawer path, which must exist behind one 44px button.
  const openCalcDetailButton=page.getByRole('button',{name:/open full Self Attention detail/});
  if(await openCalcDetailButton.count()!==1) pushError('follow-decision guide: Calculation stage is missing the explicit full-detail open button');
  await openCalcDetailButton.click();
  await page.waitForTimeout(150);
  const calcDetailHeading=await page.locator('.transformer-detail-wide h2').innerText().catch(()=>'');
  if(calcDetailHeading!=='Self Attention') pushError('follow-decision guide: explicit full-detail button did not open the Self Attention detail, got '+JSON.stringify(calcDetailHeading));
  const traceBefore=await page.locator('.attention-cell-trace').evaluate(el=>el.dataset.weight).catch(()=>null);

  // F3: the shared drawer must relabel itself FROZEN while it is showing the
  // guide's captured snapshot, not the always-on LIVE label.
  const frozenEyebrow=await page.locator('.transformer-detail-wide .detail-head .eyebrow').innerText();
  if(!frozenEyebrow.includes('FROZEN')) pushError('follow-decision guide: shared detail drawer eyebrow does not say FROZEN while guide is open, got '+JSON.stringify(frozenEyebrow));
  const frozenTraceSource=await page.locator('.attention-cell-trace').getAttribute('data-source');
  if(frozenTraceSource!=='frozen') pushError('follow-decision guide: attention trace data-source is not "frozen" while guide is open, got '+JSON.stringify(frozenTraceSource));

  // Order assertion: the guide must appear BEFORE the advanced detail drawer
  // both in the DOM and on screen, so the guide's arithmetic is never pushed
  // below this much larger panel.
  const calcOrder=await page.evaluate(()=>{
    const guide=document.querySelector('.follow-decision-guide');
    const detail=document.querySelector('.transformer-detail-wide');
    if(!guide || !detail) return null;
    return {
      domGuideBeforeDetail: !!(guide.compareDocumentPosition(detail) & Node.DOCUMENT_POSITION_FOLLOWING),
      guideTop: guide.getBoundingClientRect().top,
      detailTop: detail.getBoundingClientRect().top
    };
  });
  if(!calcOrder || !calcOrder.domGuideBeforeDetail) pushError('follow-decision guide: advanced detail drawer is not positioned after the guide in the DOM');
  if(!calcOrder || !(calcOrder.guideTop < calcOrder.detailTop)) pushError('follow-decision guide: advanced detail drawer does not render below the guide on screen '+JSON.stringify(calcOrder));

  await page.getByRole('tab',{name:/Action/}).click();
  await page.waitForTimeout(150);

  // Precondition: entering Action must NOT auto-open the advanced drawer
  // either (switching stages always closes it; only the explicit button
  // below reopens it for the new stage).
  const advancedOpenOnActionEntry=await page.locator('.transformer-detail-wide').count();
  if(advancedOpenOnActionEntry!==0) pushError('follow-decision guide: Action stage auto-opened the advanced detail drawer on entry (should stay closed until explicitly opened)');

  const entryActionValues=await readFdValues();

  // F2: Action must show the real actionScore and the real score->force
  // relation actually used by the model (10*tanh(score)), not a guess.
  const actionScoreText=entryActionValues.find(t=>t.includes('action score'));
  if(!actionScoreText || !/-?\d/.test(actionScoreText.replace('action score',''))) {
    pushError('follow-decision guide: Action stage does not show a readable action score value');
  }
  const forceLine=await page.locator('.follow-decision-guide .fd-values span').filter({hasText:'10·tanh(score)'}).innerText();
  const forceCommandLine=await page.locator('.follow-decision-guide .fd-values span').filter({hasText:'force command'}).innerText();
  const forceFromRelation=Number(forceLine.match(/-?\d+\.\d+/)?.[0]);
  const forceCommand=Number(forceCommandLine.match(/-?\d+\.\d+/)?.[0]);
  if(!Number.isFinite(forceFromRelation) || !Number.isFinite(forceCommand) || Math.abs(forceFromRelation-forceCommand)>0.01) {
    pushError('follow-decision guide: 10*tanh(score) relation does not match the displayed force command '+JSON.stringify({forceFromRelation,forceCommand}));
  }

  // New arithmetic connection: a selected final-hidden component h[j] times
  // the real action-head weight[0][j], summed over every j plus bias[0],
  // must reproduce the already-displayed frozen action score -- read from
  // the same frozen result object, no second inference.
  const hiddenDimButtons=page.locator('.follow-decision-guide .fd-action-dim-select button');
  const hiddenDimCount=await hiddenDimButtons.count();
  if(hiddenDimCount<2) pushError('follow-decision guide: Action stage final-hidden dimension selector exposes fewer than 2 dimensions, got '+hiddenDimCount);
  const productStep=page.locator('.follow-decision-guide .fd-calc-step[data-action-product]');
  const productBefore=await productStep.getAttribute('data-action-product');
  await hiddenDimButtons.nth(1).click();
  await page.waitForTimeout(80);
  const productAfter=await productStep.getAttribute('data-action-product');
  if(productBefore===productAfter) pushError('follow-decision guide: selecting a different final-hidden dimension did not change the product step');
  const badgeAfterHiddenDimChange=await page.locator('.fd-badge').innerText();
  if(badgeAfterHiddenDimChange!==badgeText0) pushError('follow-decision guide: selecting a final-hidden dimension changed the captured-event badge (frozen identity violated)');

  const sumStep=page.locator('.follow-decision-guide .fd-calc-step[data-action-sum]');
  const sumAttr=Number(await sumStep.getAttribute('data-action-sum'));
  const biasAttr=Number(await sumStep.getAttribute('data-action-bias'));
  const scoreCheckAttr=Number(await sumStep.getAttribute('data-action-score-check'));
  if(!Number.isFinite(sumAttr) || !Number.isFinite(biasAttr) || !Number.isFinite(scoreCheckAttr) || Math.abs(sumAttr+biasAttr-scoreCheckAttr)>1e-6) {
    pushError('follow-decision guide: sum(h*weight)+bias does not equal the frozen action-score check '+JSON.stringify({sumAttr,biasAttr,scoreCheckAttr}));
  }
  const actionScoreDisplayed=Number(actionScoreText.replace('action score','').trim());
  if(!Number.isFinite(actionScoreDisplayed) || Math.abs(actionScoreDisplayed-scoreCheckAttr)>1e-6) {
    pushError('follow-decision guide: displayed action score does not match the chain-equation score '+JSON.stringify({actionScoreDisplayed,scoreCheckAttr}));
  }
  const actionVectorDetailsOpenByDefault=await page.locator('.follow-decision-guide .fd-calc-step[data-action-product] .fd-vector-details').evaluateAll(els=>els.some(el=>el.open));
  if(actionVectorDetailsOpenByDefault) pushError('follow-decision guide: full per-dimension action-product details are expanded by default (should be collapsed native <details>)');
  const pipelineStepText=await page.locator('.follow-decision-guide .fd-calc-step b').filter({hasText:'final hidden'}).count();
  if(pipelineStepText<1) pushError('follow-decision guide: Action stage does not name the context -> output projection/residual -> LayerNorm/MLP -> final hidden pipeline');
  await page.screenshot({path:path.join(outDir,'desktop-follow-decision-action.jpg'),type:'jpeg',quality:84,fullPage:true});

  // Explicit full-detail open (Action): the guide's own score/force values
  // above were already fully readable without this.
  const openActionDetailButton=page.getByRole('button',{name:/open full Action head detail/});
  if(await openActionDetailButton.count()!==1) pushError('follow-decision guide: Action stage is missing the explicit full-detail open button');
  await openActionDetailButton.click();
  await page.waitForTimeout(150);
  const actionDetailHeading=await page.locator('.transformer-detail-wide h2').innerText().catch(()=>'');
  if(actionDetailHeading!=='Action head') pushError('follow-decision guide: explicit full-detail button did not open the Action head detail, got '+JSON.stringify(actionDetailHeading));
  const actionFrozenEyebrow=await page.locator('.transformer-detail-wide .detail-head .eyebrow').innerText();
  if(!actionFrozenEyebrow.includes('FROZEN')) pushError('follow-decision guide: Action head drawer eyebrow does not say FROZEN while guide is open, got '+JSON.stringify(actionFrozenEyebrow));

  await page.getByRole('tab',{name:/Result/}).click();
  await page.waitForTimeout(100);
  const applyButton=page.getByRole('button',{name:/Apply one 20ms step/});
  if(await applyButton.count()!==1) pushError('follow-decision guide: Result stage missing the explicit Apply-step button');

  // Regression: two synchronous DOM clicks in the same task must not apply
  // two physics steps. This bypasses Playwright's own actionability
  // re-checks (which a real double-click on a still-enabled button would
  // also bypass, since disabling only lands after the first click's await),
  // so it exercises the reentrancy guard directly rather than relying on a
  // human being unlikely to double-click fast enough.
  await applyButton.evaluate(el => { el.click(); el.click(); });
  await page.waitForTimeout(200);

  const tickAfterApply=Number(await page.locator('main').getAttribute('data-sync-tick'));
  if(tickAfterApply!==capturedTick+1) pushError('follow-decision guide: duplicate-click Apply-step advanced the plant by '+(tickAfterApply-capturedTick)+' ticks, expected exactly 1');

  const guideTrace=page.locator('.follow-decision-guide .decision-trace');
  const guideTickFrom=Number(await guideTrace.getAttribute('data-tick-from').catch(()=>NaN));
  const guideTickTo=Number(await guideTrace.getAttribute('data-tick-to').catch(()=>NaN));
  if(guideTickFrom!==capturedTick || guideTickTo!==capturedTick+1) pushError('follow-decision guide: displayed Result trace tick range is not t->t+1 for the captured event');

  // Source/provenance: reopening the Pipeline overview from inside the still-
  // open guide, now that Apply has moved the live plant one tick ahead of the
  // captured event, must show the SAME frozen snapshot the guide/Result stage
  // are showing (source=frozen, force matches the captured trace's applied
  // force) — not the live plant, which has already advanced past it.
  await page.locator('.pipeline-disclosure summary').click();
  await page.waitForTimeout(100);
  const pipelineAfterApply=await page.locator('.pipeline-disclosure').evaluate(el=>({
    open:el.open,
    source:el.dataset.source,
    summaryText:el.querySelector('summary')?.innerText||''
  }));
  if(pipelineAfterApply.source!=='frozen') pushError('follow-decision guide: Pipeline overview source is not "frozen" after Apply while the guide is open, got '+JSON.stringify(pipelineAfterApply.source));
  if(!pipelineAfterApply.summaryText.includes('FROZEN') || !pipelineAfterApply.summaryText.includes(String(capturedTick))) {
    pushError('follow-decision guide: Pipeline overview summary does not show FROZEN + captured tick '+capturedTick+', got '+JSON.stringify(pipelineAfterApply.summaryText));
  }
  if(!pipelineAfterApply.summaryText.includes('LIVE') || !pipelineAfterApply.summaryText.includes(String(tickAfterApply))) {
    pushError('follow-decision guide: Pipeline overview summary does not show LIVE + current tick '+tickAfterApply+' (one ahead of captured), got '+JSON.stringify(pipelineAfterApply.summaryText));
  }
  const pipelineForceAfterApply=await page.locator('.pipeline-shell .panel-head>span').innerText();
  const capturedForceText=await guideTrace.getAttribute('data-applied-policy-force');
  const capturedForceAbs=Math.abs(Number(capturedForceText)).toFixed(2);
  if(!pipelineForceAfterApply.includes(capturedForceAbs)) {
    pushError('follow-decision guide: Pipeline overview force does not match the captured event after Apply '+JSON.stringify({pipelineForceAfterApply,capturedForceAbs}));
  }
  await page.locator('.pipeline-disclosure summary').click();
  await page.waitForTimeout(80);

  // Result stage must lead with a readable (>=14px) summary of the actual
  // captured transition, matching the trace exactly, with the old verbose
  // per-field trace present but collapsed by default.
  const detailsOpenBeforeExpand=await page.locator('.follow-decision-guide .fd-details').evaluate(el=>el.open);
  if(detailsOpenBeforeExpand) pushError('follow-decision guide: full dynamics detail is not collapsed by default');
  const summaryTickText=await page.locator('.fd-result-tick').innerText();
  if(!summaryTickText.includes(String(capturedTick)) || !summaryTickText.includes(String(capturedTick+1))) {
    pushError('follow-decision guide: Result summary tick text does not match the captured trace t->t+1');
  }
  const summaryForceText=await page.locator('.fd-result-summary .fd-values').first().innerText();
  const traceAppliedForce=Number(await guideTrace.getAttribute('data-applied-policy-force'));
  if(!summaryForceText.includes(traceAppliedForce.toFixed(2))) {
    pushError('follow-decision guide: Result summary applied-force does not match the captured trace');
  }
  const summaryFontSize=await page.locator('.fd-result-tick').evaluate(el=>parseFloat(getComputedStyle(el).fontSize));
  if(summaryFontSize<14) pushError('follow-decision guide: Result summary text is below the 14px essential-text floor');

  // Revisit Input/Calculation/Action AFTER Apply: must still show the frozen
  // captured-event data, not the now-advanced live plant.
  await page.getByRole('tab',{name:/^Input/}).click();
  await page.waitForTimeout(100);
  const postApplyInputValues=await readFdValues();
  if(JSON.stringify(postApplyInputValues)!==JSON.stringify(entryInputValues)) {
    pushError('follow-decision guide: Input values changed after Apply-step (captured snapshot leaked to live) '+JSON.stringify({before:entryInputValues,after:postApplyInputValues}));
  }

  await page.getByRole('tab',{name:/Calculation/}).click();
  await page.waitForTimeout(150);
  const advancedOpenOnCalcRevisit=await page.locator('.transformer-detail-wide').count();
  if(advancedOpenOnCalcRevisit!==0) pushError('follow-decision guide: revisiting Calculation after Apply auto-opened the advanced detail drawer');
  await page.getByRole('button',{name:/open full Self Attention detail/}).click();
  await page.waitForTimeout(150);
  const traceAfter=await page.locator('.attention-cell-trace').evaluate(el=>el.dataset.weight).catch(()=>null);
  if(traceAfter!==traceBefore) pushError('follow-decision guide: Calculation attention trace changed after Apply-step (not frozen to the captured event) '+JSON.stringify({before:traceBefore,after:traceAfter}));

  await page.getByRole('tab',{name:/Action/}).click();
  await page.waitForTimeout(150);
  const postApplyActionValues=await readFdValues();
  if(JSON.stringify(postApplyActionValues)!==JSON.stringify(entryActionValues)) {
    pushError('follow-decision guide: Action values changed after Apply-step (captured snapshot leaked to live) '+JSON.stringify({before:entryActionValues,after:postApplyActionValues}));
  }

  // Result revisit must not apply a second step.
  await page.getByRole('tab',{name:/Result/}).click();
  await page.waitForTimeout(100);
  const tickAfterRevisit=Number(await page.locator('main').getAttribute('data-sync-tick'));
  if(tickAfterRevisit!==tickAfterApply) pushError('follow-decision guide: revisiting Result advanced the tick again');
  if(await applyButton.count()!==0) pushError('follow-decision guide: Apply-step button reappeared after a step was already applied');

  // New decision: fresh event id and a newly captured tick.
  const newDecisionButton=page.getByRole('button',{name:/Follow next decision/});
  await newDecisionButton.click();
  await page.waitForTimeout(150);
  const badgeText1=await page.locator('.fd-badge').innerText();
  if(badgeText1===badgeText0) pushError('follow-decision guide: new-decision capture did not change the event badge');
  const eventId0=badgeText0.match(/event #(\d+)/)?.[1];
  const eventId1=badgeText1.match(/event #(\d+)/)?.[1];
  if(!eventId1 || eventId0===eventId1) pushError('follow-decision guide: new-decision capture did not advance the event id');

  // Capture the actual Calculation stage (guide still open, advanced detail
  // CLOSED, full vectors collapsed by default) BEFORE the guide closes — a
  // screenshot taken after close would show none of the numeric trace, and
  // one taken with the advanced drawer open would recreate the detached/
  // vertical-overload layout this guide fixes.
  await page.getByRole('tab',{name:/Calculation/}).click();
  await page.waitForTimeout(120);
  const advancedOpenAtScreenshot=await page.locator('.transformer-detail-wide').count();
  if(advancedOpenAtScreenshot!==0) pushError('follow-decision guide: advanced detail drawer is open in the representative Calculation screenshot (should be closed by default)');
  const vectorDetailsOpenAtCapture=await page.locator('.follow-decision-guide .fd-vector-details').evaluateAll(els=>els.some(el=>el.open));
  if(vectorDetailsOpenAtCapture) pushWarning('follow-decision guide: full vector details were expanded before the representative screenshot was captured');
  await page.screenshot({path:path.join(outDir,'desktop-follow-decision.jpg'),type:'jpeg',quality:84,fullPage:true});

  // Exit restores public controls without any extra physics.
  const tickBeforeClose=Number(await page.locator('main').getAttribute('data-sync-tick'));
  await page.getByRole('button',{name:'close follow-one-decision guide'}).click();
  await page.waitForTimeout(100);
  if(await page.locator('.follow-decision-guide').count()!==0) pushError('follow-decision guide: close did not unmount the guide');
  const tickAfterClose=Number(await page.locator('main').getAttribute('data-sync-tick'));
  if(tickAfterClose!==tickBeforeClose) pushError('follow-decision guide: closing the guide changed the plant tick');
  const pushReenabled=await page.getByRole('button',{name:'Push →'}).isDisabled().catch(()=>true);
  if(pushReenabled) pushError('follow-decision guide: Push stayed disabled after the guide was closed');
  const pipelineRestoredOnClose=await page.locator('.pipeline-disclosure').evaluate(el=>({open:el.open,source:el.dataset.source})).catch(()=>({open:false,source:null}));
  if(!pipelineRestoredOnClose.open) pushError('follow-decision guide: Pipeline overview did not restore to its default-visible state after the guide closed');
  if(pipelineRestoredOnClose.source!=='live') pushError('follow-decision guide: Pipeline overview source is not "live" after the guide closed, got '+JSON.stringify(pipelineRestoredOnClose.source));
  const pipelineForceAfterClose=await page.locator('.pipeline-shell .panel-head>span').innerText();
  const liveCommandForceAfterClose=await page.locator('main').getAttribute('data-controller-force');
  const liveCommandForceAbs=Math.abs(Number(liveCommandForceAfterClose)).toFixed(2);
  if(!pipelineForceAfterClose.includes(liveCommandForceAbs)) {
    pushError('follow-decision guide: Pipeline overview force did not revert to the live commanded force after the guide closed '+JSON.stringify({pipelineForceAfterClose,liveCommandForceAbs}));
  }

  // Mode change must invalidate/close a still-open guide cleanly.
  await followEntry.click();
  await page.waitForTimeout(100);
  await page.getByRole('button',{name:'Vision'}).click();
  await page.waitForTimeout(150);
  if(await page.locator('.follow-decision-guide').count()!==0) pushError('follow-decision guide: mode change to Vision left the guide mounted');
  await page.getByRole('button',{name:'State'}).click();
  await page.waitForTimeout(150);
  if(await page.locator('.follow-decision-guide').count()!==0) pushError('follow-decision guide: guide leaked across a mode-change round trip');

  // Optional pixels-only mode becomes mandatory once the trained artifact is present.
  const visionButton=page.getByRole('button',{name:'Vision'});
  const visionEnabled=await visionButton.isEnabled().catch(()=>false);
  report.interactions.visionMode={available:visionEnabled};

  if(visionEnabled){
    await visionButton.click();
    await page.getByRole('button',{name:'Reset'}).click();
    await page.waitForTimeout(450);

    const pauseVision=page.getByRole('button',{name:'Pause'});
    if(await pauseVision.count()) await pauseVision.click();
    await page.waitForTimeout(60);
    const visionSync0=await readAtomicSnapshot(page);
    verifyAtomicSnapshot(visionSync0,'vision pause');
    const stepVision=page.getByRole('button',{name:'Step'});
    if(!(await stepVision.isEnabled().catch(()=>false))) pushError('vision pause: Step button is not enabled');
    else await stepVision.click();
    await page.waitForTimeout(70);
    const visionSync1=await readAtomicSnapshot(page);
    verifyAtomicSnapshot(visionSync1,'vision step');
    if(visionSync1.tick!==visionSync0.tick+1) pushError('vision Step did not advance exactly one tick');
    const visionDecisionTrace=await readDecisionTrace(page);
    verifyDecisionTrace(visionDecisionTrace,visionSync0,visionSync1,'vision decision trace');
    report.interactions.atomicSync.vision={before:visionSync0,after:visionSync1,stepDelta:visionSync1.tick-visionSync0.tick};
    report.interactions.decisionTrace.vision=visionDecisionTrace;
    const runVision=page.getByRole('button',{name:'Run'});
    if(await runVision.count()) await runVision.click();
    await page.waitForTimeout(80);

    const pipelineCount=await page.locator('.vision-pipeline').count();
    const hiddenStateCount=await page.locator('.vision-hidden-state').count();
    const stateReadoutCount=await page.locator('.state-readout').count();
    const frameCount=await page.locator('.vision-stage-frame .vision-frame').count();
    const patchGridCount=await page.locator('.vision-stage-patch .patch-grid').count();
    const patchCellCount=await page.locator('.vision-stage-patch .cell').count();
    const visionActionReadouts=await page.locator('.action-readout span').count();

    if(pipelineCount!==1) pushError('vision mode: pipeline missing');
    if(hiddenStateCount!==1) pushError('vision mode: hidden-state label missing');
    if(stateReadoutCount!==0) pushError('vision mode: explicit state readout leaked into pixels-only mode');
    if(visionActionReadouts!==2) pushError('vision mode: expected exactly 2 action-readout cells (force+push) despite hidden state, found '+visionActionReadouts);
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

  if(!fusionEnabled) pushError('fusion mode: trained artifact exists but Fusion button is disabled');
  if(fusionEnabled){
    const timeBeforeFusion=parseFloat((await page.locator('.sim-card .time').innerText()).replace(' s',''));
    await fusionButton.click();
    await page.waitForTimeout(260);

    // Mode-switch continuity: switching alone (before any Reset) must not
    // rewind the shared episode clock. Checked here, once, before the
    // deliberate fresh-episode reset below establishes a new baseline.
    const timeAfterSwitch=parseFloat((await page.locator('.sim-card .time').innerText()).replace(' s',''));
    const fusionSameEpisode=timeAfterSwitch>=timeBeforeFusion;
    if(!fusionSameEpisode) pushError('fusion mode: mode switch reset or rewound the episode clock');

    // `running` and `elapsed` are shared app state across modes: Vision's own
    // Run/push exploration can legitimately run the pole to a terminal fall
    // (status='fell') before Fusion is ever reached, which disables every
    // control (Pause/Run/Step) and cascades into unrelated failures below.
    // Qualify this mode from a known-fresh episode via the same public
    // Reset/Pause controls a user would use, rather than trusting whatever
    // state leaked over from the previous mode.
    await page.getByRole('button',{name:'Reset'}).click();
    await page.waitForTimeout(300);
    await page.getByRole('button',{name:'Pause'}).click();
    await page.waitForTimeout(60);
    const fusionSync0=await readAtomicSnapshot(page);
    verifyAtomicSnapshot(fusionSync0,'fusion pause');
    const stepFusion=page.getByRole('button',{name:'Step'});
    if(!(await stepFusion.isEnabled().catch(()=>false))) pushError('fusion pause: Step button is not enabled');
    else await stepFusion.click();
    await page.waitForTimeout(70);
    const fusionSync1=await readAtomicSnapshot(page);
    verifyAtomicSnapshot(fusionSync1,'fusion step');
    if(fusionSync1.tick!==fusionSync0.tick+1) pushError('fusion Step did not advance exactly one tick');
    const fusionDecisionTrace=await readDecisionTrace(page);
    verifyDecisionTrace(fusionDecisionTrace,fusionSync0,fusionSync1,'fusion decision trace');
    report.interactions.atomicSync.fusion={before:fusionSync0,after:fusionSync1,stepDelta:fusionSync1.tick-fusionSync0.tick};
    report.interactions.decisionTrace.fusion=fusionDecisionTrace;
    const runFusion=page.getByRole('button',{name:'Run'});
    if(await runFusion.count()) await runFusion.click();
    await page.waitForTimeout(80);

    const fusionModeAttr=await page.locator('main').getAttribute('data-observation-mode');
    const fusionPipelineCount=await page.locator('.fusion-pipeline').count();
    const fusionPairCount=await page.locator('.fusion-stage-observation .pair').count();
    const fusionTokenCount=await page.locator('.fusion-stage-token .token').count();
    const fusionAttentionCells=await page.locator('.fusion-stage-attention .cell').count();
    const fusionScenarioButtons=await page.locator('.fusion-pipeline .scenario-bar button').count();
    const fusionStateReadout=await page.locator('.state-readout').count();
    const fusionActionReadouts=await page.locator('.action-readout span').count();

    if(fusionModeAttr!=='fusion') pushError('fusion mode: main observation mode did not switch to fusion');
    if(fusionActionReadouts!==2) pushError('fusion mode: expected exactly 2 action-readout cells (force+push), found '+fusionActionReadouts);
    if(fusionPipelineCount!==1) pushError('fusion mode: pipeline missing');
    if(fusionPairCount!==8) pushError('fusion mode: expected 8 aligned timestamp pairs, found '+fusionPairCount);
    if(fusionTokenCount!==16) pushError('fusion mode: expected 16 typed tokens, found '+fusionTokenCount);
    if(fusionAttentionCells!==256) pushError('fusion mode: expected 16x16 attention matrix, found '+fusionAttentionCells+' cells');
    if(fusionScenarioButtons!==5) pushError('fusion mode: expected 5 degradation controls, found '+fusionScenarioButtons);
    if(fusionStateReadout!==1) pushError('fusion mode: explicit state readout should be visible');

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
    await page.getByRole('button',{name:'Reset'}).click();
    await page.waitForTimeout(350);

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
      stateReadoutCount:fusionStateReadout,sameEpisode:fusionSameEpisode,
      missingStateOff,partialVision,missingVisionOff,noisyActive,
      detailCount:fusionDetailCount,detailAttentionCells:fusionDetailCells,ablationCount:fusionAblations
    };
    await page.screenshot({path:path.join(outDir,'desktop-fusion-detail.jpg'),type:'jpeg',quality:84,fullPage:true});
  }

  // Deterministic side-by-side comparison replay.
  const closeFusionDetail=page.getByRole('button',{name:'close fusion detail'});
  if(await closeFusionDetail.count()) await closeFusionDetail.click();

  const compareButton=page.getByRole('button',{name:'Compare'});
  const compareEnabled=await compareButton.isEnabled().catch(()=>false);
  report.interactions.compareMode={available:compareEnabled};
  if(!compareEnabled) pushError('compare mode: Compare button is disabled despite all learned artifacts');

  if(compareEnabled){
    await compareButton.click();
    await page.waitForTimeout(260);

    const compareModeAttr=await page.locator('main').getAttribute('data-observation-mode');
    const lab=page.locator('.comparison-lab');
    const labCount=await lab.count();
    const cardCount=await page.locator('.comparison-lab .controller-card').count();
    const firstTick=Number(await lab.getAttribute('data-current-tick'));
    const firstDisturbances=await page.locator('.comparison-lab .controller-card').evaluateAll(els=>els.map(el=>Number(el.dataset.disturbance)));

    if(compareModeAttr!=='compare') pushError('compare mode: main observation mode did not switch to compare');
    if(labCount!==1) pushError('compare mode: comparison lab missing');
    if(cardCount!==3) pushError('compare mode: expected 3 controller cards, found '+cardCount);
    if(new Set(firstDisturbances).size!==1) pushError('compare mode: controller disturbances differ on shared tick');

    const pauseButton=page.getByRole('button',{name:'Pause'});
    if(await pauseButton.count()) await pauseButton.click();

    const slider=page.getByRole('slider',{name:'comparison replay tick'});
    const recordedMax=Number(await slider.getAttribute('max'));
    const scrubTarget=Math.min(5,recordedMax);
    await slider.evaluate((el,target)=>{
      el.value=String(target);
      el.dispatchEvent(new Event('input',{bubbles:true}));
    },scrubTarget);
    await page.waitForTimeout(60);
    const scrubTick=Number(await lab.getAttribute('data-current-tick'));
    if(scrubTick!==scrubTarget) pushError('compare mode: scrub tick mismatch '+scrubTick+' != '+scrubTarget);

    await page.getByRole('button',{name:'Fast-forward to end'}).click();
    await page.waitForTimeout(80);
    const done1=await lab.getAttribute('data-done');
    const finalTick1=Number(await lab.getAttribute('data-current-tick'));
    const traceLength1=Number(await lab.getAttribute('data-trace-length'));
    const metrics1=await page.locator('.comparison-lab .controller-card .metrics').allInnerTexts();

    if(done1!=='true') pushError('compare mode: fast-forward did not complete deterministic run');
    if(finalTick1!==500) pushError('compare mode: final tick is '+finalTick1+' instead of 500');
    if(traceLength1!==501) pushError('compare mode: trace length is '+traceLength1+' instead of 501');

    const replayButton=page.getByRole('button',{name:'Replay'});
    if(await replayButton.count()!==1) pushError('compare mode: completed trace does not expose Replay');
    else {
      await replayButton.click();
      await page.waitForTimeout(140);
      const replayTick=Number(await lab.getAttribute('data-current-tick'));
      if(!(replayTick>0 && replayTick<100)) pushError('compare mode: recorded Replay did not advance from tick 0, got '+replayTick);
      const replayPause=page.getByRole('button',{name:'Pause'});
      if(await replayPause.count()) await replayPause.click();
    }

    await slider.evaluate((el,target)=>{
      el.value=String(target);
      el.dispatchEvent(new Event('input',{bubbles:true}));
    },81);
    await page.waitForTimeout(60);
    const pulseTick=Number(await lab.getAttribute('data-current-tick'));
    const pulseDisturbances=await page.locator('.comparison-lab .controller-card').evaluateAll(els=>els.map(el=>Number(el.dataset.disturbance)));
    if(pulseTick!==81) pushError('compare mode: pulse scrub did not land on tick 81');
    if(pulseDisturbances.some(v=>v!==4)) pushError('compare mode: tick 81 did not apply shared +4 N disturbance to all controllers: '+pulseDisturbances.join(','));

    await page.screenshot({path:path.join(outDir,'desktop-comparison-pulse.jpg'),type:'jpeg',quality:84,fullPage:true});

    await page.getByRole('button',{name:'Reset deterministic run'}).click();
    await page.getByRole('button',{name:'Fast-forward to end'}).click();
    await page.waitForTimeout(80);
    const metrics2=await page.locator('.comparison-lab .controller-card .metrics').allInnerTexts();
    const deterministicReset=JSON.stringify(metrics1)===JSON.stringify(metrics2);
    if(!deterministicReset) pushError('compare mode: reset + fast-forward produced different final metrics');

    report.interactions.compareMode={
      available:true,modeAttr:compareModeAttr,labCount,cardCount,firstTick,
      sharedDisturbance:firstDisturbances[0],scrubTick,finalTick:finalTick1,
      traceLength:traceLength1,pulseTick,pulseDisturbances,deterministicReset
    };
    await page.screenshot({path:path.join(outDir,'desktop-comparison-final.jpg'),type:'jpeg',quality:84,fullPage:true});
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
    pushError('mobile fusion: trained artifact exists but Fusion button is disabled');
  }

  const closeMobileFusion=page.getByRole('button',{name:'close fusion detail'});
  if(await closeMobileFusion.count()) await closeMobileFusion.click();

  const mobileCompareButton=page.getByRole('button',{name:'Compare'});
  const mobileCompareEnabled=await mobileCompareButton.isEnabled().catch(()=>false);
  if(!mobileCompareEnabled){
    report.interactions.mobileCompare={enabled:false};
    pushError('mobile compare: Compare button is disabled');
  } else {
    await mobileCompareButton.click();
    await page.waitForTimeout(180);

    const mobileCompareLab=await page.locator('.comparison-lab').count();
    const mobileCompareCards=page.locator('.comparison-lab .controller-card');
    const mobileCompareCardCount=await mobileCompareCards.count();
    const rects=await mobileCompareCards.evaluateAll(els=>els.map(el=>{const r=el.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};}));
    const mobileCompareDoc=await page.evaluate(()=>({w:document.documentElement.scrollWidth,v:innerWidth}));
    const singleColumn=rects.length===3 && Math.max(...rects.map(r=>r.x))-Math.min(...rects.map(r=>r.x))<3 && rects[1].y>rects[0].y && rects[2].y>rects[1].y;

    if(mobileCompareLab!==1) pushError('mobile compare: comparison lab missing');
    if(mobileCompareCardCount!==3) pushError('mobile compare: expected 3 controller cards, found '+mobileCompareCardCount);
    if(!singleColumn) pushError('mobile compare: controller cards are not stacked in one column');
    if(mobileCompareDoc.w>mobileCompareDoc.v+2) pushError('mobile compare: page-level horizontal overflow');

    const mobilePause=page.getByRole('button',{name:'Pause'});
    if(await mobilePause.count()) await mobilePause.click();
    await page.getByRole('button',{name:'Fast-forward to end'}).click();
    await page.waitForTimeout(80);

    const mobileDone=await page.locator('.comparison-lab').getAttribute('data-done');
    const mobileTick=Number(await page.locator('.comparison-lab').getAttribute('data-current-tick'));
    if(mobileDone!=='true' || mobileTick!==500) pushError('mobile compare: deterministic fast-forward did not reach tick 500');

    report.interactions.mobileCompare={
      enabled:true,labCount:mobileCompareLab,cardCount:mobileCompareCardCount,
      singleColumn,finalTick:mobileTick
    };
    await page.screenshot({path:path.join(outDir,'mobile-comparison-final.jpg'),type:'jpeg',quality:80,fullPage:true});
  }

  await page.close();
}

// F3 regression: guide-owned Key/dim selection must round-trip exactly with
// the shared full-detail drawer (identity on open, either surface's changes
// visible in both, persisted across close/reopen/stage-nav/one physical
// step), Query must stay centrally locked to the guide's last token even
// when the hover paths on the (reopened) Pipeline overview try to move it,
// and a new decision event must deliberately reset the selection. Runs in
// its own page/context so it never shares state with runDesktop/runMobile.
async function runFollowGuideSelectionRoundtrip(browser, width, label, keyIndex, dimIndex) {
  const page = await browser.newPage({ viewport:{width,height:Math.max(900,width<500?1500:1000)}, deviceScaleFactor:1 });
  const pageErrors=[];
  page.on('pageerror', err=>pageErrors.push(String(err)));

  await page.goto(baseURL, { waitUntil:'networkidle', timeout:30000 });
  await waitLearned(page);
  await page.getByRole('button',{name:'Reset'}).click();
  await page.waitForTimeout(250);

  await page.getByRole('button',{name:'한 판단 따라가기 · Follow one decision'}).click();
  await page.waitForTimeout(150);
  if (await page.locator('.follow-decision-guide').count()!==1) pushError(label+': guide did not open exactly once');

  await page.getByRole('tab',{name:/Calculation/}).click();
  await page.waitForTimeout(120);

  const keyButtons=page.locator('.follow-decision-guide .fd-key-select button');
  const dimButtons=page.locator('.follow-decision-guide .fd-dim-select button');
  const keyCount=await keyButtons.count();
  const dimCount=await dimButtons.count();
  const lastIndex=keyCount-1;
  const chosenKey=Math.min(keyIndex,lastIndex);
  const chosenDim=Math.min(dimIndex,Math.max(0,dimCount-1));
  await keyButtons.nth(chosenKey).click();
  await dimButtons.nth(chosenDim).click();
  await page.waitForTimeout(80);

  const readTrace=()=>page.locator('.attention-cell-trace').evaluate(el=>({
    row:Number(el.dataset.row), col:Number(el.dataset.col), dim:Number(el.dataset.highlightDim),
    lockQuery:el.dataset.lockQuery, score:Number(el.dataset.score),
    preBias:Number(el.dataset.preBiasScore), bias:Number(el.dataset.biasValue)
  }));

  // Cross-check the guide's own Calculation-stage numbers against the full
  // shared drawer's numbers for the SAME cell, both read straight off the
  // live DOM (no recomputation) -- this is the actual F3 identity claim
  // (not just matching row/col/dim ids). weight/vDim/contributionDim come
  // from the guide's data-contribution-dim step; weight/highlighted-v/
  // -contribution/-context come from the drawer's data-weight and
  // .dim-highlight block. Raw values (not just a boolean) are captured into
  // the report so the coordinator can verify the comparison directly.
  const eps=1e-6;
  async function crossCheckGuideVsTrace(point) {
    const guide=await page.locator('.follow-decision-guide .fd-calc-step[data-contribution-dim]').evaluate(el=>({
      weight:Number(el.dataset.weight), vDim:Number(el.dataset.vDim), contributionDim:Number(el.dataset.contributionDim)
    }));
    const traceEl=page.locator('.attention-cell-trace');
    const drawer=await traceEl.evaluate(el=>({
      weight:Number(el.dataset.weight),
      source:el.dataset.source
    }));
    const highlight=await traceEl.locator('.dim-highlight').evaluate(el=>({
      v:Number(el.dataset.highlightedV), contribution:Number(el.dataset.highlightedContribution), context:Number(el.dataset.highlightedContext)
    })).catch(()=>null);
    const tick=Number(await page.locator('main').getAttribute('data-sync-tick'));
    const record={point, tick, guide, drawer, highlight};
    (report.interactions[label+'-crossCheck']=report.interactions[label+'-crossCheck']||[]).push(record);
    if(!highlight) { pushError(label+': '+point+': drawer has no .dim-highlight block to cross-check against the guide'); return; }
    if(Math.abs(guide.weight-drawer.weight)>eps) pushError(label+': '+point+': guide weight != drawer weight '+JSON.stringify(record));
    if(Math.abs(guide.vDim-highlight.v)>eps) pushError(label+': '+point+': guide V[dim] != drawer highlighted V '+JSON.stringify(record));
    if(Math.abs(guide.contributionDim-highlight.contribution)>eps) pushError(label+': '+point+': guide contribution[dim] != drawer highlighted contribution '+JSON.stringify(record));
  }

  // 1) Identity on open: the shared drawer must show exactly the guide's
  // current Query (locked to last)/Key/dim -- no separate sync call needed,
  // both surfaces read the same App state.
  const openButton=page.getByRole('button',{name:/open full Self Attention detail/});
  await openButton.click();
  await page.waitForTimeout(150);
  let trace=await readTrace();
  if(trace.row!==lastIndex) pushError(label+': drawer Query does not match guide fixed last token on open, row='+trace.row);
  if(trace.col!==chosenKey) pushError(label+': drawer Key does not match guide selection on open '+JSON.stringify({expected:chosenKey,got:trace.col}));
  if(trace.dim!==chosenDim) pushError(label+': drawer highlighted dim does not match guide selection on open '+JSON.stringify({expected:chosenDim,got:trace.dim}));
  if(trace.lockQuery!=='true') pushError(label+': drawer Query is not locked while guide is open');
  const queryButtonsDisabled=await page.locator('.attention-cell-trace .trace-query-button').evaluateAll(els=>els.every(el=>el.disabled));
  if(!queryButtonsDisabled) pushError(label+': Query selector buttons are not all disabled while guide is open');
  await crossCheckGuideVsTrace('first-open');

  // Bounds/typography: the new interactive per-dim tiles must meet the 44px
  // touch target floor, and the selected-dim readout must meet the 14px
  // essential-text floor -- neither shrunk nor hidden to fit.
  const productDimBoxes=await page.locator('.attention-cell-trace .product-dim').evaluateAll(els=>els.map(el=>{const r=el.getBoundingClientRect();return {w:r.width,h:r.height};}));
  if(productDimBoxes.some(b=>b.w<44||b.h<44)) pushError(label+': a product-dim tile is below the 44px touch-target floor '+JSON.stringify(productDimBoxes));
  const dimHighlightFont=await page.locator('.attention-cell-trace .dim-highlight code').evaluate(el=>parseFloat(getComputedStyle(el).fontSize)).catch(()=>null);
  if(dimHighlightFont===null||dimHighlightFont<14) pushError(label+': selected-dim readout is below the 14px essential-text floor, got '+dimHighlightFont);
  await page.screenshot({path:path.join(outDir,label+'-nondefault-detail.jpg'),type:'jpeg',quality:82,fullPage:true});

  // 1b) Provenance: Key now intentionally persists across stages, so with a
  // non-last Key still selected, opening the Action full detail must label
  // its header with the last token ("t") -- what Action actually reads
  // (hidden[last]/context[last]) -- not the persisted Key. The stored Key
  // itself must be untouched by merely opening Action, and Calculation must
  // show the exact same selection on return.
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(60);
  await page.getByRole('tab',{name:/Action/}).click();
  await page.waitForTimeout(100);
  await page.getByRole('button',{name:/open full Action head detail/}).click();
  await page.waitForTimeout(150);
  const actionEyebrow=await page.locator('.transformer-detail-wide .detail-head .eyebrow').innerText().catch(()=>'');
  const actionEyebrowSegments=actionEyebrow.split('·').map(s=>s.trim());
  if(!actionEyebrowSegments.includes('t')) {
    pushError(label+': Action full-detail header does not label the last token "t" while a non-last Key is selected, got '+JSON.stringify(actionEyebrow));
  }
  // .fd-key-select is intentionally not rendered on the Action stage (it's
  // a Calculation-stage control), so it cannot be queried here to prove the
  // stored Key survived -- that would be a null/false-negative read, not
  // evidence either way. The real proof is the existing invariant below:
  // returning to Calculation (where the selector actually renders) and
  // reopening the drawer must show the exact same Key/dim that was chosen
  // before visiting Action.
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(60);
  await page.getByRole('tab',{name:/Calculation/}).click();
  await page.waitForTimeout(100);
  await openButton.click();
  await page.waitForTimeout(120);
  trace=await readTrace();
  if(trace.col!==chosenKey||trace.dim!==chosenDim) pushError(label+': selection did not return to the original Key/dim after visiting Action '+JSON.stringify({trace,chosenKey,chosenDim}));
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(60);
  await openButton.click();
  await page.waitForTimeout(120);

  // 2) Change Key/dim IN the shared drawer; the guide (same App state) must
  // reflect it immediately -- this is the two-way half of the round trip.
  const altKey=chosenKey===0?lastIndex:0;
  const altDim=chosenDim===0?Math.max(0,Math.min(1,dimCount-1)):0;
  await page.locator('.attention-cell-trace .trace-key-button[data-index="'+altKey+'"]').click();
  await page.locator('.attention-cell-trace .product-dim[data-dim="'+altDim+'"]').click();
  await page.waitForTimeout(100);
  const guideAfterDrawerChange=await page.evaluate(()=>({
    key:Number(document.querySelector('.follow-decision-guide .fd-key-select button.active')?.dataset.index),
    dim:Number(document.querySelector('.follow-decision-guide .fd-dim-select button.active')?.dataset.index)
  }));
  if(guideAfterDrawerChange.key!==altKey) pushError(label+': guide Key selector did not follow a Key change made in the shared drawer '+JSON.stringify({expected:altKey,got:guideAfterDrawerChange.key}));
  if(guideAfterDrawerChange.dim!==altDim) pushError(label+': guide dim selector did not follow a dim change made in the shared drawer '+JSON.stringify({expected:altDim,got:guideAfterDrawerChange.dim}));

  // 3) Close/reopen the drawer: selection must persist (not reset to guide
  // defaults) since no new event has started.
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(80);
  await openButton.click();
  await page.waitForTimeout(120);
  trace=await readTrace();
  if(trace.col!==altKey||trace.dim!==altDim) pushError(label+': selection did not persist across close/reopen of the shared drawer '+JSON.stringify({trace,altKey,altDim}));

  // 4) Traverse Input -> Action -> Result -> Calculation (each stage change
  // auto-closes the drawer, never auto-opens it); Key/dim must still persist
  // and the drawer must not have appeared uninvited on any of these stages.
  for (const tabName of [/^Input/,/Action/,/Result/,/Calculation/]) {
    await page.getByRole('tab',{name:tabName}).click();
    await page.waitForTimeout(100);
    if (await page.locator('.transformer-detail-wide').count()!==0) pushError(label+': stage navigation to '+tabName+' auto-opened the shared drawer');
  }
  await openButton.click();
  await page.waitForTimeout(120);
  trace=await readTrace();
  if(trace.col!==altKey||trace.dim!==altDim) pushError(label+': selection did not persist across stage navigation '+JSON.stringify({trace,altKey,altDim}));
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(80);

  // 5) Hover the Pipeline overview (embedding token row + attention matrix
  // cell at an earlier row) while the guide is open: Query must stay locked
  // to the last token centrally (App.selectToken/selectAttention), no matter
  // which caller tries to move it -- not just the drawer's own controls.
  await page.locator('.pipeline-disclosure summary').click();
  await page.waitForTimeout(100);
  await page.locator('.embedding-overview .embedding-token').first().hover();
  await page.waitForTimeout(60);
  await page.locator('.attention-overview .row-0.col-0').hover();
  await page.waitForTimeout(60);
  await page.locator('.pipeline-disclosure summary').click();
  await page.waitForTimeout(80);
  await openButton.click();
  await page.waitForTimeout(120);
  trace=await readTrace();
  if(trace.row!==lastIndex) pushError(label+': Pipeline hover (embedding token / attention matrix) unlocked Query while guide is open, row='+trace.row);
  const colAfterHover=trace.col;
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(80);

  // 6) Apply exactly once: the live plant advances by exactly one tick, but
  // the guide's own selected Calculation cell must remain exactly what it
  // was -- no hidden extra step, no selection drift from Apply itself.
  await page.getByRole('tab',{name:/Result/}).click();
  await page.waitForTimeout(100);
  const tickBeforeApply=Number(await page.locator('main').getAttribute('data-sync-tick'));
  await page.getByRole('button',{name:/Apply one 20ms step/}).click();
  await page.waitForTimeout(200);
  const tickAfterApply=Number(await page.locator('main').getAttribute('data-sync-tick'));
  if(tickAfterApply!==tickBeforeApply+1) pushError(label+': Apply did not advance the live plant by exactly one tick, got delta '+(tickAfterApply-tickBeforeApply));

  await page.getByRole('tab',{name:/Calculation/}).click();
  await page.waitForTimeout(100);
  await openButton.click();
  await page.waitForTimeout(120);
  trace=await readTrace();
  if(trace.col!==colAfterHover||trace.dim!==altDim) pushError(label+': selected Calculation cell changed after Apply advanced the live plant '+JSON.stringify({trace,colAfterHover,altDim}));
  const preBiasCheck=Number.isFinite(trace.preBias)&&Number.isFinite(trace.bias)&&Number.isFinite(trace.score);
  if(!preBiasCheck) pushError(label+': attention trace has a non-finite score/preBias/bias value after Apply '+JSON.stringify(trace));
  await crossCheckGuideVsTrace('post-apply');
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(60);

  // 7) A new event deliberately resets Query/Key/dim to defaults. "Follow
  // next decision" only renders on the Result stage (after Apply) -- step 6
  // above navigated away to Calculation to inspect the post-Apply cell, so
  // return to Result first rather than exposing the button elsewhere.
  await page.getByRole('tab',{name:/Result/}).click();
  await page.waitForTimeout(100);
  await page.getByRole('button',{name:/Follow next decision/}).click();
  await page.waitForTimeout(150);
  await page.getByRole('tab',{name:/Calculation/}).click();
  await page.waitForTimeout(100);
  await openButton.click();
  await page.waitForTimeout(120);
  trace=await readTrace();
  if(trace.row!==lastIndex||trace.col!==lastIndex||trace.dim!==0) {
    pushError(label+': new decision event did not reset Query/Key/dim to defaults '+JSON.stringify(trace));
  }
  await page.getByRole('button',{name:'close Transformer detail'}).click();
  await page.waitForTimeout(60);

  // 8) Outside the guide, Query/Key must be fully free again, including
  // reaching a causally-masked cell (query=0, key=lastIndex).
  await page.locator('.fd-close').click();
  await page.waitForTimeout(120);
  if (await page.locator('.follow-decision-guide').count()!==0) pushError(label+': guide did not close');
  await page.locator('.attention-overview').click();
  await page.waitForTimeout(150);
  const outsideQueryButtonsDisabled=await page.locator('.attention-cell-trace .trace-query-button').evaluateAll(els=>els.some(el=>el.disabled));
  if(outsideQueryButtonsDisabled) pushError(label+': Query selector buttons are disabled outside the guide (should be fully free)');
  await page.locator('.attention-cell-trace .trace-query-button[data-index="0"]').click();
  await page.waitForTimeout(60);
  await page.locator('.attention-cell-trace .trace-key-button[data-index="'+lastIndex+'"]').click();
  await page.waitForTimeout(60);
  const outsideState=await page.locator('.attention-cell-trace').evaluate(el=>({row:Number(el.dataset.row),col:Number(el.dataset.col),masked:el.dataset.masked,weight:Number(el.dataset.weight)}));
  if(outsideState.row!==0||outsideState.col!==lastIndex) pushError(label+': free Query/Key selection outside the guide did not reach query=0/key='+lastIndex+' '+JSON.stringify(outsideState));
  if(outsideState.masked!=='true'||outsideState.weight!==0) pushError(label+': query=0/key='+lastIndex+' is not causally masked outside the guide '+JSON.stringify(outsideState));
  const futureKeyButtonHasClass=await page.locator('.attention-cell-trace .trace-key-button[data-index="'+lastIndex+'"]').evaluate(el=>el.classList.contains('future'));
  if(!futureKeyButtonHasClass) pushError(label+': future key button is not visually marked outside the guide');

  if(pageErrors.length) pushError(label+': unexpected page error(s) during guide selection roundtrip: '+JSON.stringify(pageErrors));
  report.interactions[label]={keyIndex:chosenKey,dimIndex:chosenDim,altKey,altDim,pageErrors:pageErrors.length};
  await page.close();
}

// F2 regression: a real (local test-only) model-load failure -- a Playwright
// route that fails ONLY the tiny-transformer.json request with a 503, every
// other request (including the other trained models) stays real -- must
// drive the app into the transparent toy fallback, keep it permanently
// identifiable, and show truthful (non-fabricated) content on every toy
// stage. This is a deliberate local network-failure fixture for this test
// only, not a claim about any real/production outage.
async function runModelLoadFailureFixture(browser) {
  const context=await browser.newContext({ viewport:{width:1440,height:1000}, deviceScaleFactor:1 });
  const page=await context.newPage();
  const pageErrors=[];
  page.on('pageerror', err=>pageErrors.push(String(err)));

  await page.route('**/model/tiny-transformer.json', route => route.fulfill({ status:503, contentType:'text/plain', body:'deliberate local test fixture: simulated model-fetch failure' }));

  await page.goto(baseURL, { waitUntil:'networkidle', timeout:30000 });
  await page.waitForFunction(() => (document.querySelector('.topbar span')?.textContent||'').includes('fallback'), null, { timeout:15000 });
  await page.waitForTimeout(300);

  const modelState=await page.locator('main').getAttribute('data-model-state');
  if(modelState!=='toy-fallback') pushError('model-load-failure fixture: data-model-state is not "toy-fallback", got '+JSON.stringify(modelState));

  // Persistent, not momentary: the banner is part of the normal document
  // flow (not just a topbar span that can scroll offscreen).
  const bannerCount=await page.locator('.model-fallback-banner').count();
  if(bannerCount!==1) pushError('model-load-failure fixture: persistent fallback banner is missing');
  const bannerText=await page.locator('.model-fallback-banner').innerText().catch(()=>'');
  if(!/failed to load|fallback/i.test(bannerText)) pushError('model-load-failure fixture: fallback banner text does not explain the failure, got '+JSON.stringify(bannerText));

  // Physics should not be silently racing ahead while the user reads a
  // fallback they didn't expect -- guide entry (learned-only) must be
  // disabled, which is the existing gate; explicitly assert it here.
  const guideEntryDisabled=await page.locator('.follow-decision-entry').isDisabled().catch(()=>false);
  if(!guideEntryDisabled) pushError('model-load-failure fixture: Follow-one-decision entry is not disabled in toy fallback');
  await page.getByRole('button',{name:/^(Pause|Run)$/}).click().catch(()=>{});
  await page.waitForTimeout(50);

  await page.screenshot({path:path.join(outDir,'network-fixture-toy-fallback-overview.jpg'),type:'jpeg',quality:82,fullPage:true});

  // Two more reachable places that still described the (inactive) learned
  // model, same audited wrong-concept family as the panels above: the
  // Pipeline panel title claimed a "1 block" that toy fallback truthfully
  // has none of, and every embedding token glyph hardcoded "8D" although
  // the real toy token is 4D (STATE_FIELDS). Both are label-only fixes;
  // the token-dimension label now reads the actual token length instead of
  // a hardcoded constant.
  const panelTitle=await page.locator('.pipeline-shell .panel-head strong').innerText().catch(()=>'');
  if(/1 block/.test(panelTitle)) pushError('model-load-failure fixture: Pipeline panel title still claims "1 block" in toy fallback, got '+JSON.stringify(panelTitle));
  if(!/fixed attention/i.test(panelTitle)) pushError('model-load-failure fixture: Pipeline panel title does not name the actual fixed-attention toy model, got '+JSON.stringify(panelTitle));
  const tokenDimLabels=await page.locator('.embedding-overview .dim-label').allInnerTexts();
  if(tokenDimLabels.length===0 || tokenDimLabels.some(t=>t.trim()!=='4D')) {
    pushError('model-load-failure fixture: toy embedding token dimension labels are not all "4D" '+JSON.stringify(tokenDimLabels));
  }

  const stageCases=[
    ['.embedding-overview','Embedding'],
    ['.qkv-overview','Q · K · V'],
    ['.attention-overview','Self Attention'],
    ['.block-overview','Context (pass-through)'],
    ['.action-overview','Action head'],
  ];
  for (const [sel,expectedHeading] of stageCases) {
    await page.locator(sel).click();
    await page.waitForTimeout(150);
    const heading=await page.locator('.transformer-detail-wide h2').innerText().catch(()=>'');
    if(heading!==expectedHeading) pushError('model-load-failure fixture: '+sel+' heading is '+JSON.stringify(heading)+', expected '+JSON.stringify(expectedHeading));
    const eyebrow=await page.locator('.transformer-detail-wide .detail-head .eyebrow').innerText().catch(()=>'');
    if(!eyebrow.includes('TOY FALLBACK')) pushError('model-load-failure fixture: '+sel+' eyebrow does not persistently label TOY FALLBACK, got '+JSON.stringify(eyebrow));
    const bodyText=await page.locator('.transformer-detail-wide').innerText().catch(()=>'');
    if(/Linear 4→8|LayerNorm\(token\)|Wₒ →|Linear \+ GELU|learned embedding weight|action weight ·/.test(bodyText)) {
      pushError('model-load-failure fixture: '+sel+' shows fabricated learned-only content in toy fallback: '+JSON.stringify(bodyText.slice(0,400)));
    }
    if(sel==='.qkv-overview' && !bodyText.includes('identity')) pushError('model-load-failure fixture: Q/K/V stage does not truthfully label V as identity');
    if(sel==='.action-overview' && !bodyText.includes('1.50')) pushError('model-load-failure fixture: Action stage does not show the fixed feedback gain head');
    const closeBtn=page.getByRole('button',{name:'close Transformer detail'});
    if(await closeBtn.count()) await closeBtn.click();
    await page.waitForTimeout(80);
  }

  // Attention arithmetic: dot/scale (preBias) + fixed bias must equal score
  // exactly for first/middle/last keys, the bias must be exactly
  // 1.20*keyIndex (the audited initial recency prior), masks/weights must be
  // valid, and nothing may read as NaN/undefined.
  await page.locator('.attention-overview').click();
  await page.waitForTimeout(150);

  // The upstream QK^T score-matrix heading always said "Dot product ·
  // QKᵀ/√d" even though the passed `scores` in toy fallback already has the
  // fixed +1.20*j bias baked in -- same wrong-concept family as the panels
  // above. It must now explicitly name the bias; the normal learned label
  // (no bias) is covered separately by the existing runDesktop suite.
  const scoreCalcTitle=await page.locator('.attention-expansion .calc-title').first().innerText().catch(()=>'');
  if(!/bias/i.test(scoreCalcTitle) || !/1\.20|×j/.test(scoreCalcTitle)) {
    pushError('model-load-failure fixture: score-matrix heading does not disclose the fixed bias baked into toy scores, got '+JSON.stringify(scoreCalcTitle));
  }

  const keyButtons=page.locator('.attention-cell-trace .trace-key-button');
  const keyCount=await keyButtons.count();
  const lastIndex=keyCount-1;
  const middleIndex=Math.floor(lastIndex/2);
  for (const keyIndex of [0,middleIndex,lastIndex]) {
    await keyButtons.nth(keyIndex).click();
    await page.waitForTimeout(60);
    const cell=await page.locator('.attention-cell-trace').evaluate(el=>({
      preBias:Number(el.dataset.preBiasScore), bias:Number(el.dataset.biasValue), score:Number(el.dataset.score),
      weight:Number(el.dataset.weight), hasFixedBias:el.dataset.hasFixedBias
    }));
    if(cell.hasFixedBias!=='true') pushError('model-load-failure fixture: toy attention trace does not expose the fixed-bias decomposition at key='+keyIndex);
    if(![cell.preBias,cell.bias,cell.score,cell.weight].every(Number.isFinite)) pushError('model-load-failure fixture: undefined/NaN attention value at key='+keyIndex+' '+JSON.stringify(cell));
    if(Math.abs(cell.bias-1.20*keyIndex)>1e-6) pushError('model-load-failure fixture: fixed recency bias at key='+keyIndex+' is not 1.20*j, got '+cell.bias);
    if(Math.abs(cell.preBias+cell.bias-cell.score)>1e-6) pushError('model-load-failure fixture: dot/scale + fixed bias does not equal score at key='+keyIndex+' '+JSON.stringify(cell));
    if(cell.weight<0||cell.weight>1) pushError('model-load-failure fixture: attention weight out of [0,1] at key='+keyIndex+' '+JSON.stringify(cell));
  }
  // Same-row Block check as the learned-model fixture above: toy fallback's
  // Block stage is an explicit pass-through of context[r] (Query row), so
  // changing Key alone (Query fixed, r!=c) must not change it.
  const queryButtons=page.locator('.attention-cell-trace .trace-query-button');
  if(lastIndex>0){
    await queryButtons.nth(lastIndex).click();
    await keyButtons.nth(middleIndex).click();
    await page.waitForTimeout(60);
    const toyBlockBefore=await page.locator('.block-overview code').innerText();
    const altKey=middleIndex===0?lastIndex:0;
    await keyButtons.nth(altKey).click();
    await page.waitForTimeout(60);
    const toyBlockAfter=await page.locator('.block-overview code').innerText();
    if(toyBlockAfter!==toyBlockBefore) pushError('model-load-failure fixture: toy Block (pass-through) context changed when only Key changed (Query fixed) '+JSON.stringify({toyBlockBefore,toyBlockAfter}));

    // Full Block detail must also label the Query row, and returning to
    // attention must preserve r/c.
    await page.evaluate(()=>document.querySelector('.block-overview').click());
    await page.waitForTimeout(100);
    const toyBlockEyebrow=await page.locator('.transformer-detail-wide .detail-head .eyebrow').innerText().catch(()=>'');
    if(!toyBlockEyebrow.split('·').map(s=>s.trim()).includes('t')) pushError('model-load-failure fixture: full Block detail header does not label the Query row ("t"), got '+JSON.stringify(toyBlockEyebrow));
    await page.evaluate(()=>document.querySelector('.attention-overview').click());
    await page.waitForTimeout(100);
    const toyTraceRC=await page.locator('.attention-cell-trace').evaluate(el=>({row:Number(el.dataset.row),col:Number(el.dataset.col)}));
    if(toyTraceRC.row!==lastIndex || toyTraceRC.col!==altKey) pushError('model-load-failure fixture: returning to attention did not preserve r/c '+JSON.stringify(toyTraceRC));
  }
  await page.screenshot({path:path.join(outDir,'network-fixture-toy-fallback-attention.jpg'),type:'jpeg',quality:82,fullPage:true});

  // The "Reading order reference" <details> is a second, independently
  // reachable place that used to always describe the learned 5-step
  // sequence, even in toy fallback -- same false-model bug as the main
  // panels, just a different door to it. It must now carry an explicit
  // not-active warning and must not claim the displayed learned steps are
  // what's actually running.
  const closeAttentionDetail=page.getByRole('button',{name:'close Transformer detail'});
  if(await closeAttentionDetail.count()) await closeAttentionDetail.click();
  await page.waitForTimeout(60);
  const explainSummary=page.locator('.explain-disclosure summary');
  await explainSummary.click();
  await page.waitForTimeout(100);
  const explainWarning=page.locator('.explain .claim');
  const explainWarningCount=await explainWarning.count();
  if(explainWarningCount!==1) pushError('model-load-failure fixture: reading-order reference is missing its toy-fallback not-active warning');
  const explainWarningText=await explainWarning.innerText().catch(()=>'');
  if(!/NOT ACTIVE/i.test(explainWarningText)) pushError('model-load-failure fixture: reading-order reference warning does not say the learned steps are not active, got '+JSON.stringify(explainWarningText));
  if(!/fixed|고정/.test(explainWarningText)) pushError('model-load-failure fixture: reading-order reference warning does not name the actual (fixed bias/gain) toy behavior, got '+JSON.stringify(explainWarningText));
  const explainWarningFont=await explainWarning.evaluate(el=>parseFloat(getComputedStyle(el).fontSize)).catch(()=>null);
  if(explainWarningFont===null||explainWarningFont<14) pushError('model-load-failure fixture: reading-order reference warning is below the 14px essential-text floor, got '+explainWarningFont);
  const explainDoc=await page.evaluate(()=>({w:document.documentElement.scrollWidth,v:innerWidth}));
  if(explainDoc.w>explainDoc.v+2) pushError('model-load-failure fixture: reading-order reference warning causes page-level horizontal overflow '+JSON.stringify(explainDoc));
  await page.screenshot({path:path.join(outDir,'network-fixture-toy-fallback-reference.jpg'),type:'jpeg',quality:82,fullPage:true});

  // The deliberately-rejected model HTTP response is an expected part of
  // this fixture, not an app error; only a real JS exception should fail
  // the run. Distinguish the two explicitly in the report.
  report.interactions.modelLoadFailureFixture={
    label:'DELIBERATE LOCAL TEST FIXTURE (page.route 503 on tiny-transformer.json only) -- not a real/production outage',
    modelState, bannerPresent:bannerCount===1, unexpectedPageErrors:pageErrors
  };
  if(pageErrors.length) pushError('model-load-failure fixture: unexpected JS page error(s) (network 503 itself is expected/deliberate): '+JSON.stringify(pageErrors));

  await context.close();

  // Reload WITHOUT the route: the same app/model code must recover to the
  // real learned model -- this is not a fake/production mode, just a normal
  // reload once the (test-only) network condition is gone.
  const recoveryPage=await browser.newPage({ viewport:{width:1440,height:1000}, deviceScaleFactor:1 });
  const recoveryErrors=[];
  recoveryPage.on('pageerror', err=>recoveryErrors.push(String(err)));
  await recoveryPage.goto(baseURL, { waitUntil:'networkidle', timeout:30000 });
  await waitLearned(recoveryPage);
  await recoveryPage.waitForTimeout(300);
  const recoveredModelState=await recoveryPage.locator('main').getAttribute('data-model-state');
  if(recoveredModelState!=='learned') pushError('model-load-failure fixture: normal reload without the fault did not recover to modelState="learned", got '+JSON.stringify(recoveredModelState));
  const recoveredBanner=await recoveryPage.locator('.model-fallback-banner').count();
  if(recoveredBanner!==0) pushError('model-load-failure fixture: fallback banner did not clear after a normal (non-faulty) reload');
  await recoveryPage.locator('.explain-disclosure summary').click();
  await recoveryPage.waitForTimeout(100);
  const recoveredExplainWarning=await recoveryPage.locator('.explain .claim').count();
  if(recoveredExplainWarning!==0) pushError('model-load-failure fixture: reading-order reference still shows the toy-fallback not-active warning after learned-model recovery');
  if(recoveryErrors.length) pushError('model-load-failure fixture: unexpected page error(s) on normal-reload recovery: '+JSON.stringify(recoveryErrors));
  await recoveryPage.close();
}

let browser;
try {
  browser = await chromium.launch({ headless:true });
  await runDesktop(browser);
  await runMobile(browser);
  await runResponsiveLayoutAudit(browser,1440,'audit-desktop');
  // 1280px sits just above the 1220px lab-grid breakpoint (still the
  // full two-column state-column layout, slightly narrower than 1440) --
  // added to explicitly cover the new state-column/pipeline-disclosure
  // nesting at that width, not just the two widths either side of it.
  await runResponsiveLayoutAudit(browser,1280,'audit-1280');
  await runResponsiveLayoutAudit(browser,1024,'audit-laptop');
  await runResponsiveLayoutAudit(browser,768,'audit-tablet');
  await runResponsiveLayoutAudit(browser,390,'audit-mobile');
  await runResponsiveLayoutAudit(browser,320,'audit-mobile-small');
  // F3: guide<->shared-drawer Key/dim/Query round trip at first/middle/last
  // key and dims 0/3/5, across desktop/mobile/small-mobile widths.
  await runFollowGuideSelectionRoundtrip(browser,1440,'roundtrip-1440-first-key',0,0);
  await runFollowGuideSelectionRoundtrip(browser,390,'roundtrip-390-middle-key',3,3);
  await runFollowGuideSelectionRoundtrip(browser,320,'roundtrip-320-last-key',7,5);
  // F2: deliberate local model-load-failure fixture, isolated context.
  await runModelLoadFailureFixture(browser);
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
