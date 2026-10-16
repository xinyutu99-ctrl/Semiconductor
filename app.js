(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const els = {
    input: $('dataset-input'), drop: $('dropzone'), summary: $('dataset-summary'), mapping: $('class-mapping'), warning: $('dataset-warning'), splitBtn: $('split-button'), split: $('split-visual'), trainBtn: $('train-button'), progress: $('training-progress'), progressLabel: $('progress-label'), progressValue: $('progress-value'), progressFill: $('progress-fill'), progressMeta: $('progress-meta'), curve: $('curve-placeholder'), reset: $('reset-button'), runtime: $('runtime-note'), tag: $('result-tag'), metrics: $('metric-list'), matrix: $('matrix'), matrixFoot: $('matrix-foot'), resultActions: $('result-actions'), exportBtn: $('export-button'), copyBtn: $('copy-button'), feedback: $('action-feedback'), predictInput: $('predict-input'), prediction: $('prediction-result'), toast: $('toast')
  };
  const IMG_SIZE = 64, MAX_IMAGES = 1200, SEED = 42917;
  let records = [], rawRecords = [], splits = null, model = null, report = null, busy = false, toastTimer, datasetMode = 'folders', selectedNormalLabel = '3', sourceImageCount = 0;
  const imageExt = /\.(jpe?g|png|webp|bmp)$/i;
  const classNames = {normal: 0, defective: 1};
  function parseDelimited(text, delimiter) {
    const rows=[]; let row=[], value='', quoted=false;
    for(let i=0;i<text.length;i++) { const ch=text[i];
      if(ch==='"' && quoted && text[i+1]==='"') { value+='"'; i++; }
      else if(ch==='"') quoted=!quoted;
      else if(ch===delimiter && !quoted) { row.push(value); value=''; }
      else if((ch==='\n'||ch==='\r')&&!quoted) { if(ch==='\r'&&text[i+1]==='\n')i++; row.push(value); if(row.some(cell=>cell.trim()))rows.push(row); row=[]; value=''; }
      else value+=ch;
    }
    row.push(value); if(row.some(cell=>cell.trim()))rows.push(row); return rows;
  }
  function stableHash(value) { let h=2166136261; for(let i=0;i<value.length;i++){h^=value.charCodeAt(i);h=Math.imul(h,16777619);} return h>>>0; }
  function balancedSubset(source) {
    const byLabel=new Map(); for(const r of source){if(!byLabel.has(r.rawLabel))byLabel.set(r.rawLabel,[]);byLabel.get(r.rawLabel).push(r);}
    const normalRows=byLabel.get(Number(selectedNormalLabel))||[], defectRows=source.filter(r=>r.rawLabel!==Number(selectedNormalLabel));
    const n=Math.min(normalRows.length,defectRows.length,Math.floor(MAX_IMAGES/2));
    const take=(items,count)=>items.slice().sort((a,b)=>stableHash(a.name)-stableHash(b.name)).slice(0,count);
    return [...take(normalRows,n),...take(defectRows,n)].map(r=>({...r,label:r.rawLabel===Number(selectedNormalLabel)?0:1}));
  }
  function notify(message) { els.toast.textContent = message; els.toast.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => els.toast.classList.remove('show'), 3000); }
  function setWarning(message) { els.warning.textContent = message || ''; els.warning.hidden = !message; }
  function pct(n) { return `${(n * 100).toFixed(1)}%`; }
  function seededRandom(seed) { let x = seed >>> 0; return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; }; }
  function shuffle(array, random) { for (let i = array.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [array[i], array[j]] = [array[j], array[i]]; } return array; }
  function parseFiles(files) {
    const parsed = [], problems = [];
    for (const file of files) {
      if (!imageExt.test(file.name)) continue;
      const parts = (file.webkitRelativePath || file.name).split('/').filter(Boolean);
      const classIndex = parts.findIndex(p => Object.hasOwn(classNames, p.toLowerCase()));
      if (classIndex < 0 || !parts[classIndex + 1] || !parts[classIndex + 2]) { problems.push(file.name); continue; }
      const label = classNames[parts[classIndex].toLowerCase()];
      const sample = parts[classIndex + 1];
      parsed.push({file, label, sample, group: sample, name: file.webkitRelativePath || file.name});
    }
    return {parsed, problems};
  }
  function updateSummary() {
    const normal = records.filter(r => r.label === 0).length, defect = records.length - normal;
    const ng = new Set(records.filter(r => r.label === 0).map(r => r.group)).size, dg = new Set(records.filter(r => r.label === 1).map(r => r.group)).size;
    els.summary.innerHTML = records.length ? `<div class="summary-stats"><span><b>${records.length}</b> images${datasetMode==='carinthia'?` of ${sourceImageCount}`:''}</span><span><b>${ng}</b> normal ${datasetMode==='carinthia'?'images':'samples'}</span><span><b>${dg}</b> defect ${datasetMode==='carinthia'?'images':'samples'}</span></div><div>${normal} normal · ${defect} defective${datasetMode==='carinthia'?` · deterministic balanced subset`:''}</div>` : '<span class="summary-empty">No dataset loaded</span>';
    const ok = ng >= 4 && dg >= 4 && records.length >= 16 && records.length <= MAX_IMAGES;
    els.splitBtn.disabled = !ok || busy;
    els.splitBtn.textContent = ok ? 'Create fixed split' : 'Fix dataset to continue';
    if (records.length && !ok && !els.warning.hidden) return;
    if (records.length && (ng < 4 || dg < 4)) setWarning('Need at least 4 separate image/sample groups in each class so train, validation and test can each include both labels.');
    else if (records.length > MAX_IMAGES) setWarning(`This run is limited to ${MAX_IMAGES} images for browser memory. Reduce the dataset and choose it again.`);
    else if (records.length && !records.some(r => r.label === 0) || records.length && !records.some(r => r.label === 1)) setWarning('Include both normal and defective images.');
    if (ok) setWarning('');
  }
  function loadDataset(files) {
    if (busy) return;
    const list=[...files], csv=list.find(f=>/^carinthia\.csv$/i.test(f.name));
    const clearOld=()=>{ records=[];rawRecords=[];sourceImageCount=0;splits=null;report=null;els.trainBtn.disabled=true; if(model){model.dispose();model=null;} els.predictInput.disabled=true;els.prediction.textContent='Train a model to enable single-image prediction.';els.tag.textContent='AWAITING RUN';els.tag.classList.remove('complete');els.resultActions.hidden=true;els.mapping.hidden=true;resetReport(); };
    clearOld();
    if(csv) { loadCarinthia(csv,list).catch(error=>{setWarning(error.message);updateSummary();}); return; }
    datasetMode='folders';
    const {parsed, problems} = parseFiles(list);
    records = parsed; sourceImageCount=parsed.length;
    if (!parsed.length) setWarning('No labeled images found. Use folders named normal and defective, each containing separate sample folders.');
    else if (problems.length) setWarning(`${problems.length} image(s) were skipped. Use normal/sample-id/image.jpg and defective/sample-id/image.jpg.`);
    updateSummary();
    if (parsed.length > MAX_IMAGES) { records = []; setWarning(`This browser run supports up to ${MAX_IMAGES} images. Choose a smaller dataset.`); updateSummary(); }
  }
  async function loadCarinthia(csv, files) {
    datasetMode='carinthia'; selectedNormalLabel='3';
    const text=await csv.text(), delimiter=(text.split(/\r?\n/,1)[0].match(/;/g)||[]).length>(text.split(/\r?\n/,1)[0].match(/,/g)||[]).length?';':',';
    const rows=parseDelimited(text,delimiter); if(!rows.length)throw new Error('carinthia.csv is empty or unreadable.');
    const headers=rows[0].map(v=>v.trim().replace(/^\uFEFF/,'')), fileCol=headers.indexOf('file_name'), labelCol=headers.indexOf('label');
    if(fileCol<0||labelCol<0)throw new Error('carinthia.csv must contain file_name and label columns.');
    const images=new Map(files.filter(f=>imageExt.test(f.name)).map(f=>[f.name.toLowerCase(),f]));
    rawRecords=[];let missing=0,invalid=0;
    for(const row of rows.slice(1)) {const name=(row[fileCol]||'').trim(), rawLabel=Number((row[labelCol]||'').trim()), file=images.get(name.split('/').pop().toLowerCase());
      if(!file){missing++;continue;} if(!Number.isInteger(rawLabel)||rawLabel<1||rawLabel>6){invalid++;continue;}
      rawRecords.push({file,rawLabel,name:file.webkitRelativePath||name,group:file.webkitRelativePath||name});
    }
    sourceImageCount=rawRecords.length;
    if(!sourceImageCount)throw new Error('No CSV rows matched image files. Select the data folder that contains both carinthia.csv and images/.');
    records=balancedSubset(rawRecords);
    els.mapping.hidden=false;
    els.mapping.innerHTML=`<label for="normal-label-select"><strong>Binary label rule</strong><br>Which Carinthia class is normal?</label> <select id="normal-label-select" aria-label="Choose the normal Carinthia class">${[1,2,3,4,5,6].map(id=>`<option value="${id}" ${id===3?'selected':''}>Class ${id}</option>`).join('')}</select><div id="mapping-rule">Class 3 is currently normal; all other classes count as defective. Verify this against the original label legend.</div>`;
    $('normal-label-select').addEventListener('change',e=>{selectedNormalLabel=e.target.value;$('mapping-rule').textContent=`Class ${selectedNormalLabel} is currently normal; all other classes count as defective. Verify this against the original label legend.`;records=balancedSubset(rawRecords);splits=null;report=null;if(model){model.dispose();model=null;}els.predictInput.disabled=true;els.prediction.textContent='Train a model to enable single-image prediction.';els.trainBtn.disabled=true;els.splitBtn.disabled=false;els.splitBtn.textContent='Create fixed split';els.resultActions.hidden=true;els.tag.textContent='AWAITING RUN';els.tag.classList.remove('complete');resetReport();updateSummary();});
    const counts=Object.fromEntries([1,2,3,4,5,6].map(id=>[id,rawRecords.filter(r=>r.rawLabel===id).length]));
    els.mapping.insertAdjacentHTML('beforeend',`<div>CSV label counts: ${Object.entries(counts).map(([id,n])=>`${id}: ${n}`).join(' · ')}. The dataset has no wafer/sample IDs; the test split is image-level. <a href="https://zenodo.org/records/10715190/files/carinthia_dataset.html?download=1" target="_blank" rel="noreferrer">Open official label legend</a>.</div>`);
    updateSummary();
    if(missing||invalid)setWarning(`${missing} CSV row(s) had no matching image and ${invalid} row(s) had an invalid class label; those rows were skipped.`);
  }
  function splitGroups() {
    const random = seededRandom(SEED), all = {train: [], validation: [], test: []}, assignment = new Map();
    for (const label of [0, 1]) {
      const groups = [...new Set(records.filter(r => r.label === label).map(r => r.group))];
      shuffle(groups, random);
      const testN = Math.max(1, Math.round(groups.length * .15));
      const valN = Math.max(1, Math.round(groups.length * .15));
      groups.forEach((group,index) => {
        const wanted = index < testN ? 'test' : index < testN + valN ? 'validation' : 'train';
        if (!assignment.has(group)) assignment.set(group, wanted);
      });
    }
    for (const r of records) all[assignment.get(r.group)].push(r);
    if (['train','validation','test'].some(split => [0,1].some(label => !all[split].some(r => r.label === label)))) throw new Error('These sample groups cannot provide both classes in every split. Check that each wafer/sample ID is consistent, and add more independently labeled samples.');
    return all;
  }
  function renderSplit() {
    const lengths = [splits.train.length, splits.validation.length, splits.test.length], total = lengths.reduce((a,b)=>a+b,0);
    const deg1 = Math.round(lengths[0] / total * 360), deg2 = Math.round(lengths[1] / total * 360), color2 = deg1 + deg2;
    els.split.innerHTML = `<div class="split-chart" style="background:conic-gradient(#56856a 0deg ${deg1}deg,#a9c1a8 ${deg1}deg ${color2}deg,#d49a5c ${color2}deg 360deg)"><div class="chart-label"><strong>${new Set(records.map(r=>r.group)).size}</strong><span>${datasetMode==='carinthia'?'IMAGES':'SAMPLES'}</span></div></div>`;
    const groups = Object.fromEntries(['train','validation','test'].map(key => [key, new Set(splits[key].map(r => r.group)).size]));
    $('split-note').innerHTML = `<span aria-hidden="true">✓</span><p>Fixed seed ${SEED}. ${datasetMode==='carinthia'?'Image-level split: no wafer/sample ID is present, so this does not test unseen wafers.':`${groups.train} / ${groups.validation} / ${groups.test} samples are isolated across train, validation and test.`}</p>`;
    els.splitBtn.textContent = 'Split locked'; els.splitBtn.disabled = true;
    els.trainBtn.disabled = typeof tf === 'undefined';
    if (typeof tf === 'undefined') els.runtime.textContent = 'TensorFlow.js could not load. Check your internet connection, then reload the page.';
    else els.runtime.textContent = datasetMode==='carinthia'?`${total} images · balanced sample from ${sourceImageCount} labeled rows · image-level test split`:`${total} images · ${groups.test} unseen test samples · split fixed for this run`;
  }
  function resetReport() {
    els.metrics.innerHTML = '<div class="metric-row"><span>Accuracy</span><strong>—</strong><small>overall correctness</small></div><div class="metric-row"><span>Precision</span><strong>—</strong><small>predicted defects that are real</small></div><div class="metric-row"><span>Recall</span><strong>—</strong><small>defects successfully found</small></div><div class="metric-row"><span>False negative rate</span><strong>—</strong><small>defects missed by the model</small></div>';
    els.matrix.querySelectorAll('.matrix-cell').forEach(c => { c.classList.remove('has-value'); c.querySelector('strong').textContent = '—'; });
    els.matrixFoot.textContent = 'Run an evaluation to populate the matrix.';
    els.curve.querySelector('.curve-line').innerHTML = ''; els.curve.querySelector('.curve-line').style.display = 'none'; els.curve.querySelector('.curve-empty').hidden = false;
  }
  function setBusy(value) { busy = value; els.input.disabled = value; els.splitBtn.disabled = value || !!splits; els.trainBtn.disabled = value || !splits || !!report || typeof tf === 'undefined'; }
  function readTensor(record) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        try {
          const canvas = document.createElement('canvas'); canvas.width = canvas.height = IMG_SIZE;
          const ctx = canvas.getContext('2d', {willReadFrequently: true});
          const scale = Math.max(IMG_SIZE / image.width, IMG_SIZE / image.height), w = image.width * scale, h = image.height * scale;
          ctx.drawImage(image, (IMG_SIZE - w) / 2, (IMG_SIZE - h) / 2, w, h);
          const data = ctx.getImageData(0,0,IMG_SIZE,IMG_SIZE).data, values = new Float32Array(IMG_SIZE*IMG_SIZE*3);
          for (let i=0,j=0;i<data.length;i+=4) { values[j++] = data[i]/255; values[j++] = data[i+1]/255; values[j++] = data[i+2]/255; }
          URL.revokeObjectURL(image.src); resolve({values, label:record.label, group:record.group});
        } catch (e) { reject(e); }
      };
      image.onerror = () => reject(new Error(`Could not decode ${record.name}`));
      image.src = URL.createObjectURL(record.file);
    });
  }
  function augmentPixels(values) {
    const out = new Float32Array(values.length), turn = Math.floor(Math.random()*4), flip = Math.random()<.5;
    for(let y=0;y<IMG_SIZE;y++) for(let x=0;x<IMG_SIZE;x++) {
      let sx=x, sy=y;
      if(flip) sx=IMG_SIZE-1-sx;
      if(turn===1) [sx,sy]=[sy,IMG_SIZE-1-sx];
      else if(turn===2) {sx=IMG_SIZE-1-sx;sy=IMG_SIZE-1-sy;}
      else if(turn===3) [sx,sy]=[IMG_SIZE-1-sy,sx];
      const src=(sy*IMG_SIZE+sx)*3, dst=(y*IMG_SIZE+x)*3;
      out[dst]=values[src];out[dst+1]=values[src+1];out[dst+2]=values[src+2];
    }
    return out;
  }
  function showProgress(label, value, meta) { els.progress.hidden = false; els.progressLabel.textContent = label; els.progressValue.textContent = `${Math.round(value)}%`; els.progressFill.style.width = `${Math.max(0,Math.min(100,value))}%`; els.progressMeta.textContent = meta; }
  function drawCurves(losses, accuracies) {
    const w = 500, h = 110, pad = 8;
    const path = arr => arr.map((v,i) => `${i?'L':'M'}${pad+i*(w-2*pad)/Math.max(1,arr.length-1)},${h-pad-v*(h-2*pad)}`).join(' ');
    const maxLoss = Math.max(...losses, .1), normLoss = losses.map(v => 1-v/maxLoss);
    els.curve.querySelector('.curve-line').innerHTML = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="Training loss and accuracy curves"><path d="${path(normLoss)}" fill="none" stroke="#d49a5c" stroke-width="2"/><path d="${path(accuracies)}" fill="none" stroke="#39805e" stroke-width="2"/></svg>`;
    els.curve.querySelector('.curve-line').style.display = 'block'; els.curve.querySelector('.curve-empty').hidden = true;
  }
  function createModel() {
    const m = tf.sequential();
    m.add(tf.layers.conv2d({inputShape:[IMG_SIZE,IMG_SIZE,3],filters:12,kernelSize:3,activation:'relu',padding:'same'}));
    m.add(tf.layers.maxPooling2d({poolSize:2}));
    m.add(tf.layers.conv2d({filters:24,kernelSize:3,activation:'relu',padding:'same'}));
    m.add(tf.layers.maxPooling2d({poolSize:2}));
    m.add(tf.layers.flatten()); m.add(tf.layers.dense({units:32,activation:'relu'})); m.add(tf.layers.dropout({rate:.25})); m.add(tf.layers.dense({units:1,activation:'sigmoid'}));
    m.compile({optimizer:tf.train.adam(.001),loss:'binaryCrossentropy',metrics:['accuracy']}); return m;
  }
  async function runTraining() {
    if (busy || !splits || typeof tf === 'undefined') return;
    setBusy(true); report = null; els.tag.textContent = 'TRAINING'; els.tag.classList.remove('complete'); els.resultActions.hidden = true; resetReport();
    const losses = [], accuracies = [];
    let tensors=[];
    try {
      await tf.ready();
      const all = [...splits.train,...splits.validation,...splits.test], decoded=[];
      showProgress('Preparing images…', 2, 'Resize 64 × 64 · RGB values normalized to 0–1');
      for (let i=0;i<all.length;i++) { decoded.push(await readTensor(all[i])); if (i%8===0 || i===all.length-1) { showProgress(`Preparing images · ${i+1}/${all.length}`, 3+12*(i+1)/all.length, 'Your images stay on this device.'); await tf.nextFrame(); } }
      const tensorsFor = (subset, augment=false) => {
        const rows=subset.map(r=>decoded[all.indexOf(r)]);
        if(augment) rows.push(...rows.map(row=>({...row,values:augmentPixels(row.values)})));
        const pixelCount=IMG_SIZE*IMG_SIZE*3, flat=new Float32Array(rows.length*pixelCount);
        rows.forEach((r,i)=>flat.set(r.values,i*pixelCount));
        const xs=tf.tensor4d(flat,[rows.length,IMG_SIZE,IMG_SIZE,3]);
        const ys=tf.tensor2d(rows.map(r=>[r.label]),[rows.length,1]); tensors.push(xs,ys); return {xs,ys,rows};
      };
      const train=tensorsFor(splits.train,true), val=tensorsFor(splits.validation), test=tensorsFor(splits.test);
      model=createModel();
      const maxEpochs=14;
      showProgress('Training CNN…',15,'Training images are augmented with random flips and 90° rotations.');
      await model.fit(train.xs,train.ys,{epochs:maxEpochs,batchSize:Math.min(24,train.rows.length),shuffle:true,validationData:[val.xs,val.ys],callbacks:{onEpochEnd:async(epoch,logs)=>{
        losses.push(logs.val_loss ?? logs.loss); accuracies.push(logs.val_acc ?? logs.acc ?? 0);
        showProgress(`Training · epoch ${epoch+1} of ${maxEpochs}`,15+epoch/maxEpochs*65,`Train loss ${logs.loss.toFixed(3)} · validation accuracy ${pct(logs.val_acc ?? logs.acc ?? 0)}`); drawCurves(losses,accuracies); await tf.nextFrame();
      }}});
      showProgress('Evaluating held-out samples…',84,'The fixed test split has not been used during training.');
      const predicted = model.predict(test.xs); const scores=await predicted.data(); predicted.dispose();
      const counts={tn:0,fp:0,fn:0,tp:0};
      scores.forEach((score,i)=>{const actual=test.rows[i].label, guess=score>=.5?1:0; counts[actual===0?(guess===0?'tn':'fp'):(guess===1?'tp':'fn')]++;});
      const n=Object.values(counts).reduce((a,b)=>a+b,0), precision=counts.tp/(counts.tp+counts.fp)||0, recall=counts.tp/(counts.tp+counts.fn)||0, accuracy=(counts.tp+counts.tn)/n, fnr=counts.fn/(counts.fn+counts.tp)||0;
      report={counts,accuracy,precision,recall,fnr,n,groups:new Set(test.rows.map(r=>r.group)).size,losses,accuracies,seed:SEED};
      renderReport(); els.predictInput.disabled=false; els.prediction.textContent='Choose a microscope image to classify.'; showProgress('Evaluation complete',100,`${n} held-out images${datasetMode==='carinthia'?' (image-level split)':' from unseen sample folders'}.`); els.tag.textContent='RUN COMPLETE'; els.tag.classList.add('complete'); els.resultActions.hidden=false; els.reset.hidden=false; notify('Held-out evaluation complete.');
      tensors.forEach(t=>t.dispose()); tensors=[];
    } catch(error) {
      if(model){model.dispose();model=null;}
      console.error(error); els.tag.textContent='RUN FAILED'; els.runtime.textContent=error.message || 'Training failed. Try a smaller dataset or reload the page.'; showProgress('Run stopped',0,'No test result was produced.'); notify('Training stopped. See the message below the model card.');
    } finally { tensors.forEach(t=>t.dispose()); setBusy(false); els.reset.hidden = !report; }
  }
  function renderReport() {
    const rows=[['Accuracy',pct(report.accuracy),'overall correctness'],['Precision',pct(report.precision),'predicted defects that are real'],['Recall',pct(report.recall),'defects successfully found'],['False negative rate',pct(report.fnr),'defects missed by the model']];
    els.metrics.innerHTML=rows.map(([name,value,desc])=>`<div class="metric-row has-value"><span>${name}</span><strong>${value}</strong><small>${desc}</small></div>`).join('');
    for(const [key,value] of Object.entries(report.counts)){const cell=els.matrix.querySelector(`.${key}`);cell.classList.add('has-value');cell.querySelector('strong').textContent=value;}
    els.matrixFoot.textContent=datasetMode==='carinthia'?`${report.n} held-out images · image-level split · fixed seed ${SEED}`:`${report.n} held-out images · ${report.groups} unseen sample folders · fixed seed ${SEED}`;
  }
  function splitButton() {
    try { splits=splitGroups(); renderSplit(); notify(datasetMode==='carinthia'?'Image-level split locked.':'Sample-level split locked.'); }
    catch(error){ setWarning(error.message); splits=null; els.trainBtn.disabled=true; }
  }
  function getReportText() { return `WaferScope held-out evaluation\nTest images: ${report.n}\nSplit: ${datasetMode==='carinthia'?'image-level (wafer/sample IDs unavailable)':'sample-level'}\n${datasetMode==='carinthia'?`Binary labels: class ${selectedNormalLabel} = normal; other classes = defective\nBalanced subset: ${records.length} of ${sourceImageCount} matched images\n`:''}Seed: ${report.seed}\nAccuracy: ${pct(report.accuracy)}\nPrecision: ${pct(report.precision)}\nRecall: ${pct(report.recall)}\nFalse negative rate: ${pct(report.fnr)}\nConfusion matrix (actual rows: normal, defective; predicted columns: normal, defective)\n${report.counts.tn},${report.counts.fp}\n${report.counts.fn},${report.counts.tp}`; }
  function exportCsv() {
    const rows=[['WaferScope held-out evaluation'],['Metric','Value'],['Test images',report.n],['Split type',datasetMode==='carinthia'?'Image-level; wafer/sample IDs unavailable':'Sample-level'],...(datasetMode==='carinthia'?[['Binary label rule',`Class ${selectedNormalLabel} = normal; all other labels = defective`],['Balanced subset',`${records.length} of ${sourceImageCount} matched images`]]:[]),['Unseen test samples',datasetMode==='carinthia'?'Not available':report.groups],['Split seed',report.seed],['Accuracy',report.accuracy],['Precision',report.precision],['Recall',report.recall],['False negative rate',report.fnr],[],['Confusion matrix','Predicted normal','Predicted defective'],['Actual normal',report.counts.tn,report.counts.fp],['Actual defective',report.counts.fn,report.counts.tp]];
    const csv=rows.map(row=>row.map(v=>`"${String(v??'').replaceAll('"','""')}"`).join(',')).join('\r\n'); const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})); const a=document.createElement('a');a.href=url;a.download='waferscope-test-report.csv';a.click();URL.revokeObjectURL(url);els.feedback.textContent='CSV downloaded';
  }
  async function copySummary() { try { await navigator.clipboard.writeText(getReportText()); els.feedback.textContent='Summary copied'; } catch { els.feedback.textContent='Clipboard unavailable in this browser'; } }
  async function predict(file) {
    if(!model||!file)return; els.prediction.textContent='Analyzing image…';
    try { const item=await readTensor({file,name:file.name,label:0,group:'prediction'}); const x=tf.tensor4d([item.values],[1,IMG_SIZE,IMG_SIZE,3]); const out=model.predict(x); const score=(await out.data())[0]; x.dispose();out.dispose(); const positive=score>=.5, confidence=positive?score:1-score; els.prediction.innerHTML=`Prediction: <strong>${positive?'Defective':'Normal'}</strong> · ${pct(confidence)} confidence`; els.prediction.classList.add('prediction-ready'); }
    catch(e){els.prediction.textContent=e.message||'Could not read this image.';}
  }
  els.input.addEventListener('change',e=>loadDataset([...e.target.files]));
  els.drop.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target===els.drop){e.preventDefault();els.input.click();}});
  els.drop.addEventListener('dragover',e=>{e.preventDefault();els.drop.classList.add('dragover');});
  els.drop.addEventListener('dragleave',()=>els.drop.classList.remove('dragover'));
  els.drop.addEventListener('drop',e=>{e.preventDefault();els.drop.classList.remove('dragover');const files=[...e.dataTransfer.files];if(files.length)loadDataset(files);else setWarning('This browser did not expose the dropped folder. Use Choose dataset folder instead.');});
  els.splitBtn.addEventListener('click',splitButton); els.trainBtn.addEventListener('click',runTraining); els.predictInput.addEventListener('change',e=>predict(e.target.files[0])); els.exportBtn.addEventListener('click',exportCsv); els.copyBtn.addEventListener('click',copySummary);
  els.reset.addEventListener('click',()=>{if(model)model.dispose();model=null;report=null;splits=null;els.progress.hidden=true;els.reset.hidden=true;els.predictInput.disabled=true;els.prediction.textContent='Train a model to enable single-image prediction.';els.tag.textContent='AWAITING RUN';els.tag.classList.remove('complete');els.resultActions.hidden=true;els.splitBtn.disabled=false;els.splitBtn.textContent='Create fixed split';els.trainBtn.disabled=true;els.runtime.textContent='Run cleared. Your loaded dataset is ready to split again.';resetReport();notify('Run cleared. Dataset retained.');});
  if(typeof tf==='undefined') window.addEventListener('load',()=>{if(typeof tf==='undefined')els.runtime.textContent='TensorFlow.js could not load. Check your internet connection and reload the page.';});
})();
