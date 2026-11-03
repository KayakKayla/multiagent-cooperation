"use strict";
document.addEventListener('DOMContentLoaded', () => {
  const upload=document.getElementById('upload-form'),review=document.getElementById('review-form');if(!upload)return;
  let preview=null;
  const element=(tag,text)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el;};
  function labeled(label,control){const el=element('label',label);el.append(control);return el;}
  function option(select,value,label,selected=false){const el=new Option(label,value,false,selected);select.add(el);return el;}
  function renderPreview(data){
    preview=data;review.hidden=false;review.querySelector('.form-error').textContent='';
    review.elements.title.value=data.document.title??data.document.environment.layout??'Evaluation';review.elements.notes.value=data.document.notes??'';
    document.getElementById('review-summary').textContent=`${data.episode_count} episodes; ${data.recording_count} recorded trajectories. ${data.document.participants.length} player assignments. Preview expires ${new Date(data.expires_at*1000).toLocaleString()}.`;
    const runs=document.getElementById('run-associations');runs.replaceChildren();const runChoices=new Map();
    for(const run of data.training_runs){
      const group=element('fieldset');group.dataset.sourceRun=run.source_run_id;group.append(element('legend',run.name));
      const select=element('select');select.dataset.runAssociation=run.source_run_id;const match=data.matches.training_runs[run.source_run_id];
      option(select,'new','Create a training run',!match);
      for(const existing of data.existing_runs)option(select,existing.id,`${existing.name}${existing.id===match?' · exact source match':''}`,existing.id===match);
      runChoices.set(run.source_run_id,select);group.append(labeled('Training run association',select));
      const name=element('input');name.value=run.name;name.required=true;name.maxLength=300;name.dataset.runName=run.source_run_id;
      const notes=element('textarea');notes.value=run.notes??'';notes.rows=3;notes.maxLength=20000;notes.dataset.runNotes=run.source_run_id;
      group.append(labeled('New run name',name),labeled('New run notes',notes));
      const update=()=>{name.disabled=notes.disabled=select.value!=='new';};select.addEventListener('change',update);update();
      group.append(element('p',`Algorithm: ${run.algorithm??'unknown'} · seed: ${run.seed??'unknown'} · progress unit: ${run.progress_unit??'unknown'}. Existing run annotations are preserved.`));runs.append(group);
    }
    const checkpoints=document.getElementById('checkpoint-associations');checkpoints.replaceChildren();
    for(const cp of data.checkpoints){
      const group=element('fieldset');group.append(element('legend','Exact checkpoint'));
      const sourceRun=data.training_runs.find(r=>r.source_run_id===cp.source_run_id);
      group.append(element('p',`${sourceRun?.name??cp.source_run_id} · ${cp.progress??'unknown progress'} ${sourceRun?.progress_unit??''}`));
      const hash=element('code',cp.sha256);hash.className='hash';group.append(hash);
      const players=data.document.participants.filter(p=>p.source_checkpoint_id===cp.source_checkpoint_id).map(p=>p.agent_id);group.append(element('p',`Assigned to: ${players.join(', ')}`));
      const select=element('select');select.dataset.checkpointAssociation=cp.source_checkpoint_id;const match=data.matches.checkpoints[cp.source_checkpoint_id];
      const suggestions=new Set(data.checkpoint_suggestions?.[cp.source_checkpoint_id]??[]);
      const candidates=data.existing_checkpoints.filter(x=>(x.id===match||suggestions.has(x.id))&&x.sha256===cp.sha256&&(x.progress??null)===(cp.progress??null));
      const runChoice=runChoices.get(cp.source_run_id);
      const help=element('p');help.className='muted';
      if(candidates.length){
        help.append(document.createTextNode('Exact checkpoint candidates (weights hash, progress, and training settings match): '));
        for(const [i,candidate]of candidates.entries()){
          if(i)help.append(document.createTextNode('; '));
          const link=element('a',candidate.label);link.href=Site.path(`/training-runs/${encodeURIComponent(candidate.run_id)}`);help.append(link);
        }
        help.append(document.createTextNode('. To reuse a candidate from an unfamiliar source identity, explicitly select its training run above and then select the checkpoint.'));
      }else help.textContent='No compatible exact checkpoint candidates. Register this checkpoint with the reviewed training run.';
      function updateCandidates(initial=false){
        const previous=initial?(match??'new'):select.value;select.replaceChildren();
        option(select,'new','Register uploaded checkpoint',true);
        for(const candidate of candidates){
          const entry=option(select,candidate.id,`${candidate.label}${candidate.id===match?' · exact source match':' · suggested exact checkpoint'}${runChoice.value!==candidate.run_id?' · select its training run first':''}`);
          entry.disabled=runChoice.value!==candidate.run_id;
          if(previous===candidate.id&&!entry.disabled)select.value=candidate.id;
        }
      }
      updateCandidates(true);runChoice.addEventListener('change',()=>updateCandidates());
      group.append(help,labeled('Checkpoint association',select));checkpoints.append(group);
    }
    const conflict=data.conflicts??[];review.querySelector('[type=submit]').disabled=conflict.length>0;
    if(conflict.length){review.querySelector('.form-error').textContent=conflict.join('\n');Site.toast('Source metadata conflicts must be corrected before import.',true);}
    if(data.duplicate_id){const p=element('p','This source evaluation is already published. '),a=element('a','View existing evaluation');a.href=Site.path(`/evaluations/${encodeURIComponent(data.duplicate_id)}`);p.append(a);checkpoints.prepend(p);}
    review.scrollIntoView({block:'start',behavior:'auto'});review.elements.title.focus();
  }
  upload.addEventListener('submit',async event=>{
    event.preventDefault();const file=document.getElementById('evaluation-file').files[0];if(!file)return;
    const button=upload.querySelector('[type=submit]');button.disabled=true;review.hidden=true;
    try{if(!file.name.toLowerCase().endsWith('.json'))throw new Error('Select a plain .json evaluation file.');if(file.size>20971520)throw new Error('The file exceeds 20 MiB (20,971,520 bytes). Export fewer episodes in a new evaluation; no states are silently discarded.');
      const result=await Site.request('/api/uploads',{method:'POST',headers:{'Content-Type':'application/json'},body:file});upload.querySelector('.form-error').textContent='';renderPreview(result);Site.toast('Validated. Review checkpoint assignments before publishing.');
    }catch(error){Site.showError(upload,error);}finally{button.disabled=false;}
  });
  review.addEventListener('submit',async event=>{
    event.preventDefault();const button=review.querySelector('[type=submit]');button.disabled=true;
    const body={preview_id:preview.id,title:review.elements.title.value,notes:review.elements.notes.value,run_associations:{},checkpoint_associations:{},run_annotations:{}};
    review.querySelectorAll('[data-run-association]').forEach(el=>body.run_associations[el.dataset.runAssociation]=el.value);
    review.querySelectorAll('[data-checkpoint-association]').forEach(el=>body.checkpoint_associations[el.dataset.checkpointAssociation]=el.value);
    for(const group of review.querySelectorAll('[data-source-run]'))body.run_annotations[group.dataset.sourceRun]={name:group.querySelector('[data-run-name]').value,notes:group.querySelector('[data-run-notes]').value};
    try{const result=await Site.request('/api/imports',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const resultURL=Site.recordURL(result.url,'evaluations');sessionStorage.setItem('site-notice',result.duplicate?'This evaluation was already published.':'Evaluation published.');Site.navigate(resultURL);}catch(error){Site.showError(review,error);button.disabled=false;}
  });
});
