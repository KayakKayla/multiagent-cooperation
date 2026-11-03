const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const{test}=require('node:test');

class Element{
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.value='';this.disabled=false;this.selected=false;this.textContent='';}
  append(...children){this.children.push(...children);}
  prepend(...children){this.children.unshift(...children);}
  replaceChildren(...children){this.children=children;if(this.tag==='select')this.value='';}
  add(option){this.append(option);if(option.selected||this.children.length===1)this.value=option.value;}
  addEventListener(event,callback){(this.listeners[event]??=[]).push(callback);}
  async fire(event){for(const fn of this.listeners[event]??[])await fn({preventDefault(){},currentTarget:this});}
  focus(){}scrollIntoView(){}
  querySelector(selector){return this.querySelectorAll(selector)[0]??null;}
  querySelectorAll(selector){const matches=el=>selector==='[type=submit]'?el.type==='submit':selector==='.form-error'?el.className==='form-error':selector==='[data-source-run]'?'sourceRun'in el.dataset:selector==='[data-run-association]'?'runAssociation'in el.dataset:selector==='[data-checkpoint-association]'?'checkpointAssociation'in el.dataset:selector==='[data-run-name]'?'runName'in el.dataset:selector==='[data-run-notes]'?'runNotes'in el.dataset:selector==='[data-checkpoint-preview]'?'checkpointPreview'in el.dataset:false;return this.children.flatMap(el=>typeof el==='string'?[]:[...(matches(el)?[el]:[]),...el.querySelectorAll(selector)]);}
}
function boot(matches={training_runs:{},checkpoints:{}}, omittedProgress=false, basePath='', resultURL='/evaluations/e1'){
  const upload=new Element('form'),review=new Element('form'),runHost=new Element('div'),checkpointHost=new Element('div'),summary=new Element('p'),file=new Element('input');file.files=[{name:'evaluation.json',size:100}];
  for(const form of[upload,review]){const button=new Element('button');button.type='submit';const error=new Element('p');error.className='form-error';form.append(button,error);}
  review.append(runHost,checkpointHost);review.elements={title:new Element('input'),notes:new Element('textarea')};
  const ids={'upload-form':upload,'review-form':review,'run-associations':runHost,'checkpoint-associations':checkpointHost,'review-summary':summary,'evaluation-file':file};
  const doc={querySelector:selector=>selector==='meta[name="base-path"]'?{content:basePath}:null,createElement:tag=>new Element(tag),createTextNode:text=>String(text),getElementById:id=>ids[id],addEventListener(_event,fn){this.ready=fn;}};
  const run={source_run_id:'unfamiliar-run',name:'Uploaded policy',progress_unit:'environment_steps',notes:''};
  const cp={source_checkpoint_id:'unfamiliar-checkpoint',source_run_id:run.source_run_id,progress:100,sha256:'a'.repeat(64)};
  const data={id:'preview',expires_at:2000000000,episode_count:2,recording_count:2,document:{environment:{layout:'demo'},participants:[{agent_id:'a0',source_checkpoint_id:cp.source_checkpoint_id}]},training_runs:[run],checkpoints:[cp],matches,existing_runs:[{id:'run-1',name:'Baseline'}],existing_checkpoints:[
    {id:'candidate-1',run_id:'run-1',run_name:'Baseline',label:'Baseline · 100 environment_steps',progress:100,sha256:cp.sha256},
    {id:'wrong-progress',run_id:'run-1',label:'Baseline · 200 environment_steps',progress:200,sha256:cp.sha256},
    {id:'incompatible',run_id:'other-run',label:'Unrelated model · 100 environment_steps',progress:100,sha256:cp.sha256}],checkpoint_suggestions:{[cp.source_checkpoint_id]:['candidate-1','wrong-progress']}};
  if(omittedProgress){delete cp.progress;data.existing_checkpoints[0].progress=null;}
  const requests=[],navigation=[];const context={document:doc,Option:class extends Element{constructor(label,value,_default=false,selected=false){super('option');this.textContent=label;this.value=value;this.selected=selected;}},Site:{request:async(url,options)=>{requests.push({url,options});return url==='/api/uploads'?data:{url:resultURL};},toast(){},showError(_form,error){throw error;}},sessionStorage:{setItem(){}},location:{assign(value){navigation.push(value);}},Date,Map,Set};
  const stubSite=context.Site;context.window=context;vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/site.js'),'utf8'),context);Object.assign(context.Site,stubSite);
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/upload.js'),'utf8'),context);doc.ready();
  return{upload,review,runHost,checkpointHost,requests,navigation,data};
}

test('suggested exact checkpoint explains run and requires explicit unfamiliar-run association',async()=>{
  const ui=boot();await ui.upload.fire('submit');
  const run=ui.review.querySelector('[data-run-association]'),cp=ui.review.querySelector('[data-checkpoint-association]');
  assert.equal(run.value,'new');assert.equal(cp.value,'new');assert.equal(cp.children.length,2);
  assert.match(cp.children[1].textContent,/Baseline · 100 steps.*suggested exact checkpoint/);assert.equal(cp.children[1].disabled,true);
  assert.equal(cp.children.some(option=>option.value==='wrong-progress'||option.value==='incompatible'),false);
  assert.equal(ui.checkpointHost.children[0].children.some(child=>child.tag==='p'&&child.children?.some(node=>typeof node==='string'&&node.includes('explicitly select its training run'))),true);
  run.value='run-1';await run.fire('change');assert.equal(cp.children[1].disabled,false);assert.equal(cp.value,'new');
  cp.value='candidate-1';await ui.review.fire('submit');const body=JSON.parse(ui.requests[1].options.body);
  assert.equal(body.run_associations['unfamiliar-run'],'run-1');assert.equal(body.checkpoint_associations['unfamiliar-checkpoint'],'candidate-1');
});

test('established exact source identities retain automatic run and checkpoint matches',async()=>{
  const ui=boot({training_runs:{'unfamiliar-run':'run-1'},checkpoints:{'unfamiliar-checkpoint':'candidate-1'}});await ui.upload.fire('submit');
  assert.equal(ui.review.querySelector('[data-run-association]').value,'run-1');
  const cp=ui.review.querySelector('[data-checkpoint-association]');assert.equal(cp.value,'candidate-1');assert.match(cp.children[1].textContent,/exact source match/);
});


test('omitted checkpoint progress matches an unknown-progress exact candidate',async()=>{
  const ui=boot({training_runs:{'unfamiliar-run':'run-1'},checkpoints:{'unfamiliar-checkpoint':'candidate-1'}},true);
  await ui.upload.fire('submit');
  const cp=ui.review.querySelector('[data-checkpoint-association]');
  assert.equal(cp.children.length,2);assert.equal(cp.value,'candidate-1');assert.equal(cp.children[1].disabled,false);
  await ui.review.fire('submit');const body=JSON.parse(ui.requests[1].options.body);
  assert.equal(body.checkpoint_associations['unfamiliar-checkpoint'],'candidate-1');
});


test('upload links and successful imports stay below prefix for either API URL convention',async()=>{
  for(const base of ['', '/sims'])for(const apiURL of ['/evaluations/e1',base+'/evaluations/e1']){
    const ui=boot({training_runs:{},checkpoints:{}},false,base,apiURL);await ui.upload.fire('submit');
    const help=ui.checkpointHost.children[0].children.find(child=>child.tag==='p'&&child.children?.some(node=>node.tag==='a'));
    const link=help.children.find(node=>node.tag==='a');assert.equal(link.href,base+'/training-runs/run-1');
    await ui.review.fire('submit');assert.deepEqual(ui.navigation,[base+'/evaluations/e1']);
  }
});


test('training run name updates compact checkpoint preview and submits only run annotations',async()=>{
  const ui=boot();ui.data.checkpoints[0].progress=9961472;
  await ui.upload.fire('submit');
  const name=ui.review.querySelector('[data-run-name]'),preview=ui.review.querySelector('[data-checkpoint-preview]');
  name.value='Jeff';await name.fire('input');
  assert.equal(preview.textContent,'Jeff · 9.96M steps');
  name.value='<img src=x onerror=alert(1)>';await name.fire('input');
  assert.equal(preview.textContent,'<img src=x onerror=alert(1)> · 9.96M steps');assert.equal(preview.children.length,0);
  name.value='Jeff';await name.fire('input');await ui.review.fire('submit');
  const body=JSON.parse(ui.requests[1].options.body);
  assert.equal(body.run_annotations['unfamiliar-run'].name,'Jeff');
  assert.equal(body.checkpoint_associations['unfamiliar-checkpoint'],'new');
  assert.equal(ui.data.training_runs[0].name,'Uploaded policy');assert.equal(ui.data.checkpoints[0].progress,9961472);
});

test('existing run association shows its preserved name and restores the new run draft',async()=>{
  const ui=boot();await ui.upload.fire('submit');
  const run=ui.review.querySelector('[data-run-association]'),name=ui.review.querySelector('[data-run-name]'),preview=ui.review.querySelector('[data-checkpoint-preview]');
  name.value='Jeff';await name.fire('input');run.value='run-1';await run.fire('change');
  assert.equal(name.disabled,true);assert.equal(name.value,'Baseline');assert.equal(preview.textContent,'Baseline · 100 steps');
  run.value='new';await run.fire('change');assert.equal(name.disabled,false);assert.equal(name.value,'Jeff');assert.equal(preview.textContent,'Jeff · 100 steps');
});

test('upload review hides raw source identities and hashes, including source conflicts',async()=>{
  const ui=boot();const source='12345678-1234-1234-1234-123456789abc';
  ui.data.training_runs[0].name=source;ui.data.training_runs[0].source_run_id=source;ui.data.checkpoints[0].source_run_id=source;
  ui.data.conflicts=[`Training run metadata conflicts: ${source}`, 'Checkpoint identity conflicts: unfamiliar-checkpoint'];
  await ui.upload.fire('submit');
  const visible=el=>[el.textContent,el.tag==='input'?el.value:'',...el.children.map(child=>typeof child==='string'?child:visible(child))].join(' ');
  const text=visible(ui.review);assert.equal(text.includes(source),false);assert.equal(text.includes('a'.repeat(64)),false);assert.equal(text.includes('unfamiliar-checkpoint'),false);
  assert.match(text,/Unknown training run/);assert.equal(ui.review.querySelector('[type=submit]').disabled,true);
});


test('checkpoint suggestions still require the exact weights hash',async()=>{
  const ui=boot();ui.data.existing_checkpoints.push({id:'wrong-hash',run_id:'run-1',run_name:'Baseline',progress:100,sha256:'b'.repeat(64)});
  ui.data.checkpoint_suggestions['unfamiliar-checkpoint'].push('wrong-hash');
  await ui.upload.fire('submit');
  const cp=ui.review.querySelector('[data-checkpoint-association]');assert.equal(cp.children.length,2);assert.equal(cp.children.some(option=>option.value==='wrong-hash'),false);
});

test('duplicate evaluation review retains its prefixed existing-record link',async()=>{
  const ui=boot(undefined,false,'/sims');ui.data.duplicate_id='already-published';await ui.upload.fire('submit');
  const notice=ui.checkpointHost.children[0];assert.match(notice.textContent,/already published/);assert.equal(notice.children[0].href,'/sims/evaluations/already-published');
});


test('long training run names remain full plain-text annotations in the preview and import',async()=>{
  const ui=boot();await ui.upload.fire('submit');
  const name=ui.review.querySelector('[data-run-name]'),preview=ui.review.querySelector('[data-checkpoint-preview]');
  const longName='J'.repeat(300);name.value=longName;await name.fire('input');
  assert.equal(preview.textContent,`${longName} \u00b7 100 steps`);assert.equal(preview.children.length,0);
  await ui.review.fire('submit');assert.equal(JSON.parse(ui.requests[1].options.body).run_annotations['unfamiliar-run'].name,longName);
});
