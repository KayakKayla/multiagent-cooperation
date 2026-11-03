const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const{test}=require('node:test');

class Element{
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.value='';this.disabled=false;this.selected=false;this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;if(this.tag==='select')this.value='';}
  add(option){this.append(option);if(option.selected||this.children.length===1)this.value=option.value;}
  addEventListener(event,callback){(this.listeners[event]??=[]).push(callback);}
  async fire(event){for(const fn of this.listeners[event]??[])await fn({preventDefault(){},currentTarget:this});}
  focus(){}scrollIntoView(){}
  querySelector(selector){return this.querySelectorAll(selector)[0]??null;}
  querySelectorAll(selector){const matches=el=>selector==='[type=submit]'?el.type==='submit':selector==='.form-error'?el.className==='form-error':selector==='[data-source-run]'?'sourceRun'in el.dataset:selector==='[data-run-association]'?'runAssociation'in el.dataset:selector==='[data-checkpoint-association]'?'checkpointAssociation'in el.dataset:selector==='[data-run-name]'?'runName'in el.dataset:selector==='[data-run-notes]'?'runNotes'in el.dataset:false;return this.children.flatMap(el=>typeof el==='string'?[]:[...(matches(el)?[el]:[]),...el.querySelectorAll(selector)]);}
}
function boot(matches={training_runs:{},checkpoints:{}}, omittedProgress=false){
  const upload=new Element('form'),review=new Element('form'),runHost=new Element('div'),checkpointHost=new Element('div'),summary=new Element('p'),file=new Element('input');file.files=[{name:'evaluation.json',size:100}];
  for(const form of[upload,review]){const button=new Element('button');button.type='submit';const error=new Element('p');error.className='form-error';form.append(button,error);}
  review.append(runHost,checkpointHost);review.elements={title:new Element('input'),notes:new Element('textarea')};
  const ids={'upload-form':upload,'review-form':review,'run-associations':runHost,'checkpoint-associations':checkpointHost,'review-summary':summary,'evaluation-file':file};
  const doc={createElement:tag=>new Element(tag),createTextNode:text=>String(text),getElementById:id=>ids[id],addEventListener(_event,fn){this.ready=fn;}};
  const run={source_run_id:'unfamiliar-run',name:'Uploaded policy',progress_unit:'environment_steps',notes:''};
  const cp={source_checkpoint_id:'unfamiliar-checkpoint',source_run_id:run.source_run_id,progress:100,sha256:'a'.repeat(64)};
  const data={id:'preview',expires_at:2000000000,episode_count:2,recording_count:2,document:{environment:{layout:'demo'},participants:[{agent_id:'a0',source_checkpoint_id:cp.source_checkpoint_id}]},training_runs:[run],checkpoints:[cp],matches,existing_runs:[{id:'run-1',name:'Baseline'}],existing_checkpoints:[
    {id:'candidate-1',run_id:'run-1',run_name:'Baseline',label:'Baseline · 100 environment_steps',progress:100,sha256:cp.sha256},
    {id:'wrong-progress',run_id:'run-1',label:'Baseline · 200 environment_steps',progress:200,sha256:cp.sha256},
    {id:'incompatible',run_id:'other-run',label:'Unrelated model · 100 environment_steps',progress:100,sha256:cp.sha256}],checkpoint_suggestions:{[cp.source_checkpoint_id]:['candidate-1','wrong-progress']}};
  if(omittedProgress){delete cp.progress;data.existing_checkpoints[0].progress=null;}
  const requests=[];const context={document:doc,Option:class extends Element{constructor(label,value,_default=false,selected=false){super('option');this.textContent=label;this.value=value;this.selected=selected;}},Site:{request:async(url,options)=>{requests.push({url,options});return url==='/api/uploads'?data:{url:'/evaluations/e1'};},toast(){},showError(_form,error){throw error;}},sessionStorage:{setItem(){}},location:{assign(){}},Date,Map,Set};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/upload.js'),'utf8'),context);doc.ready();
  return{upload,review,runHost,checkpointHost,requests};
}

test('suggested exact checkpoint explains run and requires explicit unfamiliar-run association',async()=>{
  const ui=boot();await ui.upload.fire('submit');
  const run=ui.review.querySelector('[data-run-association]'),cp=ui.review.querySelector('[data-checkpoint-association]');
  assert.equal(run.value,'new');assert.equal(cp.value,'new');assert.equal(cp.children.length,2);
  assert.match(cp.children[1].textContent,/Baseline · 100 environment_steps.*suggested exact checkpoint/);assert.equal(cp.children[1].disabled,true);
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
