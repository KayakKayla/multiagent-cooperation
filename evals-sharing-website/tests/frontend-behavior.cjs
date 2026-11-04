/* Execute real site listeners with DOM event dispatch/currentTarget and form reset semantics. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

class Control {
  constructor(name='',value='') { this.name=name;this.value=value;this.defaultValue=value;this.listeners={};this.disabled=false;this.hidden=false;this.dataset={};this.textContent=''; }
  addEventListener(type,callback) { this.listeners[type]=callback; }
  async fire(type) { let target=this; const event={preventDefault(){},get currentTarget(){return target;}}; const pending=this.listeners[type](event);target=null;await pending; }
  focus() {}
}
function boot(basePath='') {
  const save=new Control(),cancel=new Control(),toggle=new Control(),confirmation=new Control(),deletion=new Control();
  deletion.dataset={deleteUrl:'/api/evaluations/e1',deleteName:'Example'};
  const title=new Control('title','Initial title'),notes=new Control('notes','Initial notes'),version=new Control('version','1');
  const fields=[title,notes,version]; fields.version=version;
  const error=new Control(),heading=new Control(),noteText=new Control(),dialogTitle=new Control(),checkpoint=new Control();checkpoint.dataset={progress:'9961472',progressUnit:'environment_steps'};
  const edit=new Control();edit.elements=fields;edit.dataset.editUrl='/api/evaluations/e1';
  edit.querySelector=selector=>({'[data-edit-cancel]':cancel,'[type=submit]':save,'.form-error':error,'input:not([type=hidden])':title}[selector]);
  edit.reset=()=>fields.forEach(field=>field.value=field.defaultValue);
  const dialog=new Control();dialog.querySelector=selector=>selector==='#delete-title'?dialogTitle:error;dialog.showModal=()=>{};
  const elements={'[data-edit-url]':edit,'[data-edit-toggle]':toggle,'[data-delete-url]':deletion,'[data-record-heading]':heading,'[data-record-notes]':noteText};
  const doc={querySelectorAll:selector=>selector==='[data-checkpoint-label]'?[checkpoint]:[],querySelector:selector=>selector==='meta[name="base-path"]'?{content:basePath}:elements[selector]??null,getElementById:id=>id==='delete-dialog'?dialog:id==='confirm-delete'?confirmation:null,addEventListener(type,callback){if(type==='DOMContentLoaded')this.ready=callback;}};
  const fetched=[],navigation=[];const context={Headers,fetch:async(url,options)=>{fetched.push({url,options});return{ok:true,json:async()=>({ok:true})};},document:doc,sessionStorage:{getItem:()=>null,setItem(){}},location:{assign(value){navigation.push(value);}},FormData:class{constructor(form){this.form=form;}*[Symbol.iterator](){for(const field of this.form.elements)yield[field.name,field.value];}},setTimeout};context.window=context;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/site.js'),'utf8'),context);
  context.Site.toast=()=>{};const failures=[];context.Site.showError=(_form,failure)=>failures.push(failure.message);doc.ready();
  return{...context,save,cancel,toggle,confirmation,deletion,title,notes,version,edit,noteText,checkpoint,dialogTitle,failures,fetched,navigation};
}

test('failed permanent deletion releases the confirm button and can be retried',async()=>{
  const ui=boot();let calls=0;ui.Site.request=async()=>{calls++;await Promise.resolve();throw new Error('Referenced checkpoint');};
  await ui.deletion.fire('click');await ui.confirmation.fire('click');
  assert.equal(ui.confirmation.disabled,false);assert.deepEqual(ui.failures,['Referenced checkpoint']);
  await ui.confirmation.fire('click');assert.equal(calls,2);assert.equal(ui.confirmation.disabled,false);
});

test('save then cancel preserves saved annotations and current optimistic-edit version',async()=>{
  const ui=boot();const requests=[];ui.Site.request=async(_url,options)=>{const value=JSON.parse(options.body);requests.push(value);return{...value,version:value.version+1};};
  ui.title.value='Saved title';ui.notes.value='Saved notes';await ui.edit.fire('submit');
  assert.equal(ui.version.value,2);
  await ui.toggle.fire('click');ui.title.value='Unsaved draft';ui.notes.value='Unsaved draft';await ui.cancel.fire('click');await ui.toggle.fire('click');
  assert.equal(ui.title.value,'Saved title');assert.equal(ui.notes.value,'Saved notes');assert.equal(ui.version.value,2);
  await ui.edit.fire('submit');assert.equal(requests[1].version,2);assert.equal(requests[1].title,'Saved title');assert.deepEqual(ui.failures,[]);
});


test('application URLs, requests, and redirects support root and prefix without double prefixing',async()=>{
  for(const base of ['', '/sims']){
    const ui=boot(base);
    assert.equal(ui.Site.path('/'),base+'/');
    assert.equal(ui.Site.path('/api/evaluations/e1'),base+'/api/evaluations/e1');
    assert.equal(ui.Site.path(base+'/api/evaluations/e1'),base+'/api/evaluations/e1');
    assert.equal(ui.Site.recordURL(base+'/evaluations/e1'),base+'/evaluations/e1');
    for(const hostile of ['https://evil.example/evaluations/e1','//evil.example/evaluations/e1','/../evaluations/e1','/evaluations/e1?next=evil','/sims-other/evaluations/e1'])assert.throws(()=>ui.Site.recordURL(hostile));
    await ui.Site.request('/api/evaluations/e1',{method:'DELETE'});await ui.Site.request(base+'/api/evaluations/e1');
    assert.deepEqual(ui.fetched.map(item=>item.url),[base+'/api/evaluations/e1',base+'/api/evaluations/e1']);
    ui.Site.navigate('/');assert.deepEqual(ui.navigation,[base+'/']);
  }
});


test('shared display formatting handles progress, local dates, invalid values and epoch zero',()=>{
  const {Site}=boot();
  for (const [value,unit,expected] of [[9961472,'environment_steps','9.96M steps'],[0,'optimizer_updates','0 updates'],[null,'steps','unknown progress'],[1500,null,'1.5K units'],[1000000000,'steps','1B steps']]) assert.equal(Site.formatProgress(value,unit),expected);
  assert.equal(Site.formatDate(null),'Unknown');assert.equal(Site.formatDate('bad'),'Unknown');assert.equal(Site.formatDate('2026-02-30T12:00:00Z'),'Unknown');
  assert.equal(Site.formatDate('2026-11-03T12:30:45.123Z'),Site.formatDate('2026-11-03T07:30:45.123-05:00'));
  assert.equal(Site.formatDate(0),Site.formatDate('1970-01-01T00:00:00Z'));
  assert.equal(Site.formatDate(1793709045),Site.formatDate('2026-11-03T12:30:45Z'));
  assert.ok(!Site.formatDate('2026-11-03T12:30:45.123Z').includes('.123'));
  assert.equal(Site.formatNumber(1.23456),'1.23');assert.equal(Site.formatNumber(true),'Yes');assert.equal(Site.formatNumber('failed'),'failed');
});
test('clearing notes hides the section and adding notes reveals it',async()=>{
  const ui=boot();ui.Site.request=async()=>({version:2});
  ui.notes.value='   ';await ui.edit.fire('submit');assert.equal(ui.noteText.hidden,true);
  ui.notes.value='New note';await ui.edit.fire('submit');assert.equal(ui.noteText.hidden,false);assert.equal(ui.noteText.textContent,'New note');
});


test('renaming a run immediately updates automatic checkpoint labels and deletion confirmation',async()=>{
  const ui=boot();ui.Site.request=async()=>({version:2});
  ui.title.name='name';ui.title.value='Jeff';await ui.edit.fire('submit');
  assert.equal(ui.checkpoint.textContent,'Jeff · 9.96M steps');assert.equal(ui.checkpoint.title,'Jeff · 9.96M steps');
  assert.equal(ui.deletion.dataset.deleteName,'Jeff');await ui.deletion.fire('click');
  assert.equal(ui.dialogTitle.textContent,'Permanently delete “Jeff”?');
});
