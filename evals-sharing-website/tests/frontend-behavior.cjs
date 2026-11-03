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
function boot() {
  const save=new Control(),cancel=new Control(),toggle=new Control(),confirmation=new Control(),deletion=new Control();
  deletion.dataset={deleteUrl:'/api/evaluations/e1',deleteName:'Example'};
  const title=new Control('title','Initial title'),notes=new Control('notes','Initial notes'),version=new Control('version','1');
  const fields=[title,notes,version]; fields.version=version;
  const error=new Control(),heading=new Control(),noteText=new Control(),dialogTitle=new Control();
  const edit=new Control();edit.elements=fields;edit.dataset.editUrl='/api/evaluations/e1';
  edit.querySelector=selector=>({'[data-edit-cancel]':cancel,'[type=submit]':save,'.form-error':error,'input:not([type=hidden])':title}[selector]);
  edit.reset=()=>fields.forEach(field=>field.value=field.defaultValue);
  const dialog=new Control();dialog.querySelector=selector=>selector==='#delete-title'?dialogTitle:error;dialog.showModal=()=>{};
  const elements={'[data-edit-url]':edit,'[data-edit-toggle]':toggle,'[data-delete-url]':deletion,'[data-record-heading]':heading,'[data-record-notes]':noteText};
  const doc={querySelector:selector=>elements[selector]??null,getElementById:id=>id==='delete-dialog'?dialog:id==='confirm-delete'?confirmation:null,addEventListener(type,callback){if(type==='DOMContentLoaded')this.ready=callback;}};
  const context={document:doc,sessionStorage:{getItem:()=>null,setItem(){}},location:{assign(){}},FormData:class{constructor(form){this.form=form;}*[Symbol.iterator](){for(const field of this.form.elements)yield[field.name,field.value];}},setTimeout};context.window=context;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/site.js'),'utf8'),context);
  context.Site.toast=()=>{};const failures=[];context.Site.showError=(_form,failure)=>failures.push(failure.message);doc.ready();
  return{...context,save,cancel,toggle,confirmation,deletion,title,notes,version,edit,failures};
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
