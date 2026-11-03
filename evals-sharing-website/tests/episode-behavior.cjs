const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{test}=require('node:test');
class Element{
 constructor(){this.children=[];this.dataset={};this.listeners={};this.textContent='';this.attrs={};this.disabled=false;}
 append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}
 addEventListener(event,fn){this.listeners[event]=fn;}setAttribute(key,value){this.attrs[key]=value;}
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function setup({base='',query='?episode=1&step=2',episodes,request}={}){
 episodes??=[{index:0,seed:1,steps:2,metrics:{correct_dishes:0},recording_available:true},{index:1,seed:2,steps:3,metrics:{},recording_available:true},{index:2,seed:3,steps:null,metrics:{},recording_available:false}];
 const host=new Element();host.dataset.evaluationId='e1';const cards=episodes.map(ep=>{const el=new Element();el.dataset.episodeIndex=String(ep.index);el.disabled=!ep.recording_available;return el;});host.querySelectorAll=()=>cards;
 const ids={'episode-player':host,'replay-root':new Element(),'episode-status':new Element(),'episode-metrics':new Element(),'episode-retry':new Element(),'episode-data':{textContent:JSON.stringify(episodes)}};
 const images=['kitchen-atlas.png','sushi-atlas.png'].map(name=>{const el=new Element();el.dataset.replayAsset=name;return el;});
 const doc={getElementById:id=>ids[id],addEventListener(_event,fn){this.ready=fn;},createElement(tag){const el=new Element();if(tag==='template')el.content={querySelectorAll:()=>images};return el;}};
 const requested=[],history=[],mounted=[],recording={steps:[{}]};
 const context={document:doc,URL,URLSearchParams,AbortController,location:{href:'https://example.test'+base+'/evaluations/e1'+query,search:query},history:{replaceState(_a,_b,url){history.push(String(url));}},Site:{path:p=>base+p,formatNumber:String,request:async(url,options)=>{requested.push(base+url);return request?request(url,options):{episode:episodes.find(ep=>url.endsWith('/'+ep.index)),recording};}},fetch:async url=>{requested.push(url);return{ok:true,text:async()=>'<div>Trusted controls</div>'};},mountReplay(root,data,options){mounted.push({data,options});options.onStep(options.step);return{destroy(){}};}};
 context.window=context;vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/episode.js'),'utf8'),context);doc.ready();return{ids,cards,requested,history,mounted,images};
}
test('card selection, metrics and share URLs honor root and prefix',async()=>{
 for(const base of ['', '/sims']){const app=setup({base});await flush();assert.deepEqual(app.requested.sort(),[base+'/api/evaluations/e1/episodes/1',base+'/static/replay/viewer.html'].sort());assert.equal(app.mounted[0].options.step,2);assert.equal(app.cards[1].attrs['aria-pressed'],'true');assert.equal(app.history.at(-1),'https://example.test'+base+'/evaluations/e1?episode=1&step=2');assert.deepEqual(app.images.map(i=>i.src),[base+'/static/replay/kitchen-atlas.png',base+'/static/replay/sushi-atlas.png']);app.cards[0].listeners.click();await flush();assert.equal(app.cards[0].attrs['aria-pressed'],'true');assert.ok(app.ids['episode-metrics'].children.some(el=>el.children[1].textContent==='0'));const count=app.requested.length;app.cards[2].listeners.click();assert.equal(app.requested.length,count);}
});
test('unavailable deep links and all unavailable evaluations do not request recordings',async()=>{
 const app=setup({query:'?episode=2&step=-8'});await flush();assert.equal(app.requested.length,0);assert.match(app.ids['episode-status'].textContent,/unavailable/);assert.equal(app.cards[2].attrs['aria-pressed'],'true');assert.match(app.history.at(-1),/step=0$/);
 const none=setup({episodes:[{index:0,metrics:{},recording_available:false}],query:'?episode=bogus&step=Infinity'});await flush();assert.equal(none.requested.length,0);assert.match(none.ids['episode-status'].textContent,/unavailable/);
});
test('stale responses cannot replace newly selected episode',async()=>{
 const pending=[];const app=setup({query:'?episode=0',request:()=>new Promise(resolve=>pending.push(resolve))});app.cards[1].listeners.click();pending[1]({episode:{index:1,seed:2,steps:3,metrics:{}},recording:{name:'second'}});await flush();pending[0]({episode:{index:0,seed:1,steps:2,metrics:{}},recording:{name:'first'}});await flush();assert.equal(app.mounted.length,1);assert.equal(app.mounted[0].data.name,'second');assert.equal(app.ids['episode-player'].attrs['aria-busy'],'false');
});
test('failed recording can be retried',async()=>{
 let attempts=0;const app=setup({request:async()=>{if(++attempts===1)throw Error('offline');return{episode:{steps:1,metrics:{}},recording:{}};}});await flush();assert.equal(app.ids['episode-retry'].hidden,false);app.ids['episode-retry'].listeners.click();await flush();assert.equal(app.mounted.length,1);assert.equal(app.ids['episode-retry'].hidden,true);
});

test('selected metrics preserve booleans, text, structured evidence and precise raw values',async()=>{
 const raw={team_reward:1234.567891234,finite_predictions:true,failed:false,status:'complete',details:{sample:2}};
 const app=setup({episodes:[{index:0,seed:1000,steps:2,metrics:raw,recording_available:true}]});await flush();
 const values=Object.fromEntries(app.ids['episode-metrics'].children.map(el=>[el.children[0].textContent,el.children[1]]));
 assert.equal(values['finite predictions'].textContent,'Yes');assert.equal(values.failed.textContent,'No');assert.equal(values.status.textContent,'complete');assert.equal(values.details.textContent,'{"sample":2}');assert.equal(values['Team reward'].title,'1234.567891234');assert.equal(raw.team_reward,1234.567891234);
});
