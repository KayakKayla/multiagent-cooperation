const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const{test}=require('node:test');
class Element{
  constructor(){this.children=[];this.dataset={};this.listeners={};this.value='';this.textContent='';}
  append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}
  addEventListener(event,fn){this.listeners[event]=fn;}
}

test('selected episode, shared artwork, and share URLs stay within root or configured prefix',async()=>{
  for(const base of ['', '/sims']){
    const host=new Element();host.dataset.evaluationId='e1';
    const select=new Element();select.options=[{value:'0'},{value:'1'}];
    const root=new Element(),status=new Element(),metrics=new Element();
    const ids={'episode-player':host,'episode-select':select,'replay-root':root,'episode-status':status,'episode-metrics':metrics};
    const images=['kitchen-atlas.png','sushi-atlas.png'].map(name=>{const image=new Element();image.dataset.replayAsset=name;return image;});
    const doc={querySelector:selector=>selector==='meta[name="base-path"]'?{content:base}:null,querySelectorAll:()=>[],getElementById:id=>ids[id],addEventListener(_event,fn){this.ready=fn;},createElement(tag){const el=new Element();if(tag==='template')el.content={querySelectorAll:()=>images};return el;}};
    const requested=[],history=[];let mounted;
    const recording={schema_version:1,steps:[{agents:[{id:'agent_0'},{id:'agent_1'},{id:'agent_2'}]}]};
    const location={href:'https://example.test'+base+'/evaluations/e1?episode=1&step=2',search:'?episode=1&step=2'};
    const context={document:doc,Headers,URL,URLSearchParams,AbortController,location,history:{replaceState(_state,_unused,url){history.push(String(url));}},fetch:async(url)=>{requested.push(url);return{ok:true,text:async()=>'<div>Trusted viewer</div>',json:async()=>({episode:{seed:10,steps:2,metrics:{team_reward:0}},recording})};},mountReplay(element,data,options){mounted={element,data,options};options.onStep(Number(options.step));return{destroy(){}};}};
    context.window=context;vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/site.js'),'utf8'),context);context.Site.toast=message=>{throw new Error(message);};
    vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/episode.js'),'utf8'),context);doc.ready();await new Promise(resolve=>setImmediate(resolve));
    assert.deepEqual(requested.sort(),[base+'/api/evaluations/e1/episodes/1',base+'/static/replay/viewer.html'].sort());
    assert.equal(mounted.data.steps[0].agents.length,3);assert.equal(mounted.options.step,'2');assert.equal(select.value,'1');
    assert.deepEqual(images.map(image=>image.src),[base+'/static/replay/kitchen-atlas.png',base+'/static/replay/sushi-atlas.png']);
    assert.equal(history.at(-1),'https://example.test'+base+'/evaluations/e1?episode=1&step=2');
    assert.equal(root.children.length,1);
  }
});
