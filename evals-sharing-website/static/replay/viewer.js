
"use strict";
window.mountReplay = function(root, data, options = {}) {
const abort = new AbortController();
const listen = (target, event, fn) => target.addEventListener(event, fn, {signal: abort.signal});
const get=id=>root.querySelector('#'+id);
const frames=data.frames??[];
if(data.schema_version!==1||!data.steps.length||(frames.length>0&&data.steps.length!==frames.length))throw Error('Unsupported or incomplete replay data.');
const meta=data.metadata??{},presentation=data.presentation,last=data.steps.length-1;
const canvas=get('kitchen'),ctx=canvas.getContext('2d'),atlas=get('sprite-atlas'),extra=get('sushi-atlas');
const art=presentation?.supported===true&&presentation.scenes.length===data.steps.length;
const sushi=presentation?.kind==='sushi';
const spriteStyle=document.createElement('style');
// Direct image rules avoid browser size limits on CSS custom-property values.
spriteStyle.textContent=`.replay-viewer .portrait{background-image:url("${atlas.src}")}.replay-viewer .recipe-icon{background-image:url("${sushi?extra.src:atlas.src}");background-position:${sushi?'50% 66.6667%':'25% 100%'}}`;
if(!options.embedded)root.append(spriteStyle);
root.classList.toggle("sushi",sushi);
const sprites={red:{up:0,right:1,down:2,left:3},blue:{up:4,right:5,down:6,left:7}};
const objects={floor:8,counter:9,onion_supply:10,plates:11,hob:12,empty_pot:13,onion:14,plate:15,soup:16,serve:17,wall:18,soup_pot:19,cucumber:20,chopped_cucumber:21,raw_rice:22,nori:23,prawn:24,board:25,sink:26,drying:27,return:28,bin:29,dirty_plate:30,rice_pot:31,sushi:32,rice_supply:33,cucumber_supply:34,nori_supply:35,ingredient_supply:36,ingredient:37,recipe:38,switch:39};
const colors={red:'#b95e48',blue:'#497b98',green:'#5f8a58',purple:'#806794',yellow:'#a58c34',orange:'#be7c3e'};
const events=data.steps.flatMap((s,i)=>{if(!i)return[];const prev=data.steps[i-1].totals??{},correct=s.totals?.correct_dishes-prev.correct_dishes,wrong=s.totals?.wrong_dishes-prev.wrong_dishes;return correct>0||wrong>0?[{step:i,correct,wrong,reward:s.totals.team_reward-prev.team_reward}]:[];});
const final=data.steps[last].totals??{};
let current=0,playing=false,timer=null,original=!art&&frames.length>0;
const hashStep=options.step??new URLSearchParams(location.hash.slice(1)).get('step');
if(hashStep!==null&&Number.isFinite(Number(hashStep)))current=Math.max(0,Math.min(last,Math.trunc(Number(hashStep))));
const CELL=132,OFFSET=50,width=art?presentation.width:(data.steps[0].grid?.[0]?.length??1),height=art?presentation.height:(data.steps[0].grid?.length??1);
canvas.width=width*CELL+2*OFFSET;canvas.height=height*CELL+2*OFFSET;
get('board-wrap').style.aspectRatio=`${canvas.width} / ${canvas.height}`;
if(width>6)get('board-wrap').style.width='100%';
function sprite(index,x,y,w,h=w){if(typeof index==='string')index=objects[index];const image=index>=20?extra:atlas;if(!image.complete||!image.naturalWidth)return;index%=20;const sw=image.naturalWidth/5,sh=image.naturalHeight/4;ctx.drawImage(image,(index%5)*sw,Math.floor(index/5)*sh,sw,sh,x,y,w,h);}
function round(x,y,w,h,r,fill){ctx.fillStyle=fill;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();}
function label(text,x,y,fill='#f9f8ed',ink='#53634b',w=82){ctx.font='600 13px system-ui';round(x-w/2,y-12,w,24,7,fill);ctx.fillStyle=ink;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,x,y,w-6);}
function ingredientBadge(values,x,y){if(!values?.some(i=>i!==0))return;const text=[...new Set(values)].map(i=>`#${i}×${values.filter(v=>v===i).length}`).join(' ');label(text,x,y,'#f8f3e6','#65553b',Math.max(55,text.length*7));}
function drawItem(item,x,y,size){
 if(item.sprite==='ingredient_id'){label(`#${item.ingredient_id}`,x+size/2,y+size*.84,'#f7f0dd','#695a3e',46);return;}
 if(item.sprite==='recipe_id'){if(item.ingredients.length){const ids=[...new Set(item.ingredients)].map(i=>`${i===0?'O':'#'+i}×${item.ingredients.filter(v=>v===i).length}`).join(' ');label(ids,x+size/2,y+size*.87,'#fff6df','#5b6545',Math.min(CELL-8,Math.max(65,ids.length*7)));}if(item.active!==null)label(item.active?'ON':'OFF',x+size/2,y+size*.13,item.active?'#28614e':'#6f735f','#fff',47);return;}
 sprite(item.sprite,x,y,size);
 if(item.toppings?.length)for(const [n,topping]of item.toppings.entries())sprite(topping,x+size*(.17+n*.19),y+size*.33,size*.36);
 ingredientBadge(item.ingredients,x+size/2,y+size*.95);
 if(item.progress>0&&item.progress<1){round(x+size*.1,y+size*.98,size*.8,6,3,'#58614b');round(x+size*.1,y+size*.98,size*.8*item.progress,6,3,'#deb55e');}
}
function drawKitchen(state,scene){
 if(!art)return;
 ctx.clearRect(0,0,canvas.width,canvas.height);
 ctx.save();ctx.shadowColor='#333d3a22';ctx.shadowBlur=20;ctx.shadowOffsetY=12;round(OFFSET-9,OFFSET-9,width*CELL+18,height*CELL+22,20,'#736953');ctx.restore();
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const tile=scene.tiles[y][x],px=OFFSET+x*CELL,py=OFFSET+y*CELL;
  round(px,py,CELL,CELL,2,tile==='floor'?'#e6dfc9':tile==='wall'?'#596161':'#b58d60');
  sprite(tile==='floor'?'floor':tile==='wall'?'wall':'counter',px-10,py-10,CELL+20);
  if(tile==='floor'){ctx.strokeStyle='#d7d1bd';ctx.lineWidth=1;ctx.strokeRect(px+.5,py+.5,CELL-1,CELL-1);}
  else{ctx.fillStyle='#654c2422';ctx.fillRect(px,py+CELL-5,CELL,5);}
  if(!['floor','counter','wall','drying','return'].includes(tile))sprite(tile==='recipe'?'soup':tile,px+8,py+8,CELL-16);
  if(tile==='drying'||tile==='return')label(tile==='drying'?'CLEAN':'RETURN',px+CELL/2,py+CELL-10,tile==='drying'?'#537970':'#8c7250','#fff9e8',80);
  if(tile==='serve')label('SERVE',px+CELL/2,py+CELL-10,'#2b634f','#fff9e8',72);
  if(tile==='prawn')label('INACTIVE',px+CELL/2,py+CELL-10,'#727663','#fff9e8',87);
 }
 if(!sushi)for(const pot of scene.pots){const [x,y]=pot.position,px=OFFSET+x*CELL,py=OFFSET+y*CELL;sprite(pot.ingredients===3?'soup_pot':'empty_pot',px+13,py+10,CELL-26);if(pot.ingredients>0&&pot.ingredients<3)for(let n=0;n<pot.ingredients;n++)sprite((pot.ingredient_types[n]??0)===0?'onion':'ingredient',px+44+n*19,py+44,35);
  const raw=state.pots.find(p=>p.index===pot.index);if(raw?.contents)ingredientBadge(decodeIngredients(raw.contents),px+CELL/2,py+CELL*.56);
 }
 for(const item of scene.items){const [x,y]=item.position;drawItem(item,OFFSET+x*CELL+15,OFFSET+y*CELL+12,CELL-30);}
 if(get('grid-toggle').checked){ctx.strokeStyle='#31563d80';ctx.lineWidth=1;for(let n=0;n<=width;n++){ctx.beginPath();ctx.moveTo(OFFSET+n*CELL,OFFSET);ctx.lineTo(OFFSET+n*CELL,OFFSET+height*CELL);ctx.stroke();}for(let n=0;n<=height;n++){ctx.beginPath();ctx.moveTo(OFFSET,OFFSET+n*CELL);ctx.lineTo(OFFSET+width*CELL,OFFSET+n*CELL);ctx.stroke();}ctx.font='12px system-ui';ctx.fillStyle='#627354';ctx.textAlign='center';for(let n=0;n<width;n++)ctx.fillText(n,OFFSET+(n+.5)*CELL,OFFSET-22);for(let n=0;n<height;n++)ctx.fillText(n,OFFSET-24,OFFSET+(n+.5)*CELL);}
 for(const agent of state.agents){const [x,y]=agent.position,px=OFFSET+x*CELL,py=OFFSET+y*CELL;ctx.fillStyle='#36433222';ctx.beginPath();ctx.ellipse(px+CELL/2,py+CELL*.75,34,16,0,0,Math.PI*2);ctx.fill();if(sprites[agent.color])sprite(sprites[agent.color][agent.facing],px+12,py-6,CELL-24,CELL+4);else{round(px+25,py+16,CELL-50,CELL-35,22,colors[agent.color]??'#64748b');label(`Chef ${agent.index}`,px+CELL/2,py+CELL/2,'#ffffff','#334155',75);}const [dx,dy]={up:[0,-1],down:[0,1],right:[1,0],left:[-1,0]}[agent.facing],held=scene.held[String(agent.index)];if(held)drawItem(held,px+CELL/2+dx*29-25,py+CELL*.62+dy*21-25,53);label(String(agent.index),px+CELL-20,py+22,colors[agent.color],'#fff',25);}
 // Pot badges follow movable containers, including when carried by a chef.
 for(const pot of scene.pots){const [x,y]=pot.position,px=OFFSET+x*CELL,py=OFFSET+y*CELL;label(pot.badge,px+CELL/2,py+CELL-13,pot.ready?'#28614e':'#fff9e8',pot.ready?'#fff9e8':'#5e674d',103);if(pot.cooking&&pot.progress!==null){round(px+18,py+CELL-2,CELL-36,5,3,'#675c40');round(px+18,py+CELL-2,(CELL-36)*pot.progress,5,3,'#e5b952');}}
 canvas.setAttribute('aria-label',`Kitchen after step ${current}. ${state.agents.map(a=>`Chef ${a.index} at ${(a.position??[]).join(', ')}, facing ${a.facing}, holding ${a.holding}`).join('. ')}`);
}
function decodeIngredients(code){let value=Math.floor(code/4),index=0,result=[];while(value>0){for(let n=0;n<value%4;n++)result.push(index);index++;value=Math.floor(value/4);}return result;}
function element(tag,className,text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
function render(){
 const s=data.steps[current],scene=art?presentation.scenes[current]:null;drawKitchen(s,scene);
 if(frames.length)get('original-frame').src=frames[current];get('original-frame').alt=`Original recorded kitchen after step ${current}`;canvas.hidden=original||!art;get('original-frame').hidden=!original;get('grid-toggle').disabled=original;
 get('score').replaceChildren(document.createTextNode(s.totals?.team_reward??"Unknown"),element('small','', 'points'));get('correct').replaceChildren(document.createTextNode(s.totals?.correct_dishes??"Unknown"),element('small','',final.correct_dishes>0?`/ ${final.correct_dishes} in replay`:'in this episode'));get('wrong').textContent=s.totals?.wrong_dishes??'Unknown';
 get('step-label').replaceChildren(document.createTextNode(`Step ${current} `),element('span','',`/ ${last}`));get('state-label').textContent=current===0?'Initial kitchen':`After step ${current}`;get('recipe-name').textContent=scene?.recipe??s.recipe??'Recorded recipe';get('milestone').hidden=!s.milestone;get('milestone').textContent=s.milestone??'';
 get('timeline').value=current;get('timeline').setAttribute('aria-valuetext',`Step ${current} of ${last}`);get('step-input').value=current;get('previous').disabled=current===0;get('next').disabled=current===last;get('play').disabled=last===0;get('play').textContent=playing?'Ⅱ Pause':current===last&&last>0?'↻ Replay':'▶ Play';get('play').setAttribute('aria-pressed',String(playing));get('playback-status').textContent=playing?'Playing':current===last&&last>0?'Finished':'Paused';
 get('chefs').replaceChildren();get('chef-count').textContent=`${s.agents.length} ${s.agents.length===1?"CHEF":"CHEFS"}`;
 for(const a of s.agents){const chef=element('article',`chef ${a.color}`),top=element('div','chef-top'),portrait=element('div',art&&sprites[a.color]?'portrait':'portrait no-art');portrait.setAttribute('aria-hidden','true');const name=element('div');const participant=(data.participants??[]).find(p=>p.agent_id===a.id)??(data.participants??[])[a.index];name.append(element('h3','',`Chef ${a.index}`),element('p','agent-id',`${a.id} · ${participant?.label??a.color}`));const dot=element('span','color-dot');dot.style.backgroundColor=colors[a.color]??'#7c8b72';top.append(portrait,name,dot);chef.append(top);
  const holding=scene?.held[String(a.index)]?.description??a.holding;
  for(const [key,value] of [['Holding',['nothing','empty hands'].includes(holding)?'Empty hands':holding],['Last action',a.action??'Initial state'],['Position / facing',`(${(a.position??[]).join(', ')}) · ${a.facing}`]]){const row=element('div','chef-row');row.append(element('span','',key),element('b',key==='Last action'?'action-pill':'',value));chef.append(row);}get('chefs').append(chef);
 }
 get('pots').replaceChildren();get('pot-count').textContent=`${(s.pots??[]).length} ${(s.pots??[]).length===1?"POT":"POTS"}`;
 const pots=scene?.pots??(s.pots??[]).map(p=>({...p,ready:p.status==='ready to plate',cooking:p.timer>0,progress:null,detail:`${p.ingredients} / ${p.capacity??3} ingredients`}));
 for(const p of pots){const pot=element('div','pot'),top=element('div','pot-top');top.append(element('b','',`Pot ${p.index}`),element('span',`pot-status${p.ready?' ready':''}`,p.status));const bar=element('div',`pot-progress${p.ready?' ready':''}`),fill=element('span');bar.hidden=p.progress===null;fill.style.width=`${(p.progress??0)*100}%`;bar.append(fill);const detail=element('div','pot-detail');detail.append(element('span','',p.detail),element('span','',`(${(p.position??[]).join(', ')})`));pot.append(top,bar,detail);get('pots').append(pot);}if(!pots.length)get('pots').append(element('p','delivery-help','No pots in this layout.'));
 get('delivery-count').textContent=`${s.totals?.correct_dishes??"Unknown"} correct · ${s.totals?.wrong_dishes??"Unknown"} wrong`;
 for(const button of get('deliveries').children){const step=Number(button.dataset.step);button.classList.toggle('delivered',step<=current);button.setAttribute('aria-current',String(step===current));}
 const event=events.find(e=>e.step===current);get('toast').hidden=!event;get('toast').textContent=event?`${event.correct} correct · ${event.wrong} wrong · ${event.reward>=0?'+':''}${event.reward} reward`:'';
 if(get('exact-state').parentElement.open)get('exact-state').textContent=JSON.stringify(s,null,2);
 if(options.onStep)options.onStep(current);
}
function pause(){playing=false;clearTimeout(timer);timer=null;}
function seek(value){if(value===''||!Number.isFinite(Number(value))){render();return;}pause();current=Math.max(0,Math.min(last,Math.trunc(Number(value))));render();}
function schedule(){clearTimeout(timer);timer=setTimeout(()=>{current=Math.min(last,current+1);if(current===last)pause();render();if(playing)schedule();},(meta.frame_ms??100)/Number(get('speed').value));}
function toggle(){if(playing)pause();else if(last>0){if(current===last)current=0;playing=true;schedule();}render();}
for(const [i,event] of events.entries()){const tick=element('span',`delivery-tick${event.wrong?' wrong':''}`);tick.style.left=`${event.step/Math.max(1,last)*100}%`;get('delivery-ticks').append(tick);const button=element('button',`delivery-button${event.wrong?' wrong':''}`,String(i+1).padStart(2,'0'));button.dataset.step=event.step;button.title=`Step ${event.step}: ${event.correct} correct, ${event.wrong} wrong`;button.setAttribute('aria-label',`Jump to delivery at step ${event.step}: ${event.correct} correct, ${event.wrong} wrong`);button.addEventListener('click',()=>seek(event.step));get('deliveries').append(button);}
if(!events.length)get('delivery-help').textContent='No deliveries recorded in this episode. You can still inspect every action and kitchen state.';
get('play').addEventListener('click',toggle);get('previous').addEventListener('click',()=>seek(current-1));get('next').addEventListener('click',()=>seek(current+1));get('timeline').addEventListener('input',e=>seek(e.target.value));get('step-input').addEventListener('focus',()=>{pause();render();});get('step-input').addEventListener('input',e=>{if(e.target.value!=='')seek(e.target.value);});get('step-input').addEventListener('change',e=>seek(e.target.value));get('speed').addEventListener('change',()=>{if(playing)schedule();});get('grid-toggle').addEventListener('change',render);
function setOriginal(value){original=value;get('illustrated').setAttribute('aria-pressed',String(!value));get('original').setAttribute('aria-pressed',String(value));render();}
get('illustrated').addEventListener('click',()=>setOriginal(false));get('original').addEventListener('click',()=>setOriginal(true));get('illustrated').disabled=!art;get('original').disabled=!frames.length;get('original').title=frames.length?'':'Original PNG frames are available only in local offline replays.';
get('ingredient-note').hidden=!(art&&presentation.ingredient_ids?.length);get('fallback').hidden=art;get('fallback').textContent=art?'':`${frames.length?'Original view':'Unsupported illustrated view'}: ${presentation?.reason??'illustrated data unavailable'}. ${frames.length?'Exact recorded frames remain available.':'The exact recorded state and controls remain available below.'}`;
get('exact-state').parentElement.addEventListener('toggle',render);
listen(document,'keydown',e=>{if(options.keyboardScoped&&!root.contains(e.target))return;if(e.target.closest('input,select,textarea')||e.target.isContentEditable)return;if(e.code==='Space'&&e.target.closest('button,summary,a'))return;if(!['Space','ArrowLeft','ArrowRight'].includes(e.code))return;e.preventDefault();if(e.code==='Space'){if(!e.repeat)toggle();}else seek(current+(e.code==='ArrowLeft'?-1:1));});
const names={coord_ring:'Coordination Ring',oc2_sushi_1_3:'Sushi Kitchen · Story 1–3'};
get('title').textContent=names[meta.layout]??(meta.layout??'Unknown layout').replaceAll('_',' ').replace(/\b[a-z]/g,c=>c.toUpperCase());if(!options.embedded)document.title=`${meta.layout} · Kitchen Lab`;
const scripted=meta.mode==='scripted';get('reference').textContent=scripted?'Scripted reference':`${meta.mode} policy`;
get('eyebrow').textContent=scripted?'Replay collection / Scripted reference':'Replay collection / Frozen policy';get('subtitle').textContent=`Episode ${(meta.episode??0)+1} · ${scripted?(meta.starts??'Recorded starts'):'Test seed '+meta.seed} · ${last} ${last===1?"step":"steps"}`;
get('layout-caption').textContent=`${width} × ${height} layout · ${meta.starts??'Recorded starts'}`;
get('provenance-title').textContent=scripted?'Recorded joint actions; no learned policy.':`Frozen ${meta.mode} policy; no learning during playback.`;
get('perspective').textContent=meta.perspective??(scripted?'Saved action schedule. Artwork changes only the spectator view.':`Full kitchen shown for viewers. Chefs receive ${meta.agent_view_size==null?'full-kitchen':`local ${2*meta.agent_view_size+1} × ${2*meta.agent_view_size+1}`} observations.`);
get('timeline').max=last;get('step-input').max=last;
const onArtLoad=()=>render();atlas.onload=onArtLoad;extra.onload=onArtLoad;
function artError(){get('illustrated').disabled=true;get('fallback').hidden=false;get('fallback').textContent='Artwork unavailable. Inspect the exact recorded state below.';canvas.hidden=true;if(frames.length)setOriginal(true);}
atlas.onerror=artError;extra.onerror=artError;
setOriginal(original);
return {destroy(){pause();abort.abort();atlas.onload=null;extra.onload=null;atlas.onerror=null;extra.onerror=null;},seek};
};
