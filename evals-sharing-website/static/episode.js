"use strict";
document.addEventListener('DOMContentLoaded', () => {
  const host=document.getElementById('episode-player'); if(!host)return;
  const select=document.getElementById('episode-select'),root=document.getElementById('replay-root'),status=document.getElementById('episode-status'),metrics=document.getElementById('episode-metrics');
  let viewer=null,serial=0,controller=null,fragment=null;
  function updateURL(index,step){const url=new URL(location.href);url.searchParams.set('episode',index);url.searchParams.set('step',step);history.replaceState(null,'',url);}
  async function load(index,step=0){
    const ticket=++serial; controller?.abort();controller=new AbortController();viewer?.destroy();viewer=null;root.replaceChildren();metrics.replaceChildren();select.value=String(index);status.textContent='Loading selected episode…';
    updateURL(index,step);
    try {
      const [response,html]=await Promise.all([Site.request(`/api/evaluations/${encodeURIComponent(host.dataset.evaluationId)}/episodes/${encodeURIComponent(index)}`,{signal:controller.signal}),fragment?Promise.resolve(fragment):fetch('/static/replay/viewer.html',{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error('Replay controls could not be loaded.');return r.text();})]);
      if(ticket!==serial)return;fragment=html;
      for(const [key,value]of Object.entries(response.episode.metrics??{})){const wrap=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key.replaceAll('_',' ');dd.textContent=String(value);wrap.append(dt,dd);metrics.append(wrap);}
      if(!response.recording){status.textContent='Recording unavailable for this episode. Its recorded statistics remain available; no trajectory was reconstructed.';return;}
      status.textContent=`Episode ${Number(index)+1} · seed ${response.episode.seed??'unknown'} · ${response.episode.steps} recorded actions.`;
      // This fragment is a fixed, same-origin application asset, never uploaded HTML.
      root.innerHTML=fragment;
      viewer=window.mountReplay(root,response.recording,{embedded:true,keyboardScoped:true,step,onStep:current=>updateURL(index,current)});
    }catch(error){if(error.name==='AbortError'||ticket!==serial)return;root.replaceChildren();status.textContent=`Recording could not be loaded: ${error.message}`;Site.toast(error.message,true);}
  }
  select.addEventListener('change',()=>load(select.value));
  document.querySelectorAll('[data-episode-link]').forEach(link=>link.addEventListener('click',event=>{event.preventDefault();load(link.dataset.episodeLink);host.scrollIntoView({block:'start',behavior:'auto'});}));
  const query=new URLSearchParams(location.search);const wanted=query.get('episode');const valid=Array.from(select.options).some(o=>o.value===wanted);if(select.options.length)load(valid?wanted:select.options[0].value,query.get('step')??0);else status.textContent='No episodes available.';
});
