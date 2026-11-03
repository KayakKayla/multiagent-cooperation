"use strict";
document.addEventListener('DOMContentLoaded', () => {
  const host=document.getElementById('episode-player'); if(!host)return;
  const root=document.getElementById('replay-root'),status=document.getElementById('episode-status'),metrics=document.getElementById('episode-metrics'),retry=document.getElementById('episode-retry');
  const episodes=JSON.parse(document.getElementById('episode-data').textContent),cards=Array.from(host.querySelectorAll('[data-episode-index]'));
  let viewer=null,serial=0,controller=null,fragment=null,current=null,currentStep=0;
  const nonnegative=value=>typeof value==='string'&&/^\d+$/.test(value)&&Number.isSafeInteger(Number(value))?Number(value):0;
  function updateURL(index,step){const url=new URL(location.href);url.searchParams.set('episode',index);url.searchParams.set('step',step);history.replaceState(null,'',url);}
  function showMetrics(episode){
    metrics.replaceChildren();const values={...episode.metrics,steps:episode.steps};
    for(const key of ['team_reward','correct_dishes','wrong_dishes'])if(values[key]===undefined)values[key]=null;
    const labels={team_reward:'Team reward',correct_dishes:'Correct deliveries',wrong_dishes:'Wrong deliveries',steps:'Steps'};
    for(const [key,value]of Object.entries(values)){const wrap=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=labels[key]??key.replaceAll('_',' ');dd.textContent=value===null||value===undefined?'Unknown':typeof value==='number'?Site.formatNumber(value):typeof value==='boolean'?(value?'Yes':'No'):typeof value==='object'?JSON.stringify(value):String(value);dd.title=value==null?'Unknown':typeof value==='object'?JSON.stringify(value):String(value);wrap.append(dt,dd);metrics.append(wrap);}
  }
  async function load(episode,step=0){
    const index=episode.index,ticket=++serial;current=episode;currentStep=step;controller?.abort();controller=new AbortController();viewer?.destroy();viewer=null;root.replaceChildren();showMetrics(episode);retry.hidden=true;
    cards.forEach(card=>card.setAttribute('aria-pressed',String(card.dataset.episodeIndex===String(index))));updateURL(index,step);
    if(!episode.recording_available){host.setAttribute('aria-busy','false');status.textContent='Recording unavailable for this episode. Its recorded statistics are shown.';return;}
    host.setAttribute('aria-busy','true');status.textContent='Loading selected episode…';
    try {
      const [response,html]=await Promise.all([Site.request(`/api/evaluations/${encodeURIComponent(host.dataset.evaluationId)}/episodes/${encodeURIComponent(index)}`,{signal:controller.signal}),fragment?Promise.resolve(fragment):fetch(Site.path('/static/replay/viewer.html'),{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error('Replay controls could not be loaded.');return r.text();})]);
      if(ticket!==serial)return;fragment=html;showMetrics(response.episode);
      if(!response.recording){status.textContent='Recording unavailable for this episode. Its recorded statistics are shown.';return;}
      status.textContent=`Episode ${Number(index)+1} · seed ${response.episode.seed??'Unknown'}`;
      // Fixed same-origin application markup, never uploaded HTML.
      const template=document.createElement('template');template.innerHTML=fragment;
      for(const image of template.content.querySelectorAll('[data-replay-asset]'))image.src=Site.path('/static/replay/'+image.dataset.replayAsset);
      root.replaceChildren(template.content);
      viewer=window.mountReplay(root,response.recording,{embedded:true,keyboardScoped:true,step,onStep:position=>{if(ticket===serial)updateURL(index,position);}});
    }catch(error){if(error.name==='AbortError'||ticket!==serial)return;root.replaceChildren();status.textContent=`Recording could not be loaded: ${error.message}`;retry.hidden=false;}
    finally{if(ticket===serial)host.setAttribute('aria-busy','false');}
  }
  cards.forEach(card=>card.addEventListener('click',()=>{if(!card.disabled){const episode=episodes.find(ep=>String(ep.index)===card.dataset.episodeIndex);if(episode)load(episode);}}));
  retry.addEventListener('click',()=>{if(current)load(current,currentStep);});
  const query=new URLSearchParams(location.search),wanted=query.get('episode');const selected=/^\d+$/.test(wanted??'')?episodes.find(ep=>String(ep.index)===wanted):null;
  const initial=selected??episodes.find(ep=>ep.recording_available)??episodes[0];
  if(initial)load(initial,nonnegative(query.get('step')));else status.textContent='No episodes available.';
});
