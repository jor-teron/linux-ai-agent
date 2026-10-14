const log=document.getElementById('log'),provider=document.getElementById('provider');
const model=document.getElementById('model'),modeSel=document.getElementById('mode');
const taskSel=document.getElementById('task');
const status=document.getElementById('status'),led=document.getElementById('led');
const input=document.getElementById('input'),send=document.getElementById('send');
const note=document.getElementById('note'),confirmBar=document.getElementById('confirm'),pendingCmd=document.getElementById('pendingCmd');
const fileInput=document.getElementById('file');
const sudoPassWrap=document.getElementById('sudoPassWrap'),sudoPass=document.getElementById('sudoPass');
const themeBtn=document.getElementById('themeBtn');
const toolsToggle=document.getElementById('toolsToggle');
const thinkToggle=document.getElementById('thinkToggle');
/* catalog from /api/models: tasks, modes, providers[{label,mode,needs_key,models[{id,label}]}], defaults */
let catalog={tasks:[],modes:[],providers:{},default_provider:'gemini',default_model:'gemini-3.5-flash-lite',default_mode:'online',default_task:'chat',status_blink_ms:2000};
let pendingVisionImage=null; // {name,content,encoding,mime} set by file picker for Vision
let history=[], keys={}, providersReady={};
let pendingIsSudo=false;
let selectedReady=null; // last /api/health selected readiness

/* Optional UI_LIGHT_* overrides from .env (served via /api/models|/api/health). */
let uiLight={};
const UI_LIGHT_VARS={bg:'--bg',panel:'--panel',border:'--border',text:'--text',muted:'--muted',user:'--user',bot:'--bot',bot_border:'--bot-border',input:'--input-bg'};

/** Loose hex check: #rgb or #rrggbb (case-insensitive). */
function isHexColor(v){
  return typeof v==='string' && /^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(v.trim());
}

/** Apply or clear light-theme CSS variable overrides on :root (light only). */
function applyUiLightOverrides(theme){
  const root=document.documentElement;
  Object.keys(UI_LIGHT_VARS).forEach(k=>{
    const cssVar=UI_LIGHT_VARS[k];
    if(theme==='light' && isHexColor(uiLight[k]||'')){
      root.style.setProperty(cssVar, uiLight[k].trim());
    } else {
      // Drop inline override so stylesheet [data-theme="dark"] / defaults win
      root.style.removeProperty(cssVar);
    }
  });
}

function applyTheme(t){
  const theme=(t==='dark')?'dark':'light';
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  applyUiLightOverrides(theme);
  const meta=document.querySelector('meta[name="theme-color"]');
  if(meta){
    const lightBg=(theme==='light' && isHexColor(uiLight.bg||''))?uiLight.bg.trim():'#d8dde5';
    meta.content=(theme==='dark')?'#0f1419':lightBg;
  }
  themeBtn.textContent=(theme==='dark')?'☀':'☾';
  themeBtn.title=(theme==='dark')?'Switch to light':'Switch to dark';
}
(function initTheme(){
  const saved=localStorage.getItem('theme');
  applyTheme(saved==='dark'?'dark':'light'); // light DEFAULT
})();
themeBtn.onclick=()=>{
  const cur=document.documentElement.getAttribute('data-theme')||'light';
  applyTheme(cur==='dark'?'light':'dark');
};

/* Tools toggle: persisted; default off (matches TOOLS_DEFAULT=0). */
(function initTools(){
  const s=localStorage.getItem('tools');
  toolsToggle.checked = (s==='1' || s==='true');
})();
toolsToggle.onchange=()=>{
  localStorage.setItem('tools', toolsToggle.checked?'1':'0');
};

/* Think toggle (v0.45): persisted; default ON. Off asks reasoning models
   (e.g. qwen3 on Ollama) to skip thinking for faster replies. */
(function initThink(){
  const s=localStorage.getItem('think');
  thinkToggle.checked = !(s==='0' || s==='false');
})();
thinkToggle.onchange=()=>{
  localStorage.setItem('think', thinkToggle.checked?'1':'0');
};

/* Put a folded "Thinking" box at the top of a bot message (v0.45).
   Uses textContent only, so model text can never inject HTML. */
function addThinking(msgEl, thinking){
  if(!msgEl || !thinking) return;
  const d=document.createElement('details'); d.className='think';
  const sm=document.createElement('summary'); sm.textContent='Thinking';
  const pre=document.createElement('pre'); pre.textContent=thinking;
  d.appendChild(sm); d.appendChild(pre);
  msgEl.insertBefore(d, msgEl.firstChild);
}


/** Store ui_light map from API and re-apply if currently in light theme. */
function setUiLight(obj){
  uiLight=(obj && typeof obj==='object')?obj:{};
  const cur=document.documentElement.getAttribute('data-theme')||'light';
  if(cur==='light') applyUiLightOverrides('light');
}

/** Apply blink period from config (STATUS_BLINK_MS); green never blinks. */
function applyBlinkMs(ms){
  const n=Number(ms); const v=(n&&n>=200)?n:2000;
  document.documentElement.style.setProperty('--blink', v+'ms');
  catalog.status_blink_ms=v;
}

/** Providers visible for current Mode + Task (server already filters by ?task=). */
function providersForMode(){
  const m=(modeSel.value||'online');
  const task=(taskSel.value||'chat');
  const out={};
  Object.keys(catalog.providers||{}).forEach(pid=>{
    const meta=catalog.providers[pid]||{};
    if((meta.mode||'online')!==m) return;
    const tasks=meta.tasks;
    // Missing tasks → treat as chat-only (matches server default).
    const list=(tasks&&tasks.length)?tasks:['chat'];
    if(task && list.indexOf(task)<0) return;
    out[pid]=meta;
  });
  return out;
}


function fillTasks(){
  taskSel.innerHTML='';
  const tasks=(catalog.tasks&&catalog.tasks.length)?catalog.tasks:[
    {id:'chat',label:'Chat'},{id:'image',label:'Image'},
    {id:'video',label:'Video'},{id:'vision',label:'Vision'}
  ];
  // Locked order: Chat | Image | Video | Vision
  const order=['chat','image','video','vision'];
  const byId={}; tasks.forEach(t=>{byId[t.id]=t;});
  order.forEach(id=>{
    const t=byId[id]||{id:id,label:id.charAt(0).toUpperCase()+id.slice(1)};
    const o=document.createElement('option');
    o.value=t.id; o.textContent=t.label||t.id;
    taskSel.appendChild(o);
  });
  let st=localStorage.getItem('task');
  if(st && [...taskSel.options].some(o=>o.value===st)) taskSel.value=st;
  else taskSel.value=catalog.default_task||'chat';
  updateComposerHint();
}

function updateComposerHint(){
  const task=taskSel.value||'chat';
  if(task==='image') input.placeholder='Describe the image to generate…';
  else if(task==='video') input.placeholder='Describe the video to generate…';
  else if(task==='vision') input.placeholder='Ask about the uploaded image…';
  else input.placeholder='Ask something…';
  // Vision: prefer image accept; others keep general upload for chat files.
  if(task==='vision') fileInput.setAttribute('accept','image/*');
  else fileInput.setAttribute('accept','*/*');
}

function fillModes(){
  modeSel.innerHTML='';
  const modes=(catalog.modes&&catalog.modes.length)?catalog.modes:[
    {id:'online',label:'Online'},{id:'local',label:'Local'}
  ];
  modes.forEach(mo=>{
    const o=document.createElement('option');
    o.value=mo.id; o.textContent=mo.label||mo.id;
    modeSel.appendChild(o);
  });
  let sm=localStorage.getItem('mode');
  // One-time migrate: mode id offline → local (0.2.0)
  if(sm==='offline'){ sm='local'; localStorage.setItem('mode','local'); }
  if(sm && [...modeSel.options].some(o=>o.value===sm)) modeSel.value=sm;
  else if(catalog.default_mode) modeSel.value=catalog.default_mode;
  else modeSel.value='online';
}

function fillProviders(){
  provider.innerHTML='';
  const filtered=providersForMode();
  Object.keys(filtered).forEach(pid=>{
    const o=document.createElement('option'); o.value=pid;
    const meta=filtered[pid]||{};
    o.textContent=meta.label||pid;
    provider.appendChild(o);
  });
  const sp=localStorage.getItem('p');
  if(sp && filtered[sp]) provider.value=sp;
  else if(catalog.default_provider && filtered[catalog.default_provider]) provider.value=catalog.default_provider;
  else if(provider.options.length) provider.selectedIndex=0;
}

function fillModels(){
  const p=catalog.providers[provider.value]||{models:[]};
  const list=p.models||[];
  model.innerHTML='';
  list.forEach(entry=>{
    const id=(typeof entry==='string')?entry:(entry.id||entry.label||'');
    const label=(typeof entry==='string')?entry:(entry.label||entry.id||id);
    if(!id) return;
    const o=document.createElement('option'); o.value=id; o.textContent=label;
    model.appendChild(o);
  });
  const ids=list.map(e=>(typeof e==='string')?e:(e.id||''));
  const key='m:'+provider.value;
  const s=localStorage.getItem(key);
  const def=catalog.default_model;
  const pdef=(catalog.providers[provider.value]||{}).default;
  if(s&&ids.includes(s)) model.value=s;
  else if(provider.value===catalog.default_provider && ids.includes(def)) model.value=def;
  else if(pdef&&ids.includes(pdef)) model.value=pdef;
  else if(ids[0]) model.value=ids[0];
}

/** Short status + LED: OK / No key / No model (no provider/model name in text). */
function updateStatus(){
  const pid=provider.value;
  const meta=(catalog.providers[pid])||{};
  const label=meta.label||pid;
  const verEl=document.getElementById('ver');
  if(verEl) verEl.textContent='v'+(window._ver||'?');
  applyBlinkMs(catalog.status_blink_ms);

  let ok=false;
  let text='…';
  const mode=meta.mode||modeSel.value||'online';
  if(selectedReady && selectedReady.label){
    ok=!!selectedReady.ready;
    if(ok) text='OK';
    else {
      const reason=selectedReady.reason||'not ready';
      if(reason==='missing API key') text='No key';
      else if(reason==='runtime unreachable') text='No model';
      else if(reason==='model not installed') text='No model';
      else text=(mode==='online')?'No key':'No model';
    }
  } else if(mode==='online'){
    ok=!!keys[pid];
    text=ok?'OK':'No key';
  } else {
    // Local without selected blob yet — use providers_ready map if present
    ok=!!(providersReady&&providersReady[pid]);
    text=ok?'OK':'No model';
  }
  status.textContent=text;
  status.className=ok?'ok':'bad';
  led.className=ok?'ok':'bad';
  led.title=ok?'Ready':'Not ready';
  statusWrapTitle(ok,label);
}
function statusWrapTitle(ok,label){
  const w=document.getElementById('statusWrap');
  if(w) w.title=ok?('Ready: '+label):('Not ready: '+label);
}

function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function add(role,text,tools){const d=document.createElement('div'); d.className='msg '+(role==='user'?'user':'bot'); d.textContent=text;
  if(tools&&tools.length){const x=document.createElement('details'); x.className='tools'; x.innerHTML='<summary>'+tools.length+' tool(s)</summary><pre>'+esc(JSON.stringify(tools,null,2))+'</pre>'; d.appendChild(x);
    const names=[]; tools.forEach(t=>{const r=t.result||{}; const n=r.name||(r.path&&String(r.path).split('/').pop()); if(n&&(t.name==='write_file'||r.path)) names.push(n);});
    if(names.length){const dl=document.createElement('div'); dl.className='dl'; dl.innerHTML=names.map(n=>'<a href="/api/download?name='+encodeURIComponent(n)+'" download>'+esc(n)+'</a>').join(' · '); d.appendChild(dl);}
  }
  if(role!=='user' && text && text.indexOf('[[download:')>=0){
    const dl=document.createElement('div'); dl.className='dl';
    const re=/\[\[download:([^\]]+)\]\]/g; let m; const seen={};
    while((m=re.exec(text))){const n=m[1].trim(); if(n&&!seen[n]){seen[n]=1; dl.innerHTML+=(dl.innerHTML?' · ':'')+'<a href="/api/download?name='+encodeURIComponent(n)+'" download>'+esc(n)+'</a>';}}
    if(dl.innerHTML) d.appendChild(dl);
  }
  log.appendChild(d); log.scrollTop=log.scrollHeight;}
async function refreshPending(){
  try{
    const p=await fetch('/api/pending').then(r=>r.json());
    if(p.command){
      pendingCmd.textContent=p.command;
      pendingIsSudo=!!p.sudo;
      confirmBar.classList.add('show');
      if(pendingIsSudo){sudoPassWrap.classList.add('show');}
      else{sudoPassWrap.classList.remove('show'); sudoPass.value='';}
    }else{
      confirmBar.classList.remove('show');
      sudoPassWrap.classList.remove('show');
      sudoPass.value='';
      pendingIsSudo=false;
    }
  }catch(e){}
}
/** Health poll with current provider/model so LED matches selection. */
async function fetchHealth(){
  const q='?provider='+encodeURIComponent(provider.value||'')+'&model='+encodeURIComponent(model.value||'');
  const h=await fetch('/api/health'+q).then(r=>r.json());
  keys=h.keys||keys;
  providersReady=h.providers_ready||providersReady;
  selectedReady=h.selected||null;
  if(h.status_blink_ms) applyBlinkMs(h.status_blink_ms);
  if(h.ui_light) setUiLight(h.ui_light);
  if(h.version) window._ver=h.version;
  return h;
}

/** Load persisted Markdown chat (GET /api/chat/history) and render turns.
 *  .md on disk is source of truth; JS is display-only. Cap in-memory history
 *  sent to the model at 20 entries (server also applies HISTORY_TURNS). */
async function loadChatHistory(){
  try{
    const data=await fetch('/api/chat/history').then(r=>r.json());
    const turns=data.turns||[];
    if(!turns.length) return;
    // Clear any placeholder; re-render from disk.
    log.innerHTML='';
    turns.forEach(t=>{
      const role=(t.role==='user')?'user':'bot';
      add(role, t.content||'');
    });
    // Keep a short tail for the next POST /api/chat history field.
    history=turns.map(t=>({role:t.role==='user'?'user':'assistant',content:t.content||''}));
    if(history.length>20) history=history.slice(-20);
  }catch(e){ /* offline / empty — leave blank chat */ }
}

async function boot(){
  try{
    const savedTask=localStorage.getItem('task')||'chat';
    const m=await fetch('/api/models?task='+encodeURIComponent(savedTask)).then(r=>r.json());
    catalog={
      tasks:m.tasks||[],
      modes:m.modes||[],
      providers:m.providers||{},
      default_provider:m.default_provider||'gemini',
      default_model:m.default_model||'',
      default_mode:m.default_mode||'online',
      default_task:m.default_task||'chat',
      status_blink_ms:m.status_blink_ms||2000
    };
    if(m.ui_light) setUiLight(m.ui_light);
    applyBlinkMs(catalog.status_blink_ms);
    fillTasks();
    fillModes();
    fillProviders();
    fillModels();
    const h=await fetchHealth();
    window._ws=h.workspace||'';
    window._ver=h.version||'?';
    if(h.due_reminders&&h.due_reminders.length){
      note.textContent='Due reminders: '+h.due_reminders.map(r=>r.text).join('; ');
    } else note.textContent='';
    updateStatus();
    await refreshPending();
    await loadChatHistory();
  }catch(e){
    status.textContent='No server';
    status.className='bad';
    led.className='bad';
  }
}
async function reloadCatalogForTask(){
  const task=taskSel.value||'chat';
  try{
    const m=await fetch('/api/models?task='+encodeURIComponent(task)).then(r=>r.json());
    catalog.providers=m.providers||catalog.providers;
    catalog.tasks=m.tasks||catalog.tasks;
    if(m.ui_light) setUiLight(m.ui_light);
    fillProviders();
    fillModels();
  }catch(e){}
  updateComposerHint();
}

taskSel.onchange=()=>{
  localStorage.setItem('task',taskSel.value);
  pendingVisionImage=null;
  reloadCatalogForTask().then(()=>fetchHealth().then(updateStatus).catch(()=>updateStatus()));
};

modeSel.onchange=()=>{
  localStorage.setItem('mode',modeSel.value);
  fillProviders();
  fillModels();
  localStorage.setItem('p',provider.value);
  fetchHealth().then(updateStatus).catch(()=>updateStatus());
};
provider.onchange=()=>{
  localStorage.setItem('p',provider.value);
  fillModels();
  fetchHealth().then(updateStatus).catch(()=>updateStatus());
};
model.onchange=()=>{
  localStorage.setItem('m:'+provider.value,model.value);
  fetchHealth().then(updateStatus).catch(()=>updateStatus());
};

document.getElementById('btnUpdate').onclick=async()=>{
  if(!confirm('Pull latest from git and restart the agent service?')) return;
  try{
    const res=await fetch('/api/update',{method:'POST'}).then(r=>r.json());
    add('bot', res.ok ? (res.message||('Updated to v'+(res.version||'?'))) : ('Update failed: '+(res.message||res.error||'?')));
    if(res.version){window._ver=res.version; updateStatus();}
  }catch(e){add('bot','Update error: '+e);}
};

document.getElementById('btnConfirm').onclick=async()=>{
  const body={};
  if(pendingIsSudo){
    body.password=sudoPass.value||'';
  }
  let res;
  try{
    res=await fetch('/api/confirm',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)
    }).then(r=>r.json());
  }finally{
    sudoPass.value=''; // discard password from the field
  }
  add('bot', res.ok?('Done.'+((res.stdout||res.stderr)?('\n'+(res.stdout||'')+(res.stderr?('\n'+res.stderr):'')):'')):(res.error||'Confirm failed'));
  refreshPending();
};
document.getElementById('btnCancel').onclick=async()=>{
  sudoPass.value='';
  await fetch('/api/cancel',{method:'POST'});
  add('bot','Shell cancelled.');
  refreshPending();
};
fileInput.onchange=async()=>{
  const f=fileInput.files&&fileInput.files[0]; if(!f) return;
  send.disabled=true;
  try{
    const buf=await f.arrayBuffer();
    const bytes=new Uint8Array(buf);
    let b64=''; const chunk=0x8000;
    for(let i=0;i<bytes.length;i+=chunk){b64+=String.fromCharCode.apply(null,bytes.subarray(i,i+chunk));}
    b64=btoa(b64);
    const task=taskSel.value||'chat';
    // Vision: keep image in memory for the next Send (also upload to workspace).
    if(task==='vision' || /^image\//i.test(f.type||'')){
      pendingVisionImage={name:f.name,content:b64,encoding:'base64',mime:f.type||'image/jpeg'};
    }
    const isText=/\.(txt|md|py|sh|json|csv|html|css|js|env|log|yml|yaml)$/i.test(f.name)&&f.size<200000;
    let body;
    if(isText){
      const text=await f.text();
      body={name:f.name,content:text};
    }else{
      body={name:f.name,content:b64,encoding:'base64'};
    }
    const res=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
    if(res.ok){
      if(task==='vision') add('bot','Vision image ready: '+f.name+' (click Send with your question)');
      else add('bot','Uploaded '+f.name+' → download: /api/download?name='+encodeURIComponent(f.name)+'  ([[download:'+f.name+']])');
    } else add('bot','Upload failed: '+(res.error||'?'));
  }catch(e){add('bot','Upload error: '+e);} finally{fileInput.value=''; send.disabled=false;}
};
input.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit();}
});
document.getElementById('f').onsubmit=async ev=>{
  ev.preventDefault();
  const task=taskSel.value||'chat';
  let message=input.value.trim();
  if(!message && task!=='vision') return;
  if(!message && task==='vision') message='Describe this image.';
  if(task==='vision' && !pendingVisionImage){
    add('bot','Vision needs an image — use Upload first, then Send.');
    return;
  }
  add('user', message+(task!=='chat'?'  ['+task+']':''));
  input.value=''; send.disabled=true;
  try{
    const payload={
      message, provider:provider.value, model:model.value, history,
      tools:!!toolsToggle.checked, think:!!thinkToggle.checked, task
    };
    if(task==='vision' && pendingVisionImage){
      payload.image=pendingVisionImage;
    }
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)});
    const data=await res.json();
    if(!data.ok) add('bot', data.error||data.reply||'Failed', data.tools); else {
      add('bot', data.reply, data.tools);
      // Folded thinking above the reply text (only when the model sent some).
      if(data.thinking) addThinking(log.lastChild, data.thinking);
      // Speed line under the reply (v0.44): "42 tokens · 12.3 t/s".
      // Only shown when the provider sent a token count.
      if(data.stats && data.stats.tps && log.lastChild){
        const sp=document.createElement('div'); sp.className='speed';
        sp.textContent=data.stats.tokens+' tokens · '+data.stats.tps+' t/s';
        sp.title=data.stats.seconds+'s model time';
        log.lastChild.appendChild(sp);
      }
      // Surface generated media download links from media field too.
      if(data.media && data.media.name){
        const dl=document.createElement('div'); dl.className='dl';
        dl.innerHTML='<a href="/api/download?name='+encodeURIComponent(data.media.name)+'" download>'+esc(data.media.name)+'</a>';
        log.lastChild && log.lastChild.appendChild(dl);
      }
      history.push({role:'user',content:message},{role:'assistant',content:data.reply});
      if(history.length>20) history=history.slice(-20);
    }
    if(task==='vision') pendingVisionImage=null;
    await refreshPending();
    await fetchHealth();
    updateStatus();
    const h=await fetch('/api/health').then(r=>r.json());
    if(h.due_reminders&&h.due_reminders.length) note.textContent='Due reminders: '+h.due_reminders.map(r=>r.text).join('; ');
  }catch(e){add('bot','Network error: '+e);} finally{send.disabled=false; input.focus();}
};
boot();
setInterval(refreshPending, 4000);
setInterval(async()=>{try{
  const h=await fetchHealth();
  updateStatus();
  if(h.due_reminders&&h.due_reminders.length) note.textContent='Due reminders: '+h.due_reminders.map(r=>r.text).join('; ');
  else if(note.textContent.startsWith('Due reminders:')) note.textContent='';
}catch(e){}}, 15000);
