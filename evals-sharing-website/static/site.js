"use strict";
window.Site = {
  base: () => document.querySelector('meta[name="base-path"]')?.content ?? '',
  localPath(value) {
    if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || /[\\\u0000-\u0020]/.test(value)) throw new Error('Invalid application path.');
    const base=this.base();
    const local=base && (value===base || value.startsWith(base+'/')) ? value.slice(base.length)||'/' : value;
    if (local.split(/[?#]/)[0].split('/').some(part=>part==='.'||part==='..')) throw new Error('Invalid application path.');
    return local;
  },
  path(value) { return this.base()+this.localPath(value); },
  recordURL(value, collection='(?:evaluations|training-runs)') {
    const local=this.localPath(value);
    if (!new RegExp(`^/${collection}/[a-zA-Z0-9_-]+$`).test(local)) throw new Error('Invalid research record link.');
    return this.path(local);
  },
  navigate(value) { location.assign(this.path(value)); },
  csrf: () => document.querySelector('meta[name="csrf-token"]')?.content ?? '',
  toast(message, error = false) {
    const box = document.createElement('div'); box.className = `toast${error ? ' error' : ''}`;
    if (error) box.setAttribute('role', 'alert');
    const text = document.createElement('span'); text.textContent = message;
    const close = document.createElement('button'); close.type = 'button'; close.textContent = '×'; close.setAttribute('aria-label', 'Dismiss notification'); close.onclick = () => box.remove();
    box.append(text, close); document.getElementById('toasts').append(box);
    if (!error) setTimeout(() => box.remove(), 7000);
  },
  async request(url, options = {}) {
    const headers = new Headers(options.headers);
    if (options.method && options.method !== 'GET') headers.set('X-CSRF-Token', this.csrf());
    const response = await fetch(this.path(url), {...options, headers, credentials: 'same-origin'});
    let data; try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) {
      const detail = data.detail;
      const message = typeof detail === 'string' ? detail : typeof detail?.message === 'string' ? detail.message : Array.isArray(detail) ? detail.map(x => `${x.loc?.join('.') ?? ''}: ${x.msg}`).join('\n') : `Request failed (${response.status}).`;
      const error = new Error(message); error.data = data; error.status = response.status; throw error;
    }
    return data;
  },
  showError(form, error) {
    const target = form.querySelector('.form-error'); target.replaceChildren(document.createTextNode(error.message));
    const refs = error.data?.references ?? error.data?.detail?.references;
    if (Array.isArray(refs)) for (const ref of refs) {
      const href = ref.url ?? (ref.evaluation_id ? `/evaluations/${encodeURIComponent(ref.evaluation_id)}` : null);
      let url;try { if(href)url=this.recordURL(href); } catch { /* Untrusted references remain plain text. */ }
      if (url) { const link = document.createElement('a'); link.href = url; link.textContent = ref.title ?? ref.name ?? 'Referenced record'; target.append(document.createElement('br'), link); }
      else target.append(document.createTextNode(`\n${ref.title ?? ref.name ?? ref.id ?? 'Record remains referenced.'}`));
    }
    const fields = error.data?.fields ?? {};
    form.querySelectorAll('[aria-invalid]').forEach(el => el.removeAttribute('aria-invalid'));
    for (const [name, message] of Object.entries(fields)) {
      const control = form.elements.namedItem(name); if (control) { control.setAttribute('aria-invalid', 'true'); control.setAttribute('title', String(message)); }
    }
    this.toast(error.message, true);
  }
};
document.addEventListener('DOMContentLoaded', () => {
  const edit = document.querySelector('[data-edit-url]');
  document.querySelector('[data-edit-toggle]')?.addEventListener('click', () => { edit.hidden = !edit.hidden; if (!edit.hidden) edit.querySelector('input:not([type=hidden])').focus(); });
  edit?.querySelector('[data-edit-cancel]').addEventListener('click', () => { edit.reset(); edit.hidden = true; document.querySelector('[data-edit-toggle]').focus(); });
  edit?.addEventListener('submit', async event => {
    event.preventDefault(); const button = edit.querySelector('[type=submit]'); button.disabled = true;
    const values = Object.fromEntries(new FormData(edit)); values.version = Number(values.version);
    try {
      const updated = await Site.request(edit.dataset.editUrl, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify(values)});
      const record = updated.evaluation ?? updated.run ?? updated;
      edit.elements.version.value = record.version;
      // Cancellation returns to the latest successful save, including its version.
      for (const control of edit.elements) if ('defaultValue' in control) control.defaultValue = control.value;
      document.querySelector('[data-record-heading]').textContent = values.title ?? values.name;
      document.querySelector('[data-record-notes]').textContent = values.notes || 'No notes.'; edit.querySelector('.form-error').textContent = ''; edit.hidden = true; Site.toast('Annotations saved.');
    } catch (error) { Site.showError(edit,error); } finally { button.disabled = false; }
  });
  const dialog = document.getElementById('delete-dialog'); let deletion;
  document.querySelector('[data-delete-url]')?.addEventListener('click', event => { deletion = event.currentTarget; dialog.querySelector('[data-dialog-error]').textContent=''; dialog.querySelector('#delete-title').textContent=`Permanently delete “${deletion.dataset.deleteName}”?`; dialog.showModal(); });
  document.getElementById('confirm-delete')?.addEventListener('click', async event => {
    const button=event.currentTarget;button.disabled=true;
    try { await Site.request(deletion.dataset.deleteUrl,{method:'DELETE'}); sessionStorage.setItem('site-notice','Record permanently deleted.'); Site.navigate('/'); }
    catch(error) { Site.showError(dialog,error); } finally { button.disabled=false; }
  });
  document.querySelector('[data-logout]')?.addEventListener('submit', async event => { event.preventDefault(); try { await Site.request('/auth/logout',{method:'POST'}); Site.navigate('/'); } catch(error) { Site.toast(error.message,true); } });
  const notice=sessionStorage.getItem('site-notice'); if(notice){ sessionStorage.removeItem('site-notice'); Site.toast(notice); }
});
