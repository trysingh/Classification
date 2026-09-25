/* app.js - shared helpers for every page. Exposes window.App. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 3 | 8)).toString(16); }));
  const fmtBytes = n => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
  const show = s => { const e = typeof s === 'string' ? $(s) : s; if (e) e.hidden = false; };
  const hide = s => { const e = typeof s === 'string' ? $(s) : s; if (e) e.hidden = true; };

  /* Same colour-per-category rule as the server (crc32 % 8), so the editor matches result pages. */
  const CRC = (() => { const t = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  const crc32 = str => { let c = 0xFFFFFFFF; for (const x of new TextEncoder().encode(str)) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
  const catClass = (name, others = 'Others') => name === others ? 'c-others' : 'c' + (crc32(name) % 8);

  /* ---- API ---- */
  class ApiError extends Error {
    constructor(message, o = {}) { super(message); Object.assign(this, {status: 0, hint: null, detail: null, requestId: null}, o); }
  }
  async function parse(res) {
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON body */ }
    if (res.ok) return body;
    const e = (body && body.error) || {};
    throw new ApiError(e.message || `The server replied with an error (${res.status}).`, {status: res.status, hint: e.hint, detail: e.detail, requestId: e.request_id});
  }
  async function api(url, {method = 'GET', json, form} = {}) {
    const opt = {method, headers: {Accept: 'application/json'}};
    if (json !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(json); }
    if (form) opt.body = form;
    let res;
    try { res = await fetch(url, opt); }
    catch (_) { throw new ApiError('Cannot reach the server. Check your network connection.', {status: 0}); }
    return parse(res);
  }
  /* XHR instead of fetch: gives upload progress. */
  function upload(url, form, onProgress) {
    return new Promise((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open('POST', url);
      x.setRequestHeader('Accept', 'application/json');
      x.upload.onprogress = e => e.lengthComputable && onProgress && onProgress(e.loaded / e.total);
      x.onerror = () => reject(new ApiError('The upload was interrupted. Check your connection and try again.', {status: 0}));
      x.onload = async () => { try { resolve(await parse(new Response(x.responseText, {status: x.status}))); } catch (e) { reject(e); } };
      x.send(form);
    });
  }

  /* ---- feedback ---- */
  const errorHtml = e => `<strong>${esc(e.message)}</strong>` + (e.hint ? `<span>${esc(e.hint)}</span>` : '') +
    (e.detail ? `<details><summary>Technical details</summary><pre>${esc(e.detail)}</pre></details>` : '') +
    (e.requestId ? `<span class="small">Reference: <span class="mono">${esc(e.requestId)}</span></span>` : '');
  function showErr(sel, e) { const el = $(sel); el.innerHTML = errorHtml(e); show(el); }
  function toast(msg, kind = 'info', ms = 4500) {
    const t = document.createElement('div');
    t.className = `toast ${kind}`; t.textContent = msg; t.setAttribute('role', 'status');
    $('#toasts').appendChild(t); setTimeout(() => t.remove(), ms);
  }
  /* Busy state for any button: spinner + label, blocks double clicks. */
  function setBusy(btn, busy, label) {
    if (busy) { btn.dataset.label = btn.innerHTML; btn.innerHTML = `<span class="spin"></span>${esc(label || 'Working...')}`; btn.classList.add('busy'); btn.setAttribute('aria-busy', 'true'); }
    else { if (btn.dataset.label) btn.innerHTML = btn.dataset.label; btn.classList.remove('busy'); btn.removeAttribute('aria-busy'); }
  }

  /* ---- polling that survives network drops: keeps retrying with back-off and shows a banner ---- */
  function setOnline(ok) { const c = $('#conn'); if (c) c.hidden = ok; }
  function poll(url, {every, onData, until, onError} = {}) {
    let stop = false, fails = 0;
    const base = every || +document.body.dataset.poll || 1500;
    (async function loop() {
      while (!stop) {
        try {
          const d = await api(url);
          fails = 0; setOnline(true); onData && onData(d);
          if (until && until(d)) return;
        } catch (e) {
          if (e.status >= 400 && e.status < 500) { onError && onError(e); return; }   // gone / forbidden: stop
          fails++; setOnline(false);                                                   // network or 5xx: keep trying
        }
        await sleep(Math.min(base * (1 + fails), 10000));
      }
    })();
    return () => { stop = true; };
  }

  /* ---- global "job running" pill; refreshActive() updates it immediately when a page knows a job just ended ---- */
  function paintPill(d) {
    const pill = $('#active-pill'); if (!pill) return;
    pill.hidden = d.count === 0;
    $('#active-text').textContent = d.count === 1 ? `1 job running (${Math.round(d.percent)}%)` : `${d.count} jobs running (${Math.round(d.percent)}%)`;
    pill.href = d.count === 1 ? `/jobs/${d.jobs[0].id}` : '/jobs';
  }
  const refreshActive = () => api('/api/jobs/active').then(paintPill).catch(() => {});

  /* ---- shell: mobile menu ---- */
  document.addEventListener('DOMContentLoaded', () => {
    const rail = $('#rail'), tg = $('#rail-toggle');
    if (tg) tg.addEventListener('click', () => rail.classList.toggle('open'));
    document.addEventListener('click', e => { if (rail && rail.classList.contains('open') && !rail.contains(e.target) && e.target !== tg) rail.classList.remove('open'); });
    if ($('#active-pill')) poll('/api/jobs/active', {every: 3000, onData: paintPill, onError: () => {}});
  });

  window.App = {$, $$, esc, sleep, uuid, fmtBytes, show, hide, api, upload, errorHtml, showErr, toast, setBusy, poll, catClass, ApiError, refreshActive};
})();
