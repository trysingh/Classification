/* taxonomy.js - tree editor kept in memory; nothing is stored until "Save taxonomy". */
(() => {
  const {$, $$, esc, api, showErr, setBusy, toast, catClass, show, hide} = App;
  const root = $('#tax'), profile = root.dataset.profile, OTHERS = root.dataset.others;
  const maxMain = +root.dataset.maxMain, maxSub = +root.dataset.maxSub;
  const data = JSON.parse($('#tax-data').textContent);

  const toItems = obj => {
    const items = Object.entries(obj || {}).filter(([k]) => k !== OTHERS).map(([name, subs]) => ({name, subs: [...subs]}));
    return [...items, {name: OTHERS, subs: []}];                       // the catch-all always exists and stays last
  };
  const toObj = items => Object.fromEntries(items.map(i => [i.name, i.name === OTHERS ? [] : i.subs]));
  const lc = s => s.trim().toLowerCase();

  let items = toItems(data.taxonomy), baseline = JSON.stringify(toObj(items));
  const isDirty = () => JSON.stringify(toObj(items)) !== baseline;

  function msg(text, kind = 'info') { const m = $('#tax-msg'); m.className = `alert ${kind}`; m.textContent = text; show(m); }

  function render(focus) {
    $('#editor').innerHTML = items.map((it, i) => {
      const o = it.name === OTHERS, over = !o && it.subs.length > maxSub;
      return `<div class="tax-card ${catClass(it.name, OTHERS)}" data-i="${i}">
        <div class="head"><input type="text" class="name" value="${esc(it.name)}" title="${esc(it.name)}" ${o ? 'readonly' : ''} aria-label="Category name" maxlength="80">
          ${o ? '' : '<button type="button" class="btn sm danger" data-act="del-main">Remove</button>'}</div>
        ${o ? `<p class="hint">Catch-all. Rows that fit nowhere land here, and the engine names a specific sub-category for each one.</p>` : `
          <div class="chips">${it.subs.map((s, j) => `<span class="chip">${esc(s)}<button type="button" data-act="del-sub" data-j="${j}" aria-label="Remove ${esc(s)}">&times;</button></span>`).join('') || '<span class="hint">No sub-categories yet.</span>'}</div>
          <input type="text" class="sub-in" placeholder="Add a sub-category, press Enter" maxlength="80" autocomplete="off">
          <span class="hint" ${over ? 'style="color:var(--warn)"' : ''}>${it.subs.length} of ${maxSub} recommended${over ? ', small models lose accuracy above this' : ''}</span>`}
      </div>`;
    }).join('');
    const over = items.length > maxMain, b = $('#count-badge');
    b.textContent = `${items.length} of ${maxMain} categories`; b.className = `badge ${over ? 'b-warn' : ''}`;
    $('#dirty').hidden = !isDirty(); $('#save').disabled = !isDirty();
    if (focus != null) { const el = $(`.tax-card[data-i="${focus}"] .sub-in`); if (el) el.focus(); }
  }

  function addMain() {
    const inp = $('#new-main'), name = inp.value.replace(/\s+/g, ' ').trim();
    if (!name) return;
    if (items.some(i => lc(i.name) === lc(name))) return msg(`"${name}" already exists.`, 'warn');
    items.splice(items.length - 1, 0, {name, subs: []});
    inp.value = ''; hide('#tax-msg'); render(items.length - 2);
  }
  $('#add-main').addEventListener('click', addMain);
  $('#new-main').addEventListener('keydown', e => { if (e.key === 'Enter') addMain(); });

  const ed = $('#editor');
  ed.addEventListener('click', e => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const i = +b.closest('.tax-card').dataset.i;
    if (b.dataset.act === 'del-main') { items.splice(i, 1); render(); }
    if (b.dataset.act === 'del-sub') { items[i].subs.splice(+b.dataset.j, 1); render(i); }
  });
  ed.addEventListener('keydown', e => {
    if (!e.target.classList.contains('sub-in') || e.key !== 'Enter') return;
    const i = +e.target.closest('.tax-card').dataset.i, v = e.target.value.replace(/\s+/g, ' ').trim();
    if (!v) return;
    if (items[i].subs.some(s => lc(s) === lc(v))) return msg(`"${v}" is already in ${items[i].name}.`, 'warn');
    items[i].subs.push(v); hide('#tax-msg'); render(i);
  });
  ed.addEventListener('change', e => {
    if (!e.target.classList.contains('name')) return;
    const i = +e.target.closest('.tax-card').dataset.i, v = e.target.value.replace(/\s+/g, ' ').trim();
    if (!v || items.some((x, k) => k !== i && lc(x.name) === lc(v))) { msg(v ? `"${v}" already exists.` : 'A category needs a name.', 'warn'); return render(); }
    items[i].name = v; render();
  });

  /* ---- load / import ---- */
  const starter = $('#use-starter');
  if (starter) starter.addEventListener('click', () => { if (isDirty() && !confirm('Replace your unsaved changes with the starter taxonomy?')) return; items = toItems(data.starter); render(); msg('Starter loaded. Review it, then save.', 'info'); });
  $('#import').addEventListener('click', () => $('#import-file').click());
  $('#import-file').addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const obj = JSON.parse(await f.text());
      const src = obj.taxonomy && typeof obj.taxonomy === 'object' ? obj.taxonomy : obj;          // accept {"taxonomy": {...}} too
      if (!src || Array.isArray(src) || Object.values(src).some(v => !Array.isArray(v))) throw new Error('Expected {"Category": ["Sub-category", ...]}.');
      items = toItems(src); render(); msg(`Imported ${items.length - 1} categories from ${f.name}. Review, then save.`, 'info');
    } catch (err) { msg(`That file is not a valid taxonomy: ${err.message}`, 'error'); }
    e.target.value = '';
  });

  /* ---- save ---- */
  $('#save').addEventListener('click', async () => {
    const btn = $('#save'); setBusy(btn, true, 'Saving...'); hide('#tax-msg');
    try {
      const r = await api('/api/taxonomy/save', {method: 'POST', json: {profile, taxonomy: toObj(items), note: $('#note').value.trim()}});
      items = toItems(r.taxonomy); baseline = JSON.stringify(toObj(items)); render();
      toast(`Saved as version ${r.version}`, 'ok');
      if (r.warnings.length) sessionStorage.setItem('taxWarn', r.warnings.join(' '));   // shown again after the reload
      setTimeout(() => location.reload(), 600);                                          // reload refreshes the history list
    } catch (e) { showErr('#tax-msg', e); $('#tax-msg').className = 'alert error'; }
    finally { setBusy(btn, false); $('#save').disabled = !isDirty(); }
  });

  /* ---- System 2 suggest / restore ---- */
  const gen = $('#gen');
  if (gen) gen.addEventListener('click', async () => {
    if (isDirty() && !confirm('You have unsaved changes. System 2 works from the last SAVED taxonomy, so unsaved edits will be lost. Continue?')) return;
    setBusy(gen, true, 'System 2 is working...'); show('#gen-note'); hide('#gen-error');
    try {
      const mode = document.querySelector('input[name=gen-mode]:checked').value;
      await api('/api/taxonomy/generate', {method: 'POST', json: {profile, file_id: +$('#gen-file').value, mode}});
      location.reload();
    } catch (e) { showErr('#gen-error', e); setBusy(gen, false); hide('#gen-note'); }
  });
  $$('[data-restore]').forEach(b => b.addEventListener('click', async () => {
    if (!confirm('Restore this version? It is saved as a new version, so nothing is lost.')) return;
    setBusy(b, true, 'Restoring...');
    try { await api('/api/taxonomy/restore', {method: 'POST', json: {profile, version_id: +b.dataset.restore}}); location.reload(); }
    catch (e) { showErr('#tax-msg', e); setBusy(b, false); }
  }));

  const carried = sessionStorage.getItem('taxWarn');
  if (carried) { sessionStorage.removeItem('taxWarn'); msg(carried, 'warn'); }
  window.addEventListener('beforeunload', e => { if (isDirty()) { e.preventDefault(); e.returnValue = ''; } });
  render();
})();
