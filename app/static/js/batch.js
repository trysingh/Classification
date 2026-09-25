/* batch.js - scan a folder, choose files (+ supplier per file), start ONE background job. */
(() => {
  const {$, esc, api, upload, showErr, setBusy, uuid, fmtBytes, show, hide, toast} = App;
  const token = uuid();            // resubmitting after a dropped connection returns the same job
  const memo = {};                 // per-file choices survive re-scans: {name: {checked, supplier}}
  let files = [];
  const val = s => $(s).value.trim();
  const rows = () => [...document.querySelectorAll('#files-body tr')];

  function remember() {
    rows().forEach(tr => { memo[tr.dataset.name] = {checked: tr.querySelector('.sel').checked, supplier: tr.querySelector('.sup').value}; });
  }

  async function scan() {
    hide('#scan-error');
    const btn = $('#scan'); setBusy(btn, true, 'Scanning...');
    try {
      remember();
      load(await api('/api/batch/scan', {method: 'POST', json: {folder: val('#folder'), profile: $('#profile').value}}));
    } catch (e) { showErr('#scan-error', e); }
    finally { setBusy(btn, false); }
  }

  function load(d) { $('#folder').value = d.folder; files = d.files; render(); }

  function render() {
    show('#files-panel'); show('#options');
    $('#files-title').textContent = files.length ? `${files.length} file${files.length === 1 ? '' : 's'} in this folder` : 'No supported files in this folder';
    $('#files-body').innerHTML = files.map(f => {
      const m = memo[f.name] || {}, usable = !f.error;
      const checked = m.checked ?? (usable && !f.previous && (f.text_column || val('#text-col')));
      const status = f.error ? `<span class="small" style="color:var(--bad)">${esc(f.error)}</span>`
        : f.previous ? `<a class="badge b-info" href="/files/${f.previous.file_id}">Processed before</a>` : '<span class="badge">New</span>';
      const col = f.error ? '<span class="badge b-bad">Unreadable</span>' : f.text_column ? esc(f.text_column) : '<span class="badge b-warn">Not detected</span>';
      return `<tr data-name="${esc(f.name)}">
        <td><input type="checkbox" class="sel" ${checked ? 'checked' : ''} ${usable ? '' : 'disabled'} aria-label="Include ${esc(f.name)}"></td>
        <td><strong>${esc(f.name)}</strong></td><td class="r">${fmtBytes(f.size)}</td><td class="min">${esc(f.modified)}</td>
        <td>${col}</td>
        <td><input type="text" class="sup" list="sup-list" value="${esc(m.supplier || '')}" placeholder="Optional" autocomplete="off"></td>
        <td>${status}</td></tr>`;
    }).join('');
    updateStart();
  }

  function updateStart() {
    const n = rows().filter(tr => tr.querySelector('.sel').checked).length;
    $('#start').disabled = n === 0;
    $('#start-hint').textContent = n ? `${n} file${n === 1 ? '' : 's'} selected. Processing continues on the server if you leave.` : 'Select at least one file.';
    $('#sel-all').checked = n > 0 && n === rows().filter(tr => !tr.querySelector('.sel').disabled).length;
  }

  $('#scan').addEventListener('click', scan);
  $('#profile').addEventListener('change', () => files.length && scan());      // column detection depends on the profile
  $('#folder').addEventListener('keydown', e => { if (e.key === 'Enter') scan(); });
  $('#files-body').addEventListener('change', e => e.target.classList.contains('sel') && updateStart());
  $('#sel-all').addEventListener('change', e => { rows().forEach(tr => { const c = tr.querySelector('.sel'); if (!c.disabled) c.checked = e.target.checked; }); updateStart(); });
  $('#apply-btn').addEventListener('click', () => {
    const v = val('#apply-supplier'); if (!v) return toast('Type a supplier name first.', 'error');
    rows().forEach(tr => { const s = tr.querySelector('.sup'); if (!s.value.trim()) s.value = v; });
  });

  /* Users without server access can drop files straight into the folder. */
  $('#add-files').addEventListener('change', async e => {
    const list = [...e.target.files]; if (!list.length) return;
    const fd = new FormData(); list.forEach(f => fd.append('files', f));
    fd.append('folder', val('#folder')); fd.append('profile', $('#profile').value);
    const label = $('#add-label'), orig = label.textContent;
    label.classList.add('busy'); label.textContent = 'Uploading...'; hide('#scan-error');
    try {
      remember();
      const d = await upload('/api/batch/upload', fd, p => { label.textContent = `Uploading ${Math.round(p * 100)}%`; });
      d.saved.forEach(n => { memo[n] = {checked: true, supplier: ''}; });
      load(d); toast(`${d.saved.length} file${d.saved.length === 1 ? '' : 's'} added to the folder`, 'ok');
    } catch (err) { showErr('#scan-error', err); }
    finally { label.classList.remove('busy'); label.textContent = orig; e.target.value = ''; }
  });

  $('#start').addEventListener('click', async () => {
    hide('#submit-error');
    const items = rows().filter(tr => tr.querySelector('.sel').checked)
      .map(tr => ({name: tr.dataset.name, supplier: tr.querySelector('.sup').value.trim()}));
    const btn = $('#start'); setBusy(btn, true, 'Starting job...');
    try {
      const r = await api('/api/batch/submit', {method: 'POST', json: {
        folder: val('#folder'), profile: $('#profile').value, backend: $('#backend').value,
        items, narration_column: val('#text-col'), client_token: token}});
      location.href = `/jobs/${r.job_id}`;
    } catch (e) { showErr('#submit-error', e); setBusy(btn, false); updateStart(); }
  });

  scan();                                                                      // show the default folder straight away
})();
