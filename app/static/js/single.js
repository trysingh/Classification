/* single.js - upload -> preview/column mapping -> supplier -> submit -> live progress -> results. */
(() => {
  const {$, esc, api, upload, showErr, errorHtml, setBusy, poll, uuid, fmtBytes, show, hide, toast} = App;
  const state = {uploadId: null, guesses: {}, columns: [], token: uuid()};   // token: a double click or retry creates ONE job
  const requireSupplier = $('#form-view').dataset.requireSupplier === 'true';
  const suppliers = JSON.parse($('#suppliers-data').textContent);
  const drop = $('#drop'), input = $('#file');

  /* ---- choosing a file ---- */
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => e.dataTransfer.files[0] && pick(e.dataTransfer.files[0]));
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => input.files[0] && pick(input.files[0]));
  $('#pv-clear').addEventListener('click', () => { reset(); input.value = ''; });

  function reset(keepError) {
    Object.assign(state, {uploadId: null, guesses: {}, columns: []});
    hide('#preview'); hide('#columns'); show('#drop');
    if (!keepError) hide('#file-error');
    refresh();
  }

  async function pick(file) {
    reset();
    hide('#drop'); show('#upload-state');
    $('#upload-bar').style.width = '0%';
    $('#upload-label').textContent = `Uploading ${file.name}...`;
    const fd = new FormData(); fd.append('file', file);
    try {
      const d = await upload('/api/single/upload', fd, p => {
        $('#upload-bar').style.width = Math.round(p * 100) + '%';
        if (p >= 1) $('#upload-label').textContent = 'Reading the file...';
      });
      Object.assign(state, {uploadId: d.upload_id, guesses: d.guesses, columns: d.columns});
      renderPreview(d);
    } catch (e) {
      show('#drop'); showErr('#file-error', e); input.value = '';
    } finally { hide('#upload-state'); }
    refresh();
  }

  function renderPreview(d) {
    $('#pv-name').textContent = d.filename;
    $('#pv-meta').textContent = `${d.rows.toLocaleString()} rows, ${d.columns.length} columns, ${fmtBytes(d.size)}`;
    $('#pv-table').innerHTML = `<thead><tr>${d.columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>` +
      d.sample.map(r => `<tr>${d.columns.map(c => `<td><div class="trunc" style="max-width:16rem">${esc(r[c])}</div></td>`).join('')}</tr>`).join('') + '</tbody>';
    const opts = d.columns.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
    $('#text-col').innerHTML = opts;
    $('#amount-col').innerHTML = `<option value="">None</option>` + opts;
    applyGuess();
    hide('#drop'); show('#preview'); show('#columns');
  }
  /* Column guesses depend on the chosen dataset type, so re-apply when it changes. */
  function applyGuess() {
    const g = state.guesses[$('#profile').value] || {};
    if (g.text) $('#text-col').value = g.text;
    $('#amount-col').value = g.amount || '';
  }
  $('#profile').addEventListener('change', () => state.uploadId && applyGuess());

  /* ---- supplier: picking a known name fills the rest ---- */
  $('#sup-name').addEventListener('input', () => {
    const hit = suppliers.find(s => s.name.toLowerCase() === $('#sup-name').value.trim().toLowerCase());
    if (hit) { $('#sup-code').value = hit.code; $('#sup-country').value = hit.country; $('#sup-email').value = hit.contact_email; $('#sup-notes').value = hit.notes; }
    $('#sup-known').hidden = !hit;
    refresh();
  });

  function refresh() {
    const needSup = requireSupplier && !$('#sup-name').value.trim();
    $('#go').disabled = !state.uploadId || needSup;
    $('#go-hint').textContent = !state.uploadId ? 'Upload a file to continue.' : needSup ? 'Enter the supplier name to continue.' : 'Ready. The file is processed in the background.';
  }

  /* ---- submit ---- */
  $('#go').addEventListener('click', async () => {
    const btn = $('#go');
    hide('#submit-error'); setBusy(btn, true, 'Starting...');
    try {
      const r = await api('/api/single/submit', {method: 'POST', json: {
        upload_id: state.uploadId, profile: $('#profile').value, backend: $('#backend').value,
        narration_column: $('#text-col').value, amount_column: $('#amount-col').value, client_token: state.token,
        supplier: {name: $('#sup-name').value.trim(), code: $('#sup-code').value.trim(), country: $('#sup-country').value.trim(),
                   contact_email: $('#sup-email').value.trim(), notes: $('#sup-notes').value.trim()}}});
      startRun(r);
    } catch (e) { showErr('#submit-error', e); setBusy(btn, false); }
  });

  /* ---- live run panel: polls the job; survives dropped connections (banner + retry) ---- */
  function startRun(r) {
    hide('#form-view'); show('#run-view'); window.scrollTo(0, 0);
    $('#run-job').href = `/jobs/${r.job_id}`;
    poll(`/api/jobs/${r.job_id}/status`, {
      until: d => d.finished,
      onData: d => {
        const f = d.files[0], total = f.total_rows || 0, counting = f.status === 'running' && f.phase === 'Classifying' && total > 0;
        $('#run-phase').textContent = f.status === 'queued' ? 'Waiting for a free worker...' : f.phase + '...';
        $('#run-progress').classList.toggle('indet', !counting);
        $('#run-bar').style.width = counting ? f.percent + '%' : '35%';
        $('#run-count').textContent = counting ? `${f.processed_rows.toLocaleString()} of ${total.toLocaleString()} rows` : '';
        if (d.finished) finish(f, r);
      },
      onError: e => { show('#run-error'); $('#run-error').innerHTML = errorHtml(e); },
    });
  }

  function finish(f, r) {
    hide('#run-spin'); hide('#run-note');
    $('#run-progress').classList.remove('indet');
    if (f.status === 'completed') {
      $('#run-title').textContent = 'Classification finished';
      $('#run-phase').textContent = `${f.processed_rows.toLocaleString()} rows classified. Opening the results...`;
      $('#run-bar').style.width = '100%';
      const open = $('#run-open'); open.href = `/files/${r.file_id}`; show(open);
      setTimeout(() => { location.href = open.href; }, 1200);
    } else {
      $('#run-title').textContent = f.status === 'cancelled' ? 'The run was cancelled' : 'The file could not be classified';
      $('#run-phase').textContent = '';
      if (f.status !== 'cancelled') { show('#run-error'); $('#run-error').innerHTML = errorHtml({message: f.error_message || 'Processing stopped unexpectedly.', hint: f.error_hint, detail: f.error_trace}); }
      show('#run-again');
    }
  }
  refresh();
})();
