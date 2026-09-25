/* job_detail.js - renders the job from status JSON, polls until finished, offers cancel / retry. */
(() => {
  const {$, esc, api, poll, setBusy, showErr, errorHtml, show, hide, toast} = App;
  const id = $('#job').dataset.id;
  const LABEL = {queued: 'Queued', running: 'Running', completed: 'Completed', completed_with_errors: 'Completed with errors', failed: 'Failed', cancelled: 'Cancelled'};
  const openTraces = new Set();               // keep "Technical details" open across re-renders
  let stopPolling = null;

  function fileRow(f) {
    const counting = f.status === 'running' && f.phase === 'Classifying' && f.total_rows;
    const sub = f.status === 'failed' ? '' : `<div class="small muted">${esc(f.phase)}${counting ? `: ${f.processed_rows.toLocaleString()} of ${f.total_rows.toLocaleString()} rows` : ''}</div>`;
    const err = f.status === 'failed' ? `<div class="alert error" style="margin-top:.4rem"><strong>${esc(f.error_message || 'Failed')}</strong>${f.error_hint ? `<span>${esc(f.error_hint)}</span>` : ''}
        ${f.error_trace ? `<details data-f="${f.id}" ${openTraces.has(f.id) ? 'open' : ''}><summary>Technical details</summary><pre>${esc(f.error_trace)}</pre></details>` : ''}</div>` : '';
    const bar = f.status === 'failed' ? '' : `<div class="progress ${f.status === 'running' && !counting ? 'indet' : ''}"><i style="width:${f.status === 'running' && !counting ? 35 : f.percent}%"></i></div>`;
    const open = f.processed_rows ? `<a class="btn sm" href="/files/${f.id}">${f.status === 'completed' ? 'Results' : 'Partial results'}</a>` : '';
    return `<tr>
      <td><strong>${esc(f.filename)}</strong>${f.supplier ? `<div class="small muted">${esc(f.supplier)}</div>` : ''}</td>
      <td><span class="badge s-${f.status}">${LABEL[f.status] || f.status}</span></td>
      <td>${bar}${sub}${err}</td>
      <td class="r">${f.review_rows.toLocaleString()}</td><td class="r">${f.others_rows.toLocaleString()}</td>
      <td class="r">${open}</td></tr>`;
  }

  function render(d) {
    const st = $('#job-status'); st.className = `badge s-${d.status}`; st.textContent = LABEL[d.status] || d.status;
    $('#files-body').innerHTML = d.files.map(fileRow).join('');
    const c = d.counts;
    $('#overall').textContent = `${c.completed} of ${c.files} done${c.failed ? `, ${c.failed} failed` : ''}, ${Math.round(d.percent)}% overall`;
    $('#live').hidden = d.finished;
    if (d.finished) App.refreshActive();
    $('#cancel').hidden = d.finished;
    $('#retry').hidden = !(d.finished && (c.failed || c.cancelled));
    if (d.error_message) { show('#job-error'); $('#job-error').innerHTML = errorHtml({message: d.error_message}); }
  }

  $('#files-body').addEventListener('toggle', e => {
    const f = e.target.dataset && e.target.dataset.f; if (!f) return;
    e.target.open ? openTraces.add(+f) : openTraces.delete(+f);
  }, true);

  function start() {
    stopPolling = poll(`/api/jobs/${id}/status`, {until: d => d.finished, onData: render, onError: e => { show('#job-error'); $('#job-error').innerHTML = errorHtml(e); }});
  }

  $('#cancel').addEventListener('click', async () => {
    if (!confirm('Cancel this job? Files already finished are kept; the file in progress stops at its next checkpoint.')) return;
    const b = $('#cancel'); setBusy(b, true, 'Cancelling...');
    try { render(await api(`/api/jobs/${id}/cancel`, {method: 'POST'})); toast('Cancel requested.', 'ok'); }
    catch (e) { showErr('#job-error', e); }
    finally { setBusy(b, false); }
  });

  $('#retry').addEventListener('click', async () => {
    const b = $('#retry'); setBusy(b, true, 'Retrying...'); hide('#job-error');
    try { render(await api(`/api/jobs/${id}/retry`, {method: 'POST'})); toast('Retry started. Rows already classified are kept.', 'ok'); start(); }
    catch (e) { showErr('#job-error', e); }
    finally { setBusy(b, false); }
  });

  const initial = JSON.parse($('#job-data').textContent);
  render(initial);
  if (!initial.finished) start();
})();
