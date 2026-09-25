/* file_detail.js - live progress for an in-flight file, plus reprocess and delete actions. */
(() => {
  const {$, api, poll, setBusy, showErr, toast} = App;
  const root = $('#file'), id = root.dataset.id, jobId = root.dataset.job;

  if (['queued', 'running'].includes(root.dataset.status)) {
    poll(`/api/jobs/${jobId}/status`, {
      until: d => d.finished,
      onData: d => {
        const f = d.files.find(x => String(x.id) === id) || d.files[0], counting = f.phase === 'Classifying' && f.total_rows;
        $('#live-phase').textContent = f.status === 'queued' ? 'Waiting for a free worker...' : f.phase + '...';
        $('#live-progress').classList.toggle('indet', !counting);
        $('#live-bar').style.width = counting ? f.percent + '%' : '35%';
        $('#live-count').textContent = counting ? `${f.processed_rows.toLocaleString()} of ${f.total_rows.toLocaleString()} rows` : '';
        if (d.finished) location.reload();
      },
    });
  }

  const re = $('#reprocess');
  if (re) re.addEventListener('click', async () => {
    setBusy(re, true, 'Starting...');
    try { const r = await api(`/api/files/${id}/reprocess`, {method: 'POST'}); location.href = `/jobs/${r.job_id}`; }
    catch (e) { showErr('#action-error', e); setBusy(re, false); }
  });

  const del = $('#delete');
  if (del) del.addEventListener('click', async () => {
    if (!confirm('Delete this result and its downloads? The original file is not touched.')) return;
    setBusy(del, true, 'Deleting...');
    try { await api(`/api/files/${id}/delete`, {method: 'POST'}); location.href = '/files?ok=' + encodeURIComponent('Result deleted.'); }
    catch (e) { showErr('#action-error', e); setBusy(del, false); }
  });
})();
