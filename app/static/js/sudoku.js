/* sudoku.js - drives the /sudoku Lab: new board, one step at a time, showing which technique
   placed each cell and, for classifier-driven cells, the model's actual probability distribution
   over that cell's legal candidates. Stateless server (see sudoku.py controller): this file owns
   the current grid and resends it with every "next step" request. */
(() => {
  const {$, esc, api, setBusy, showErr} = App;
  let grid = null, solution = null, given = null, lastCell = -1, autoTimer = null, busy = false;

  const sentClass = m => m === 'naked_single' ? 'naked' : m === 'hidden_single' ? 'hidden' : 'classifier';
  const methodLabel = m => m === 'naked_single' ? 'Naked single' : m === 'hidden_single' ? 'Hidden single' : 'Classifier pick';

  function buildGrid() {
    const el = $('#grid');
    el.innerHTML = '';
    for (let i = 0; i < 81; i++) {
      const r = (i / 9) | 0, c = i % 9;
      const d = document.createElement('div');
      d.className = 'sud-cell';
      d.dataset.i = i;
      if (r % 3 === 0) d.dataset.r3 = '0'; else if (r % 3 === 2) d.dataset.r3 = '1';
      if (c % 3 === 0) d.dataset.c3 = '0'; else if (c % 3 === 2) d.dataset.c3 = '1';
      el.appendChild(d);
    }
  }

  function paintGrid() {
    const cells = $('#grid').children;
    for (let i = 0; i < 81; i++) {
      const cell = cells[i];
      cell.textContent = grid[i] || '';
      cell.classList.toggle('given', !!given[i]);
      cell.classList.remove('just-placed');
    }
  }

  function markPlaced(i, method, wrong) {
    const cell = $('#grid').children[i];
    cell.className = 'sud-cell just-placed ' + sentClass(method) + (wrong ? ' wrong' : '');
    if ((i / 9 | 0) % 3 === 0) cell.dataset.r3 = '0'; else if ((i / 9 | 0) % 3 === 2) cell.dataset.r3 = '1';
    if ((i % 9) % 3 === 0) cell.dataset.c3 = '0'; else if ((i % 9) % 3 === 2) cell.dataset.c3 = '1';
    cell.textContent = grid[i];
    lastCell = i;
  }

  function updateProgress() {
    const filled = grid.filter(Boolean).length;
    $('#progress').textContent = `${filled} of 81 cells (${81 - filled} to go)`;
  }

  async function newBoard() {
    const btn = $('#new-board');
    setBusy(btn, true, 'Generating...');
    clearInterval(autoTimer); autoTimer = null; $('#auto-play').textContent = 'Auto-play';
    try {
      const d = await api('/api/sudoku/new', {method: 'POST', json: {difficulty: $('#difficulty').value}});
      grid = d.puzzle.slice(); solution = d.solution; given = d.given; lastCell = -1;
      buildGrid(); paintGrid(); updateProgress();
      $('#next-step').disabled = false; $('#auto-play').disabled = false;
      $('#why').innerHTML = `<span class="hint">${d.clues} clues given, ${d.empties} cells to solve. Press Next step.</span>`;
    } catch (e) { showErr('#why', e); }
    finally { setBusy(btn, false); }
  }

  function renderWhy(d) {
    const badge = `<span class="badge ${d.method === 'classifier' ? 'b-warn' : 'b-info'}">${methodLabel(d.method)}</span>`;
    const pos = `<span class="small muted">row ${d.row + 1}, column ${d.col + 1}</span>`;
    let body = '';
    if (d.method === 'naked_single') {
      body = `<p class="small">This cell had exactly one legal candidate left: <b>${d.value}</b>. Every other digit is already used somewhere in its row, column or box.</p>`;
    } else if (d.method === 'hidden_single') {
      body = `<p class="small"><b>${d.value}</b> has nowhere else it could legally go in this ${esc(d.unit_kind)} -- every other empty cell in that ${esc(d.unit_kind)} has already ruled it out.</p>`;
    } else {
      const cands = d.candidates.join(', ');
      body = `<p class="small">Neither technique applied anywhere on the board. Legal candidates here: <b>${cands}</b>.</p>`;
      if (d.probabilities) {
        const rows = Object.entries(d.probabilities).sort((a, b) => b[1] - a[1])
          .map(([digit, p]) => `<div class="row" style="gap:.5rem;margin-bottom:.2rem">
              <span class="small num" style="width:1.4rem;flex:none">${esc(digit)}</span>
              <div class="bar" style="flex:1"><i style="width:${(p * 100).toFixed(0)}%;background:${+digit === d.model_pick ? (d.correct ? 'var(--ok)' : 'var(--bad)') : 'var(--ink-3)'}"></i></div>
              <span class="small num" style="width:2.5rem;text-align:right">${(p * 100).toFixed(0)}%</span></div>`).join('');
        const verdict = d.correct
          ? `<div class="alert ok small" style="margin-top:.4rem">Model picked <b>${d.model_pick}</b> -- correct.</div>`
          : `<div class="alert warn small" style="margin-top:.4rem">Model picked <b>${d.model_pick}</b>, but the correct value is <b>${d.value}</b>. Placing ${d.value} so solving keeps moving.</div>`;
        body += rows + verdict;
        body += `<details class="sub" style="margin-top:.5rem"><summary class="small muted">What the model was asked</summary><pre style="white-space:pre-wrap">${esc(d.context)}</pre></details>`;
      } else if (d.note) {
        body += `<div class="hint" style="margin-top:.4rem">${esc(d.note)} Placing the correct value (<b>${d.value}</b>) directly.</div>`;
      }
    }
    $('#why').innerHTML = `<div class="row">${badge}${pos}</div>${body}`;
  }

  async function nextStep() {
    if (busy || !grid) return;
    busy = true;
    const btn = $('#next-step');
    setBusy(btn, true, 'Thinking...');
    try {
      const d = await api('/api/sudoku/step', {method: 'POST',
        json: {grid, solution, backend: $('#backend').value}});
      if (d.done) {
        $('#why').innerHTML = '<div class="alert ok">Solved.</div>';
        $('#next-step').disabled = true; $('#auto-play').disabled = true;
        clearInterval(autoTimer); autoTimer = null; $('#auto-play').textContent = 'Auto-play';
        return;
      }
      grid[d.cell] = d.value;
      markPlaced(d.cell, d.method, d.method === 'classifier' && d.correct === false);
      updateProgress();
      renderWhy(d);
    } catch (e) {
      showErr('#why', e);
      clearInterval(autoTimer); autoTimer = null; $('#auto-play').textContent = 'Auto-play';
    } finally { setBusy(btn, false); busy = false; }
  }

  $('#new-board').addEventListener('click', newBoard);
  $('#next-step').addEventListener('click', nextStep);
  $('#auto-play').addEventListener('click', () => {
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; $('#auto-play').textContent = 'Auto-play'; return; }
    $('#auto-play').textContent = 'Stop auto-play';
    autoTimer = setInterval(nextStep, 550);   // nextStep's own `busy` guard skips a tick if the last call hasn't returned
  });

  newBoard();
})();
