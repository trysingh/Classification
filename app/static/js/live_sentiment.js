/* live_sentiment.js - dynamic hybrid: score locally in the browser once it's confirmed fast
   enough on this device, otherwise (or if it later slows down) fall back to the server. Local
   and remote both implement the exact same algorithm against the exact same lexicon file, so
   which one answers a given keystroke is invisible in the result - only latency differs. */
(() => {
  const {$, esc} = App;
  const cfg = window.LIVE_SENTIMENT;
  const FALLBACK_WINDOW = 3;

  let lexicon = null;             // {words, negatorSet, intensifiers, meta} once fetched
  let mode = 'remote';            // current effective mode: 'remote' | 'local'
  let forced = 'auto';            // user override from the Mode selector: 'auto' | 'local' | 'remote'
  let consecutiveSlow = 0;
  let debounceTimer = null;
  let inFlight = null;            // AbortController for the current remote request
  let seq = 0;                    // guards a slow, now-stale remote response from overwriting a newer result

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const tokenize = t => (t.toLowerCase().match(/[a-z']+/g) || []);
  const sentClass = label => label === 'Positive' ? 'pos' : label === 'Negative' ? 'neg' : 'neu';

  /* ---- identical algorithm to LexiconSentimentBackend.score() in app/engine/backends.py ---- */
  function scoreLocal(text) {
    const tokens = tokenize(text), matches = [];
    const window = (lexicon.meta && lexicon.meta.negation_window) || FALLBACK_WINDOW;
    let total = 0;
    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i], base = lexicon.words[tok];
      if (base === undefined) continue;
      let mult = 1, negate = false;
      for (let back = 1; back <= window; back++) {
        const j = i - back;
        if (j < 0) break;
        const t2 = tokens[j];
        if (lexicon.negatorSet.has(t2)) negate = true;
        if (lexicon.intensifiers[t2] !== undefined) mult = lexicon.intensifiers[t2];
      }
      const val = +(base * mult * (negate ? -1 : 1)).toFixed(3);
      matches.push({word: tok, base, multiplier: mult, negated: negate, contribution: val});
      total += val;
    }
    const avg = matches.length ? +(total / matches.length).toFixed(4) : 0;
    return {avg, matches};
  }
  const labelFor = avg => avg > cfg.neutralBand ? 'Positive' : avg < -cfg.neutralBand ? 'Negative' : 'Neutral';

  /* ---- capability probe: warm the lexicon off the critical path, promote to local if it's fast ---- */
  const idle = fn => (window.requestIdleCallback || (f => setTimeout(f, 200)))(fn, {timeout: 2000});

  function probe() {
    const mem = navigator.deviceMemory;           // undefined on Safari/Firefox: unknown, try anyway
    if (mem !== undefined && mem < 1) { setBadge('remote', 'low-memory device'); return; }
    idle(async () => {
      try {
        const t0 = performance.now();
        const res = await fetch(cfg.lexiconUrl, {cache: 'force-cache'});
        const data = await res.json();
        data.negatorSet = new Set(data.negators);
        lexicon = data;
        const loadMs = performance.now() - t0;
        const t1 = performance.now();
        scoreLocal('a short warmup sentence used only to time the scorer');
        const scoreMs = performance.now() - t1;
        if (scoreMs < 5) { mode = 'local'; setBadge('local', `lexicon loaded in ${loadMs.toFixed(0)}ms`); }
        else setBadge('remote', 'local scoring too slow on this device');
      } catch (e) { setBadge('remote', 'lexicon unavailable, using the server'); }
    });
  }

  function setBadge(m, note) {
    const b = $('#engine-badge');
    b.textContent = (m === 'local' ? '\u26a1 Local' : '\u2601 Server') + (note ? ` \u2014 ${note}` : '');
    b.className = `badge ${m === 'local' ? 'b-ok' : 'b-info'}`;
  }

  /* ---- render whatever produced the result, local or remote ---- */
  function paint(label, avg, matches, ms, m) {
    const pill = $('#pill');
    pill.textContent = label;
    pill.className = `badge pill-${sentClass(label)}`;
    $('#score-bar-wrap').className = `bar sent-${sentClass(label)}`;
    $('#score-bar').style.width = (50 + clamp(avg, -3, 3) / 3 * 50) + '%';
    $('#latency').textContent = `${m === 'local' ? '\u26a1' : '\u2601'} ${ms.toFixed(1)}ms`;
    // The badge reflects whichever path actually answered THIS result, not just the initial
    // probe outcome, so forcing a mode is visibly honest rather than stuck on the first reading.
    setBadge(m, forced !== 'auto' ? 'forced by you' : m === 'local' ? 'client-side' : 'server round trip');
    $('#matches').innerHTML = matches.length
      ? matches.slice(0, 12).map(mm => `<span class="chip" title="base ${mm.base}${mm.multiplier !== 1 ? ` \u00d7 ${mm.multiplier}` : ''}${mm.negated ? ', negated' : ''}">${esc(mm.word)} <b class="num">${mm.contribution > 0 ? '+' : ''}${mm.contribution}</b></span>`).join('')
      : '<span class="hint">No sentiment words matched yet.</span>';
  }

  /* ---- input handling: local gets a tiny debounce (smoothing only); remote gets a real one + cancellation ---- */
  function onInput() {
    clearTimeout(debounceTimer);
    const effective = forced === 'auto' ? mode : forced;
    debounceTimer = setTimeout(effective === 'local' && lexicon ? runLocal : runRemote, effective === 'local' && lexicon ? 60 : 280);
  }

  function runLocal() {
    const t0 = performance.now();
    const {avg, matches} = scoreLocal($('#text').value);
    const ms = performance.now() - t0;
    paint(labelFor(avg), avg, matches, ms, 'local');
    if (forced === 'auto') {                      // auto-downgrade if this device starts struggling mid-session
      consecutiveSlow = ms > 8 ? consecutiveSlow + 1 : 0;
      if (consecutiveSlow >= 3) { mode = 'remote'; consecutiveSlow = 0; setBadge('remote', 'local scoring slowed down'); }
    }
  }

  async function runRemote() {
    const text = $('#text').value, my = ++seq;
    if (inFlight) inFlight.abort();
    inFlight = new AbortController();
    const t0 = performance.now();
    try {
      const res = await fetch('/api/live/sentiment', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text}), signal: inFlight.signal});
      const d = await res.json();
      if (my !== seq) return;                      // a newer keystroke already started another request
      paint(d.label, d.score, d.matches, performance.now() - t0, 'server');
    } catch (e) { /* aborted (superseded) or offline: the next keystroke retries */ }
  }

  $('#mode-select').addEventListener('change', e => { forced = e.target.value; onInput(); });
  $('#text').addEventListener('input', onInput);
  probe();
})();
