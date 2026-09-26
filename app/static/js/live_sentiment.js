/* live_sentiment.js - dynamic hybrid: score locally in the browser once it's confirmed fast
   enough on this device, otherwise (or if it later slows down) fall back to the server. Local
   and remote both implement the exact same algorithm against the exact same lexicon file, so
   which one answers a given keystroke is invisible in the result - only latency differs.

   v2: mirrors LexiconSentimentBackend in app/engine/backends.py line-for-line (phrases, clause
   splitting for negation scope + contrast weighting, diminishers, emojis, capped CAPS/punctuation
   emphasis, mixed-signal flag). Keep the two in lockstep by hand if either changes. */
(() => {
  const {$, esc} = App;
  const cfg = window.LIVE_SENTIMENT;
  const FALLBACK_WINDOW = 3;

  let lexicon = null;             // {words, phrases, negatorSet, intensifiers, diminishers, emojis, meta} once fetched
  let mode = 'remote';            // current effective mode: 'remote' | 'local'
  let forced = 'auto';            // user override from the Mode selector: 'auto' | 'local' | 'remote'
  let consecutiveSlow = 0;
  let debounceTimer = null;
  let inFlight = null;            // AbortController for the current remote request
  let seq = 0;                    // guards a slow, now-stale remote response from overwriting a newer result

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const tokenize = t => (t.toLowerCase().match(/[a-z']+/g) || []);
  const sentClass = label => label === 'Positive' ? 'pos' : label === 'Negative' ? 'neg' : 'neu';

  /* Same single-pass suffix strip as _stem() in app/engine/backends.py: "crashing" -> "crash" so
     it hits the same lexicon entry without listing every inflection by hand. */
  function stemWord(w) {
    for (const suf of ['ing', 'ed', 'es', 's', 'e']) {
      if (w.endsWith(suf) && w.length - suf.length >= 3) return w.slice(0, -suf.length);
    }
    return w;
  }

  function emphasisMultiplier(text) {
    const overallMax = Math.max((lexicon.meta && lexicon.meta.cap_boost_max) || 1.3,
                                 (lexicon.meta && lexicon.meta.punct_boost_max) || 1.3) - 1;
    const wordsAll = text.split(/\s+/).filter(Boolean);
    const caps = (text.match(/\b[A-Z]{3,}\b/g) || []).length;
    const capsRatio = wordsAll.length ? caps / wordsAll.length : 0;
    const capsBoost = capsRatio * 0.6;
    const bangs = text.trim().match(/[!?]{1,}$/);
    const punctBoost = bangs ? 0.08 * bangs[0].length : 0;
    return +(1 + Math.min(capsBoost + punctBoost, overallMax)).toFixed(4);
  }

  function emojiMatches(text) {
    const out = [];
    for (const [emo, base] of Object.entries(lexicon.emojis || {})) {
      const count = text.split(emo).length - 1;
      if (count > 0) {
        const mult = Math.min(1 + 0.2 * (count - 1), 1.5);
        out.push({type: 'emoji', word: emo, base, multiplier: +mult.toFixed(3), negated: false,
                  contribution: +(base * mult).toFixed(3), weight: 1.0});
      }
    }
    return out;
  }

  function matchPhrases(tokens) {
    const consumed = new Set(), matches = [];
    for (const [words, score] of lexicon.phrasesSorted) {       // longest phrases first
      const n = words.length;
      let i = 0;
      while (i <= tokens.length - n) {
        let free = true;
        for (let k = i; k < i + n; k++) if (consumed.has(k)) { free = false; break; }
        if (free && words.every((w, k) => tokens[i + k] === w)) {
          matches.push({type: 'phrase', word: words.join(' '), base: score, multiplier: 1, negated: false,
                        contribution: +score.toFixed(3)});
          for (let k = i; k < i + n; k++) consumed.add(k);
          i += n;
        } else i += 1;
      }
    }
    return {matches, consumed};
  }

  function scoreTokens(tokens, weight) {
    const window = (lexicon.meta && lexicon.meta.negation_window) || FALLBACK_WINDOW;
    const {matches: phraseMatches, consumed} = matchPhrases(tokens);
    phraseMatches.forEach(m => { m.weight = weight; });
    const wordMatches = [];
    for (let i = 0; i < tokens.length; i++) {
      if (consumed.has(i)) continue;
      const tok = tokens[i];
      let base = lexicon.words[tok];
      if (base === undefined) base = lexicon.stemIndex[stemWord(tok)];
      if (base === undefined) continue;
      let mult = 1, negate = false;
      for (let back = 1; back <= window; back++) {
        const j = i - back;
        if (j < 0) break;
        const t2 = tokens[j];
        if (lexicon.negatorSet.has(t2)) negate = true;
        if (lexicon.intensifiers[t2] !== undefined) mult = lexicon.intensifiers[t2];
        else if (lexicon.diminishers[t2] !== undefined) mult = lexicon.diminishers[t2];
      }
      const val = +(base * mult * (negate ? -1 : 1)).toFixed(3);
      wordMatches.push({type: 'word', word: tok, base, multiplier: mult, negated: negate, contribution: val, weight});
    }
    return phraseMatches.concat(wordMatches);
  }

  function scoreSentence(sentence) {
    const tokens = tokenize(sentence);
    const breakerIdx = tokens.findIndex(t => lexicon.scopeBreakers.has(t));
    if (breakerIdx === -1) return scoreTokens(tokens, 1.0);
    // The clause AFTER a contrast word usually carries the speaker's real point ("great UI but
    // slow app" leans negative, not a wash), so it's weighted up; before, weighted down.
    return scoreTokens(tokens.slice(0, breakerIdx), 0.6).concat(scoreTokens(tokens.slice(breakerIdx + 1), 1.6));
  }

  /* identical algorithm to LexiconSentimentBackend.score() */
  function scoreLocal(text) {
    let matches = [];
    for (const sentence of text.split(/[.!?]+/)) {
      if (sentence.trim()) matches = matches.concat(scoreSentence(sentence));
    }
    matches = matches.concat(emojiMatches(text));
    if (!matches.length) return {avg: 0, matches: []};
    const weightTotal = matches.reduce((s, m) => s + m.weight, 0) || 1;
    const rawAvg = matches.reduce((s, m) => s + m.contribution * m.weight, 0) / weightTotal;
    const avg = +(rawAvg * emphasisMultiplier(text)).toFixed(4);
    return {avg, matches};
  }

  function signalSummary(matches) {
    const pos = matches.filter(m => m.contribution > 0).reduce((s, m) => s + m.contribution * m.weight, 0);
    const neg = matches.filter(m => m.contribution < 0).reduce((s, m) => s + m.contribution * m.weight, 0);
    return {positive_score: +pos.toFixed(4), negative_score: +neg.toFixed(4), mixed: pos > 0.5 && neg < -0.5};
  }

  const labelFor = avg => avg > cfg.neutralBand ? 'Positive' : avg < -cfg.neutralBand ? 'Negative' : 'Neutral';
  function confFor(avg, matches) {
    const magnitude = Math.min(Math.abs(avg) / ((cfg.neutralBand * 4) || 1e-9), 1);
    if (!matches.length) return +magnitude.toFixed(4);
    const volume = Math.min(matches.length / 4, 1);
    const agreement = signalSummary(matches).mixed ? 0.5 : 1;
    return +Math.min(0.6 * magnitude + 0.25 * volume * agreement + 0.15 * agreement, 1).toFixed(4);
  }

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
        data.diminishers = data.diminishers || {};
        data.emojis = data.emojis || {};
        data.scopeBreakers = new Set((data.meta && data.meta.scope_breakers) || []);
        data.phrasesSorted = Object.entries(data.phrases || {})
          .map(([p, s]) => [p.split(' '), s]).sort((a, b) => b[0].length - a[0].length);
        data.stemIndex = {};
        for (const [w, s] of Object.entries(data.words)) {
          const st = stemWord(w);
          if (data.stemIndex[st] === undefined) data.stemIndex[st] = s;
        }
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
  function paint(label, avg, matches, ms, m, signals) {
    const pill = $('#pill');
    pill.textContent = label;
    pill.className = `badge pill-${sentClass(label)}`;
    $('#mixed-badge').hidden = !(signals && signals.mixed);
    $('#score-bar-wrap').className = `bar sent-${sentClass(label)}`;
    $('#score-bar').style.width = (50 + clamp(avg, -3, 3) / 3 * 50) + '%';
    $('#latency').textContent = `${m === 'local' ? '\u26a1' : '\u2601'} ${ms.toFixed(1)}ms`;
    setBadge(m, forced !== 'auto' ? 'forced by you' : m === 'local' ? 'client-side' : 'server round trip');
    if (signals) {
      $('#signal-split').textContent = `+${signals.positive_score.toFixed(2)} positive \u00b7 ${signals.negative_score.toFixed(2)} negative`;
    }
    $('#matches').innerHTML = matches.length
      ? matches.slice(0, 12).map(mm => {
          const tag = mm.type === 'phrase' ? '<b class="tag">phrase</b>' : mm.type === 'emoji' ? '<b class="tag">emoji</b>' : '';
          const title = mm.type === 'word' ? `base ${mm.base}${mm.multiplier !== 1 ? ` \u00d7 ${mm.multiplier}` : ''}${mm.negated ? ', negated' : ''}` : `base ${mm.base}`;
          return `<span class="chip" title="${title}">${tag}${esc(mm.word)} <b class="num">${mm.contribution > 0 ? '+' : ''}${mm.contribution}</b></span>`;
        }).join('')
      : '<span class="hint">No sentiment words matched yet.</span>';
  }

  /* ---- input handling: local gets a tiny debounce (smoothing only); remote gets a real one + cancellation ---- */
  function onInput() {
    clearTimeout(debounceTimer);
    const effective = forced === 'auto' ? mode : forced;
    debounceTimer = setTimeout(effective === 'local' && lexicon ? runLocal : runRemote, effective === 'local' && lexicon ? 60 : 280);
  }

  function runLocal() {
    const text = $('#text').value;
    const t0 = performance.now();
    const {avg, matches} = scoreLocal(text);
    const ms = performance.now() - t0;
    paint(labelFor(avg), avg, matches, ms, 'local', signalSummary(matches));
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
      paint(d.label, d.score, d.matches, performance.now() - t0, 'server',
            {positive_score: d.positive_score, negative_score: d.negative_score, mixed: d.mixed});
    } catch (e) { /* aborted (superseded) or offline: the next keystroke retries */ }
  }

  $('#mode-select').addEventListener('change', e => { forced = e.target.value; onInput(); });
  $('#text').addEventListener('input', onInput);
  probe();
})();
