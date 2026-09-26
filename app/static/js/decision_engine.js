/* decision_engine.js — reusable real-time decision engine (generalizes live_sentiment.js's
   hybrid local/remote pattern from text scoring to per-tick game-action selection).

   Two speeds, same split as the sentiment demo:
     - LOCAL reflex: every animation frame, synchronous, zero network -- this is what actually
       drives the game. Each mode supplies {reflex: {readState(), decide(state, calibration)},
       applyAction(action, result), actions}. decide() must be a line-for-line port of the
       matching ReflexDecisionBackend.decide() in app/engine/backends.py so local play and a
       server review never disagree -- same guarantee the project already gives the sentiment
       lexicon between browser and server.
     - REMOTE advisory: throttled, cancellable, POSTs to /api/decide/<mode>. Never blocks a
       tick. Called only when the reflex flags ambiguous:true, or on a slow periodic cadence.
       Whichever System-1 backend the server has configured for that mode (laya / causal_lm /
       keyword) answers it -- swap models server-side (DEFAULT_PROFILES[<mode>].backend, or
       per-request); this file never changes because of that.

   Adding a new mode (Mario, a different runner, a card game's quick-play decisions...) needs
   no engine change: register a new {reflex, applyAction, actions} adapter and a matching
   backend.py ReflexDecisionBackend(mode=...) + reflex-rules JSON + config.py profile. */
(() => {
  const modes = new Map();
  const state = {running: false, mode: null, raf: null, advisoryTimer: null,
                 lastAdvisory: 0, inFlight: null, seq: 0, calibration: {}};

  function registerMode(name, adapter) { modes.set(name, adapter); }

  function start(name, {advisoryMs = 800, minAdvisoryGapMs = 150, onDecision} = {}) {
    const m = modes.get(name);
    if (!m) throw new Error(`decision_engine: unknown mode '${name}' -- registerMode() first`);
    stop();
    state.running = true; state.mode = name; state.calibration = {};

    const tick = () => {
      if (!state.running) return;
      const s = m.reflex.readState();
      const result = m.reflex.decide(s, state.calibration);
      if (result.action) m.applyAction(result.action, result);
      if (onDecision) onDecision(result, 'reflex');
      if (result.ambiguous) askSystem1(name, m, s, result, onDecision, minAdvisoryGapMs);
      state.raf = requestAnimationFrame(tick);
    };
    state.raf = requestAnimationFrame(tick);

    // Periodic advisory even without an ambiguous call, so a slowly-drifting reflex still gets
    // occasional System-1 grounding (see calibrate()) rather than only correcting at the edge.
    state.advisoryTimer = setInterval(() => {
      const s = m.reflex.readState();
      if (s.obstacle_type) askSystem1(name, m, s, null, onDecision, 0);
    }, advisoryMs);
  }

  function stop() {
    state.running = false;
    if (state.raf) cancelAnimationFrame(state.raf);
    if (state.advisoryTimer) clearInterval(state.advisoryTimer);
    if (state.inFlight) state.inFlight.abort();
  }

  async function askSystem1(name, m, gameState, reflexResult, onDecision, minGapMs) {
    const now = performance.now();
    if (now - state.lastAdvisory < minGapMs) return;      // don't spam the server on rapid ambiguity
    state.lastAdvisory = now;
    if (state.inFlight) state.inFlight.abort();
    state.inFlight = new AbortController();
    const my = ++state.seq;
    try {
      const res = await fetch(`/api/decide/${name}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({state: gameState, force_system1: !!reflexResult}),
        signal: state.inFlight.signal,
      });
      const d = await res.json();
      if (my !== state.seq) return;                        // superseded by a newer call
      if (onDecision) onDecision(d, 'system1');
      calibrate(d);
    } catch (e) { /* aborted, offline, or blocked by the host page's CSP: reflex already acted */ }
  }

  // Nudge the LOCAL reflex threshold toward what the System-1 model picked when they disagree,
  // instead of only logging the disagreement -- a tiny online calibration loop. Deliberately
  // client-local and ephemeral only: the server stays stateless per request (same "no Job, no
  // DB" discipline as live.py), so this never touches the shared rules JSON itself.
  function calibrate(d) {
    if (d.source !== 'system1' || !d.reflex || d.action === d.reflex.action) return;
    const delta = d.action === 'Jump' && d.reflex.action === 'Run' ? +0.03
                : d.action === 'Run' && d.reflex.action === 'Jump' ? -0.03 : 0;
    if (!delta) return;
    const cur = state.calibration.lead_multiplier ?? 1.15;
    state.calibration.lead_multiplier = Math.max(0.9, Math.min(1.6, cur + delta));
  }

  window.DecisionEngine = {registerMode, start, stop, state};
})();
