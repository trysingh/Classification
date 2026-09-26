/* dino_adapter.js — mode adapter for the "dino" quick-decision profile (see app/config.py
   DEFAULT_PROFILES["dino"] and app/engine/backends.py ReflexDecisionBackend). Loads the SAME
   rules JSON the Python backend loads and ports ReflexDecisionBackend.decide() line-for-line,
   so a local decision here and a /api/decide/dino decision from the server are never
   contradictory -- only which one actually pressed the key differs.

   Two readState() sources behind one {reflex, applyAction, actions} interface:
     - createMiniAdapter() -> drives dino_mini_game.js, a same-origin canvas clone. No CSP or
       extension concerns; this is what /decide/dino uses by default.
     - createLiveAdapter() -> drives the REAL chrome://dino via window.Runner.instance_
       (Chromium's own offline.js exposes Runner.instance_.horizon.obstacles / .tRex /
       .currentSpeed). Paste this file + decision_engine.js into the chrome://dino devtools
       console to try it against the real page.
       CAVEAT: chrome:// pages restrict outbound network requests from page scripts, so
       fetch('/api/decide/dino') from THIS context may be blocked outright -- if so, the
       reflex path below still plays the game unaided, it just never gets a System-1 opinion
       on the calls it flags ambiguous. If you want live escalation against the real page
       rather than the in-app clone, route that one fetch through a small MV3 extension's
       background service worker (not subject to the page's CSP) via chrome.runtime.sendMessage,
       rather than fighting the page's own CSP. */
(() => {
  let RULES = null;
  async function loadRules() {
    if (RULES) return RULES;
    const res = await fetch('/static/data/dino_reflex_rules.json');
    RULES = await res.json();
    return RULES;
  }

  /* Straight JS port of ReflexDecisionBackend.decide() in app/engine/backends.py. Keep the two
     in lockstep by hand -- same tradeoff the project already made for live_sentiment.js vs.
     LexiconSentimentBackend.score(), deliberately no code-gen between them. `calibration` is
     client-local only (see decision_engine.js's calibrate()); the server-side twin always uses
     the base rules file, so a server review is always the un-nudged baseline. */
  function decide(state, calibration = {}) {
    if (!state.obstacle_type) return {action: 'Run', confidence: 1, ambiguous: false, trigger_px: null};
    const p = RULES.physics, t = RULES.thresholds;
    const leadMult = calibration.lead_multiplier ?? t.lead_multiplier;
    const band = calibration.ambiguous_margin_px ?? t.ambiguous_margin_px;
    const airtimeFrames = p.INITIAL_JUMP_VELOCITY / (p.GRAVITY || 0.6);
    const speed = Math.max(state.speed ?? p.SPEED, 0.1);
    const triggerPx = speed * airtimeFrames * leadMult;
    const dist = state.obstacle_x ?? 9999;
    if (dist > triggerPx + band) return {action: 'Run', confidence: 1, ambiguous: false, trigger_px: triggerPx};
    const profile = RULES.obstacle_profiles[state.obstacle_type] || {action: 'Jump'};
    const margin = triggerPx - dist;
    const ambiguous = Math.abs(margin) <= band;
    const confidence = ambiguous ? 0.5 : Math.min(Math.abs(margin) / ((band * 2) || 1e-9), 1);
    return {action: profile.action, confidence, ambiguous, trigger_px: triggerPx};
  }

  // duck_y_threshold is the one Chromium constant the public source doesn't fix -- verify it
  // in your Chrome build: Runner.instance_.horizon.obstacles[0].yPos while a pterodactyl is on
  // screen, at low / mid / high flight, and adjust dino_reflex_rules.json accordingly.
  function classifyPterodactyl(obstacle, duckY) {
    return obstacle.yPos >= duckY ? 'PTERODACTYL_MID' : 'PTERODACTYL_HIGH';
  }

  window.createMiniAdapter = async () => {
    await loadRules();
    return {
      reflex: {readState: () => DinoMini.getState(), decide},
      applyAction: (action) => DinoMini.applyAction(action),
      actions: ['Jump', 'Duck', 'Run'],
    };
  };

  function dispatchKey(type, keyCode) {
    const evt = new KeyboardEvent(type, {bubbles: true, cancelable: true});
    Object.defineProperty(evt, 'keyCode', {get: () => keyCode});   // Chrome's own handler reads e.keyCode
    document.dispatchEvent(evt);
  }

  window.createLiveAdapter = async () => {
    await loadRules();
    return {
      reflex: {
        readState: () => {
          const r = window.Runner && window.Runner.instance_;
          if (!r || !r.playing || !r.horizon || !r.horizon.obstacles.length) return {obstacle_type: null};
          const o = r.horizon.obstacles[0];
          const rawType = (o.typeConfig && o.typeConfig.type) || 'CACTUS_SMALL';
          const type = rawType.startsWith('PTERODACTYL') ? classifyPterodactyl(o, RULES.thresholds.duck_y_threshold) : rawType;
          return {obstacle_type: type, obstacle_x: o.xPos, obstacle_y: o.yPos, speed: r.currentSpeed};
        },
        decide,
      },
      applyAction: (action) => {
        if (action === 'Jump') { dispatchKey('keydown', 38); dispatchKey('keyup', 38); }
        else if (action === 'Duck') { dispatchKey('keydown', 40); setTimeout(() => dispatchKey('keyup', 40), 300); }
      },
      actions: ['Jump', 'Duck', 'Run'],
    };
  };
})();
