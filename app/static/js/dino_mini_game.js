/* dino_mini_game.js — small same-origin canvas clone used to test the decision engine without
   touching chrome://dino or its CSP. Physics is fetched from dino_reflex_rules.json's
   "physics" block (not hardcoded here) so the reflex thresholds mean the same thing in the
   clone as they would in the real game. Exposes:
     window.DinoMini = {start(canvasEl), getState(), applyAction(action), reset()} */
(() => {
  const GROUND_Y = 120, PLAYER_X = 40, PLAYER_W = 20, DUCK_H = 12, STAND_H = 24;
  let duckTimeout = null;
  let ctx, canvas, raf, physics = null;
  let player, obstacles, speed, distance, crashed, ducking;
  // Reaction-window instrumentation: spawnedAt/arrivedAt live on each obstacle object (see
  // spawnMaybe() and step()). lastWindowMs is the most recently resolved (spawn -> arrival)
  // window in ms, exposed via getState() so decide.html can show it without polling internals.
  let lastWindowMs = null;
  const MAX_REACTION_LOG = 30;
  let reactionLog = [];
  let reactionCallback = null;

  async function loadPhysics() {
    if (physics) return physics;
    const res = await fetch('/static/data/dino_reflex_rules.json');
    physics = (await res.json()).physics;
    return physics;
  }

  function reset() {
    player = { y: GROUND_Y - STAND_H, vy: 0, jumping: false };
    obstacles = [];
    speed = physics.SPEED;
    distance = 0;
    crashed = false;
    ducking = false;
    lastWindowMs = null;
  }

  function spawnMaybe() {
    // if (obstacles.length && obstacles[obstacles.length - 1].x > canvas.width - 220) return;
    if (obstacles.length && (canvas.width - obstacles[obstacles.length - 1].x) < 250) return;
    if (Math.random() > 0.02) return;
    const flying = Math.random() < 0.35;
    // y is chosen relative to the duck hitbox, not just "somewhere above ground": with
    // STAND_H=24, DUCK_H=12, GROUND_Y=120, the duck box top is always GROUND_Y - DUCK_H = 108.
    // A duck-clearable obstacle's bottom edge (y + h) must sit above that line. y = GROUND_Y-34
    // gives bottom = 100 (8px clearance) while still overlapping the standing box (96-120), so
    // standing still collides but ducking clears it. The previous y = GROUND_Y-20 gave bottom =
    // 114, which is BELOW 108 -- ducking could never clear it, guaranteeing a crash on the
    // first pterodactyl (~35% of spawns), which is what was happening within a couple of seconds.
    // spawnedAt = the earliest moment this obstacle is knowable (it spawns already in-bounds,
    // so "spawned" and "first seen" are the same instant here). arrivedAt is filled in by
    // step() the first frame the obstacle's x-range reaches the player -- the deadline by
    // which a successful avoidance action must already have been applied.
    obstacles.push(flying
      ? { x: canvas.width, type: 'PTERODACTYL_MID', w: 18, h: 14, y: GROUND_Y - 34, spawnedAt: performance.now(), arrivedAt: null }
      : { x: canvas.width, type: 'CACTUS_SMALL', w: 14, h: 24, y: GROUND_Y - 24, spawnedAt: performance.now(), arrivedAt: null });
  }

  function step() {
    if (crashed) return;
    speed = Math.min(speed + physics.ACCELERATION, physics.MAX_SPEED);
    distance += speed;
    if (player.jumping) {
      player.vy += physics.GRAVITY;
      player.y += player.vy;
      if (player.y >= GROUND_Y - STAND_H) { player.y = GROUND_Y - STAND_H; player.jumping = false; player.vy = 0; }
    }
    spawnMaybe();
    obstacles.forEach(o => o.x -= speed);
    obstacles = obstacles.filter(o => o.x + o.w > 0);
    const h = ducking ? DUCK_H : STAND_H;
    const pTop = player.y + (STAND_H - h);
    for (const o of obstacles) {
      const overlapX = PLAYER_X + PLAYER_W > o.x && PLAYER_X < o.x + o.w;
      if (overlapX && o.arrivedAt == null) {
        // First frame the obstacle's x-range reaches the player: the window for successfully
        // avoiding it (however it's resolved this frame) is now fixed, win or lose.
        o.arrivedAt = performance.now();
        lastWindowMs = o.arrivedAt - o.spawnedAt;
        const entry = { ms: Math.round(lastWindowMs), type: o.type };
        reactionLog.unshift(entry);
        if (reactionLog.length > MAX_REACTION_LOG) reactionLog.length = MAX_REACTION_LOG;
        if (reactionCallback) reactionCallback(entry);
      }
      const overlapY = pTop + h > o.y && pTop < o.y + o.h;
      if (overlapX && overlapY) crashed = true;
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ccc'; ctx.fillRect(0, GROUND_Y, canvas.width, 2);
    const h = ducking ? DUCK_H : STAND_H;
    ctx.fillStyle = crashed ? '#c33' : '#333';
    ctx.fillRect(PLAYER_X, player.y + (STAND_H - h), PLAYER_W, h);
    ctx.fillStyle = '#2a6';
    obstacles.forEach(o => ctx.fillRect(o.x, o.y, o.w, o.h));
    ctx.fillStyle = '#333'; ctx.font = '12px monospace';
    ctx.fillText(`score ${Math.floor(distance)}  speed ${speed.toFixed(1)}${crashed ? '  CRASHED (auto-restarts)' : ''}`, 8, 16);
  }

  function loop() { step(); draw(); raf = requestAnimationFrame(loop); }

  async function start(canvasEl) {
    canvas = canvasEl; ctx = canvas.getContext('2d');
    await loadPhysics();
    reset();
    if (raf) cancelAnimationFrame(raf);
    loop();
  }

  // obstacle_x is distance from the player's leading edge, not raw canvas x -- matches what
  // ReflexDecisionBackend.decide() / dino_adapter.js's decide() expect for `dist`.
  function getState__OLD() {
    const next = obstacles[0];
    if (!next) return { obstacle_type: null };
    return {
      obstacle_type: next.type, obstacle_x: Math.max(next.x - (PLAYER_X + PLAYER_W), 0),
      obstacle_y: next.y, speed, crashed
    };
  }
  function getState() {
    if (!obstacles) return { obstacle_type: null, speed: 0, crashed: false, reaction_window_ms: lastWindowMs };
    // Find the first obstacle whose trailing edge has not yet passed the player
    const next = obstacles.find(o => o.x + o.w > PLAYER_X);
    // reaction_window_ms is sticky (last resolved obstacle) so the UI has something to show
    // in the gap between one obstacle arriving and the next one spawning.
    const base = { speed, crashed, reaction_window_ms: lastWindowMs };
    if (!next) return { ...base, obstacle_type: null };
    return {
      ...base,
      obstacle_type: next.type,
      obstacle_x: Math.max(next.x - (PLAYER_X + PLAYER_W), 0),
      obstacle_y: next.y,
      // Live elapsed time since this obstacle first existed -- ticks up every call until
      // arrivedAt is set, at which point reaction_window_ms above reflects the same obstacle.
      obstacle_seen_ms: performance.now() - next.spawnedAt
    };
  }

  function applyAction(action) {
    if (crashed) { reset(); return; }
    if (action === 'Jump' && !player.jumping) { player.jumping = true; player.vy = -physics.INITIAL_JUMP_VELOCITY; }
    else if (action === 'Duck') {
      ducking = true;
      // setTimeout(() => { ducking = false; }, 300); }
      if (duckTimeout) clearTimeout(duckTimeout);
      duckTimeout = setTimeout(() => { ducking = false; }, 300);
    }
  }

  window.DinoMini = {
    start, getState, applyAction, reset: () => physics && reset(),
    onReaction: (cb) => { reactionCallback = cb; }
  };
})();
