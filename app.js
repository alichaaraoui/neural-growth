// V6 demo. Place points or draw a shape; the trained network grows it one step at a time
// (one step = iters_per_step simulator iterations), for as long as you let it. "Compare" grows
// the real simulation beside it from the same start.
//
// Rendering: each side keeps a signed distance field of its shape (negative inside). The network
// morphs smoothly between steps by blending the field of the last step into the field of the
// new one every animation frame, so its outline moves continuously like the simulator's curve.
// Fill = the field thresholded with a soft edge, coloured by age (magma, old -> new). Line =
// the field's zero contour (network, marching squares) or the simulator's real curve.
import * as S from './sim.js';

const ort = window.ort;
const MODEL_URL = 'model/v6.onnx';
const META_URL = 'model/model.json';    // {"label": "...", "iters_per_step": 16}
let ITERS_PER_STEP = 16;                // simulator iterations per network step
const BASE_STEP_MS = 420;               // on-screen time per network step at 1x
let stepMs = BASE_STEP_MS;
const MAX_STEPS = 200;                  // safety stop
const R = S.RES, N = R * R;

const $ = id => document.getElementById(id);
const inputCv = $('input'), ictx = inputCv.getContext('2d');
const ctxs = { model: $('model').getContext('2d'), sim: $('sim').getContext('2d') };

// ---- colour --------------------------------------------------------------------------

const MAGMA = ['#000004', '#1c1044', '#4f127b', '#812581', '#b5367a', '#e55064', '#fb8761', '#fec287', '#fcfdbf']
  .map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
// the same magma ramp in both themes; on light paper it stops at peach so the newest growth stays visible
const magmaLUT = (lo, hi) => Array.from({ length: 256 }, (_, i) => {
  const x = (lo + (hi - lo) * i / 255) * (MAGMA.length - 1), k = Math.min(MAGMA.length - 2, Math.floor(x)), t = x - k;
  return MAGMA[k].map((c, j) => Math.round(c + t * (MAGMA[k + 1][j] - c)));
});
const BUCKETS = 24;
const THEMES = {
  dark: { lut: magmaLUT(0.2, 1), edge: 'rgba(255,240,225,0.4)', bg: '#0f0f11' },
  light: { lut: magmaLUT(0.12, 0.84), edge: 'rgba(45,16,40,0.3)', bg: '#fbfaf9' },
};
for (const T of Object.values(THEMES))
  T.buckets = Array.from({ length: BUCKETS }, (_, b) => { const c = T.lut[Math.round(255 * b / (BUCKETS - 1))]; return `rgb(${c[0]},${c[1]},${c[2]})`; });
let theme = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
let LUT = THEMES[theme].lut, bucketColor = THEMES[theme].buckets;

// ---- state ---------------------------------------------------------------------------

let session = null, mode = 'points', view = 'fill', compare = false;
let points = [], stroke = null, start = null;
let state = 'input';       // input | growing | paused
let runId = 0;
let lanes = { model: null, sim: null };
let msAvg = 0, stepOnce = false, ended = false;   // ended: growth reached the edge or the step limit

// ---- the network ---------------------------------------------------------------------

// The network runs in a worker (infer.js) so the page stays smooth while it thinks.
// iPhone/iPad use WASM: Safari's WebGPU path is the one that crashed on phones.
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
let worker = null, jobs = new Map(), jobId = 0;

function loadModel() {
  worker = new Worker('infer.js');
  worker.onmessage = ({ data: m }) => {
    if (m.type === 'progress') { if (!session) setStatus(`loading model… ${Math.round(m.p * 100)}%`); return; }
    if (m.type === 'loaded') { onLoaded(m); return; }
    const job = jobs.get(m.id); if (!job) return;
    jobs.delete(m.id);
    if (m.type === 'result') { lastModelMs = m.ms; job.resolve(m.next); }
    else job.reject(new Error(m.msg));
  };
  worker.onerror = e => { for (const j of jobs.values()) j.reject(e); jobs.clear(); };
  worker.postMessage({ type: 'load', url: MODEL_URL, base: new URL('.', location.href).href, eps: isIOS || !navigator.gpu ? ['wasm'] : ['webgpu', 'wasm'] });
  fetch(META_URL, { cache: 'no-store' }).then(r => r.json()).then(meta => {
    if (meta.label) { $('modelLabel').textContent = meta.label.split(' ')[0]; $('modelLabel').title = meta.label; }
    if (meta.iters_per_step) ITERS_PER_STEP = meta.iters_per_step;
    $('itersWord').textContent = ITERS_PER_STEP;
  }).catch(() => {});
  $('itersWord').textContent = ITERS_PER_STEP;
}

function onLoaded(m) {
  session = m.ok ? { inputNames: m.inputs, ep: m.ep } : null;
  hasParams = !!session && session.inputNames.includes('params');
  $('charSec').hidden = !hasParams;
  syncParams();
  setStatus(session ? '' : 'the model failed to load');
  refreshButtons();
}

let queue = Promise.resolve();          // one network call at a time
let lastModelMs = 0;                    // time of the last network call alone
function modelStep(fill, prev) {
  const job = queue.then(() => new Promise((resolve, reject) => {
    const id = ++jobId;
    jobs.set(id, { resolve, reject });
    worker.postMessage({ type: 'run', id, R, fill: fill.slice(), prev: prev.slice(), params: hasParams ? Float32Array.from(pv) : null });
  }));
  queue = job.catch(() => {});
  return job;
}

// ---- one line: keep the largest piece and fill its holes ---------------------------------
// Applied after every network step (unless "raw network output" is on), so the pattern stays one
// closed line. The cleaned image is what gets fed back in, drawn, and aged.

let rawOutput = false;
const _lab = new Int32Array(N), _q = new Int32Array(N);
function oneLine(fill) {
  // 1. 8-connected pieces of filled pixels; keep the largest
  _lab.fill(0);
  let best = 0, bestN = 0, id = 0;
  for (let s0 = 0; s0 < N; s0++) {
    if (!fill[s0] || _lab[s0]) continue;
    id++; let h = 0, t = 0, n = 0; _q[t++] = s0; _lab[s0] = id;
    while (h < t) {
      const i = _q[h++], x = i % R, y = (i / R) | 0; n++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if ((dx || dy) && xx >= 0 && xx < R && yy >= 0 && yy < R) {
          const j = yy * R + xx;
          if (fill[j] && !_lab[j]) { _lab[j] = id; _q[t++] = j; }
        }
      }
    }
    if (n > bestN) { bestN = n; best = id; }
  }
  const out = new Float32Array(N);
  if (!best) return out;
  for (let i = 0; i < N; i++) out[i] = _lab[i] === best ? 1 : 0;
  // 2. fill holes: flood the background from the canvas border (4-connected); unreached background is a hole
  const seen = _lab; seen.fill(0);
  let h = 0, t = 0;
  for (let k = 0; k < R; k++) for (const i of [k, (R - 1) * R + k, k * R, k * R + R - 1])
    if (!out[i] && !seen[i]) { seen[i] = 1; _q[t++] = i; }
  while (h < t) {
    const i = _q[h++], x = i % R;
    for (const j of [x > 0 ? i - 1 : -1, x < R - 1 ? i + 1 : -1, i - R, i + R])
      if (j >= 0 && j < N && !out[j] && !seen[j]) { seen[j] = 1; _q[t++] = j; }
  }
  for (let i = 0; i < N; i++) if (!out[i] && !seen[i]) out[i] = 1;
  return out;
}

// ---- signed distance field -------------------------------------------------------------

const CLAMP = 14, SQ2 = Math.SQRT2;
const _din = new Float32Array(N), _dout = new Float32Array(N), _tmp = new Float32Array(N);

function chamfer(d) {
  for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
    const i = y * R + x; let v = d[i]; if (v === 0) continue;
    if (x > 0) v = Math.min(v, d[i - 1] + 1);
    if (y > 0) {
      v = Math.min(v, d[i - R] + 1);
      if (x > 0) v = Math.min(v, d[i - R - 1] + SQ2);
      if (x < R - 1) v = Math.min(v, d[i - R + 1] + SQ2);
    }
    d[i] = v;
  }
  for (let y = R - 1; y >= 0; y--) for (let x = R - 1; x >= 0; x--) {
    const i = y * R + x; let v = d[i]; if (v === 0) continue;
    if (x < R - 1) v = Math.min(v, d[i + 1] + 1);
    if (y < R - 1) {
      v = Math.min(v, d[i + R] + 1);
      if (x < R - 1) v = Math.min(v, d[i + R + 1] + SQ2);
      if (x > 0) v = Math.min(v, d[i + R - 1] + SQ2);
    }
    d[i] = v;
  }
}

// fill (0/1) -> smooth signed distance in pixels, negative inside, clamped to +/-CLAMP
function makeSDF(fill) {
  for (let i = 0; i < N; i++) { _din[i] = fill[i] ? 1e4 : 0; _dout[i] = fill[i] ? 0 : 1e4; }
  chamfer(_din); chamfer(_dout);
  for (let i = 0; i < N; i++) {
    const v = fill[i] ? 0.5 - _din[i] : _dout[i] - 0.5;
    _tmp[i] = v < -CLAMP ? -CLAMP : v > CLAMP ? CLAMP : v;
  }
  // 3x3 box blur: removes the pixel staircase so contours are smooth
  const out = new Float32Array(N);
  for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
    let s = 0, n = 0;
    for (let d = -1; d <= 1; d++) { const xx = x + d; if (xx >= 0 && xx < R) { s += _tmp[y * R + xx]; n++; } }
    _din[y * R + x] = s / n;
  }
  for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
    let s = 0, n = 0;
    for (let d = -1; d <= 1; d++) { const yy = y + d; if (yy >= 0 && yy < R) { s += _din[yy * R + x]; n++; } }
    out[y * R + x] = s / n;
  }
  return out;
}

// ---- a growing pattern: its image, the step each pixel was born, its distance field ----

function newLane(fill0) {
  const birth = new Int16Array(N).fill(-1), offSince = new Int16Array(N).fill(-1);
  for (let i = 0; i < N; i++) if (fill0[i]) birth[i] = 0;
  const sdf = makeSDF(fill0);
  const cv = document.createElement('canvas'); cv.width = cv.height = R;
  const g = cv.getContext('2d');
  return { fill: fill0, prev: fill0, birth, offSince, step: 0, sdf, sdfPrev: sdf,
           field: new Float32Array(N), cv, g, img: g.createImageData(R, R), curve: null };
}

// a pixel that blinks off for a moment at the growing edge keeps its age (no flicker)
function advanceAge(lane, fill, t) {
  const { birth, offSince } = lane;
  for (let i = 0; i < N; i++) {
    if (fill[i]) {
      if (birth[i] < 0 || (offSince[i] >= 0 && t - offSince[i] > 2)) birth[i] = t;
      offSince[i] = -1;
    } else if (birth[i] >= 0 && offSince[i] < 0) offSince[i] = t;
  }
}

function ageBucket(lane, px, py) {
  const x = Math.min(R - 1, Math.max(0, px | 0)), y = Math.min(R - 1, Math.max(0, py | 0));
  let b = lane.birth[y * R + x];
  if (b < 0) {   // just outside the old fill: look around, else it is the newest growth
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const v = lane.birth[Math.min(R - 1, Math.max(0, y + dy)) * R + Math.min(R - 1, Math.max(0, x + dx))];
      if (v > b) b = v;
    }
    if (b < 0) b = lane.step;
  }
  return Math.round((BUCKETS - 1) * Math.max(0, b) / Math.max(1, lane.step));
}

// the field blended between the last step and this one (a = 0..1)
function blend(lane, a) {
  const f = lane.field, p = lane.sdfPrev, q = lane.sdf;
  if (a >= 1 || p === q) return q;
  for (let i = 0; i < N; i++) f[i] = p[i] + a * (q[i] - p[i]);
  return f;
}

// marching squares at 0 -> line segments grouped by age bucket (edges: 0 top, 1 right, 2 bottom, 3 left)
const CASES = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]],
  [[2, 3]], [[2, 0]], [[0, 1], [2, 3]], [[2, 1]], [[1, 3]], [[1, 0]], [[0, 3]], []];
function traceField(lane, v) {
  const out = Array.from({ length: BUCKETS }, () => []);
  const lerp = (a, b) => a / (a - b);
  for (let y = 0; y < R - 1; y++) for (let x = 0; x < R - 1; x++) {
    const a = v[y * R + x], b = v[y * R + x + 1], c = v[(y + 1) * R + x + 1], d = v[(y + 1) * R + x];
    const k = (a < 0) | ((b < 0) << 1) | ((c < 0) << 2) | ((d < 0) << 3);
    if (k === 0 || k === 15) continue;
    const T = [x + lerp(a, b), y], Rt = [x + 1, y + lerp(b, c)], B = [x + lerp(d, c), y + 1], L = [x, y + lerp(a, d)];
    const E = [T, Rt, B, L];
    for (const [i, j] of CASES[k]) {
      const p = E[i], q = E[j];
      const bk = ageBucket(lane, (p[0] + q[0]) / 2 + .5, (p[1] + q[1]) / 2 + .5);
      out[bk].push(p[0] + .5, p[1] + .5, q[0] + .5, q[1] + .5);
    }
  }
  return out;
}

// the simulator's own curve -> segments grouped by age bucket
const toGrid = v => (v + S.WIN) * (R / (2 * S.WIN));
function traceCurve(lane, pts) {
  const out = Array.from({ length: BUCKETS }, () => []);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    const x1 = toGrid(p[0]), y1 = toGrid(p[1]), x2 = toGrid(q[0]), y2 = toGrid(q[1]);
    out[ageBucket(lane, (x1 + x2) / 2, (y1 + y2) / 2)].push(x1, y1, x2, y2);
  }
  return out;
}

// the field as an age-coloured image with a soft, anti-aliased edge
function paintFill(lane, v) {
  const d = lane.img.data, s = Math.max(1, lane.step), birth = lane.birth;
  for (let i = 0; i < N; i++) {
    const a = 0.5 - v[i];
    if (a <= 0) { d[i * 4 + 3] = 0; continue; }
    const b = birth[i] >= 0 ? birth[i] : s, col = LUT[Math.round(255 * b / s)];
    d[i * 4] = col[0]; d[i * 4 + 1] = col[1]; d[i * 4 + 2] = col[2]; d[i * 4 + 3] = a >= 1 ? 255 : a * 255;
  }
  lane.g.putImageData(lane.img, 0, 0);
  return lane.cv;
}

// ---- drawing -------------------------------------------------------------------------

function render(which, a = 1) {
  const lane = lanes[which], ctx = ctxs[which], W = ctx.canvas.width, k = W / R;
  ctx.clearRect(0, 0, W, W);
  if (!lane) return;
  const v = blend(lane, a);
  if (view === 'fill') {
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(paintFill(lane, v), 0, 0, W, W);
  }
  const lines = lane.curve ? traceCurve(lane, lane.curve) : traceField(lane, v);
  ctx.lineCap = ctx.lineJoin = 'round';
  ctx.lineWidth = view === 'fill' ? k * 0.45 : k * 0.85;
  for (let b = 0; b < BUCKETS; b++) {
    const L = lines[b]; if (!L.length) continue;
    ctx.strokeStyle = view === 'fill' ? THEMES[theme].edge : bucketColor[b];
    ctx.beginPath();
    for (let i = 0; i < L.length; i += 4) { ctx.moveTo(L[i] * k, L[i + 1] * k); ctx.lineTo(L[i + 2] * k, L[i + 3] * k); }
    ctx.stroke();
  }
}

function touchesEdge(fill, m = 3) {
  for (let i = 0; i < R; i++)
    for (let j = 0; j < m; j++)
      if (fill[j * R + i] || fill[(R - 1 - j) * R + i] || fill[i * R + j] || fill[i * R + R - 1 - j]) return true;
  return false;
}

// ---- readout -------------------------------------------------------------------------

function setReadout(step, fill) {
  $('stepNum').textContent = step;
  $('iters').textContent = step * ITERS_PER_STEP;
  $('ms').textContent = msAvg ? Math.round(msAvg) : '–';
  const r = fill ? reach(fill) : 0;
  $('reach').textContent = Math.round(100 * r) + '%';
  document.querySelectorAll('.plate').forEach(p => p.classList.toggle('near', r > .8));
}

// how far the shape reaches toward the canvas edge, 0..1
function reach(fill) {
  let m = 0; const c = (R - 1) / 2;
  for (let y = 0; y < R; y += 2) for (let x = 0; x < R; x += 2)
    if (fill[y * R + x]) { const d = Math.max(Math.abs(x - c), Math.abs(y - c)); if (d > m) m = d; }
  return Math.min(1, m / (c - 3));
}

// the canvas border pulses once when growth reaches it
function hitEdge() {
  document.querySelectorAll('.plate').forEach(p => { p.classList.remove('hit'); void p.offsetWidth; p.classList.add('hit'); });
}

// ---- growth loop ---------------------------------------------------------------------

const nextFrame = () => new Promise(r => requestAnimationFrame(r));

async function run() {
  const me = ++runId;
  state = 'growing'; refreshButtons();
  const M = lanes.model, Sm = lanes.sim;
  let pending = modelStep(M.fill, M.prev), msSum = 0, msN = 0;

  while (state === 'growing' && me === runId) {
    let raw;
    try { raw = await pending; }
    catch (e) { if (me === runId) { console.warn(e); stop('the network stopped, press grow to try again'); } return; }
    if (me !== runId || state !== 'growing') return;
    const next = rawOutput ? raw : oneLine(raw);   // one closed line unless the raw view is on
    msSum += lastModelMs; msN++; msAvg = msSum / msN;

    // network: new step -> morph from the old field to the new one
    M.prev = M.fill; M.fill = next; M.step++;
    advanceAge(M, next, M.step);
    M.sdfPrev = M.sdf; M.sdf = makeSDF(next);
    pending = modelStep(M.fill, M.prev);           // start the next step while this one animates

    // animate this step: morph the network, play the simulator's iterations
    const t0 = performance.now();
    let simDone = 0;
    for (;;) {
      const a = Math.min(1, (performance.now() - t0) / stepMs);
      if (Sm) {
        const want = Math.round(a * ITERS_PER_STEP);
        if (simDone < want) {
          while (simDone < want) { Sm.curve = Sm.gen.next().value; simDone++; Sm.step++; }
          Sm.fill = S.renderFill(Sm.curve);        // simulator age is counted in iterations
          advanceAge(Sm, Sm.fill, Sm.step);
          Sm.sdf = Sm.sdfPrev = makeSDF(Sm.fill);
          render('sim');
        }
      }
      render('model', a);
      if (a >= 1) break;
      await nextFrame();
      if (me !== runId) return;
    }

    setReadout(M.step, M.fill);
    if (stepOnce) { stepOnce = false; stop('paused after one step'); break; }
    if (touchesEdge(M.fill)) { hitEdge(); ended = true; stop('reached the edge of the canvas'); break; }
    if (M.step >= MAX_STEPS) { ended = true; stop(`stopped at ${MAX_STEPS} steps`); break; }
  }
}

function stop(msg) { state = 'paused'; setStatus(msg || 'paused'); refreshButtons(); }

function begin() {
  if (!session) return;
  if (state === 'growing') { stop(); return; }
  if (state === 'paused' && ended) {              // start over from the same shape
    const s = start, m = mode; reset(); mode = m;
    if (m === 'draw') stroke = s; else points = s.map(p => [...p]);
  } else if (state === 'paused') { setStatus(''); run(); return; }
  start = mode === 'draw' ? stroke : (points.length >= 3 ? S.orderPoints(points) : null);
  if (!start || start.length < 3) return;
  if (S.polygonArea(start) < 20) { setStatus('that shape is too thin, spread it out'); return; }
  const f0 = S.renderFill(start);
  lanes.model = newLane(f0);
  if (compare) {
    lanes.sim = newLane(f0);
    applySim();
    lanes.sim.gen = S.grow(start, 1);             // one iteration at a time, for smooth playback
    lanes.sim.curve = start;
  } else lanes.sim = null;
  clearInput(); showHint(false); setStatus('');
  render('model'); if (compare) render('sim');
  run();
}

function reset(msg = '') {
  runId++; state = 'input'; points = []; stroke = null; start = null; msAvg = 0; stepOnce = false; ended = false;
  lanes = { model: null, sim: null };
  setReadout(0, null);
  document.querySelectorAll('.plate').forEach(p => p.classList.remove('hit'));
  render('model'); render('sim');
  clearInput(); showHint(true); setStatus(msg); refreshButtons();
}

// ---- input: points or a drawn shape ---------------------------------------------------

const toWorld = e => {
  const r = inputCv.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * 2 * S.WIN - S.WIN, ((e.clientY - r.top) / r.height) * 2 * S.WIN - S.WIN];
};
const toCv = v => (v + S.WIN) / (2 * S.WIN) * inputCv.width;
function clearInput() { ictx.clearRect(0, 0, inputCv.width, inputCv.height); }

let drawing = false;
function drawInput() {
  clearInput();
  if (state !== 'input') return;             // start marks only before growth begins
  const shape = mode === 'draw' ? stroke : (points.length >= 3 ? S.orderPoints(points) : null);
  const ink = theme === 'light' ? '27,26,29' : '237,237,238';
  ictx.lineWidth = 2.5; ictx.strokeStyle = `rgb(${ink})`; ictx.lineJoin = ictx.lineCap = 'round';
  if (shape) {
    ictx.beginPath(); shape.forEach((p, i) => ictx[i ? 'lineTo' : 'moveTo'](toCv(p[0]), toCv(p[1])));
    if (mode !== 'draw' || !drawing) { ictx.closePath(); ictx.fillStyle = `rgba(${ink},0.06)`; ictx.fill(); }
    if (mode !== 'draw') ictx.setLineDash([10, 9]);
    ictx.stroke(); ictx.setLineDash([]);
  }
  if (mode !== 'draw') {
    for (const p of points) {
      ictx.fillStyle = `rgba(${ink},0.14)`; ictx.beginPath(); ictx.arc(toCv(p[0]), toCv(p[1]), 15, 0, 7); ictx.fill();
      ictx.fillStyle = `rgb(${ink})`; ictx.beginPath(); ictx.arc(toCv(p[0]), toCv(p[1]), 6, 0, 7); ictx.fill();
    }
  }
}

inputCv.addEventListener('pointerdown', e => {
  if (!session) return;
  if (state !== 'input') reset();
  showHint(false);
  if (mode !== 'draw') {
    if (uiMode !== 'points') { setUiMode('points'); points = []; }
    points.push(toWorld(e)); drawInput();
    setStatus(points.length < 3 ? `add ${3 - points.length} more point${points.length === 2 ? '' : 's'}` : 'press grow or space');
  } else {
    drawing = true; stroke = [toWorld(e)];
    try { inputCv.setPointerCapture(e.pointerId); } catch {}
    drawInput();
  }
  refreshButtons();
});
inputCv.addEventListener('pointermove', e => {
  if (!drawing) return;
  const p = toWorld(e), q = stroke[stroke.length - 1];
  if (Math.hypot(p[0] - q[0], p[1] - q[1]) > 0.4) { stroke.push(p); drawInput(); }
});
inputCv.addEventListener('pointerup', () => {
  if (!drawing) return;
  drawing = false;
  if (stroke.length < 5) { stroke = null; drawInput(); setStatus('draw a bigger shape'); refreshButtons(); return; }
  stroke = S.resampleClosed(stroke, 1.0);
  drawInput(); begin();
});

// ---- shapes and random points -----------------------------------------------------------

// the 20 start shapes used in training (world units; the canvas spans +/-54)
const ring = (n, f) => Array.from({ length: n }, (_, i) => f(i * 2 * Math.PI / n));
const poly = (n, r, rot = -Math.PI / 2) => ring(n, a => [r * Math.cos(a + rot), r * Math.sin(a + rot)]);
const star = (k, r1, r2) => ring(2 * k, a => { const i = Math.round(a / (Math.PI / k)), r = i % 2 ? r2 : r1; return [r * Math.cos(a - Math.PI / 2), r * Math.sin(a - Math.PI / 2)]; });
function crescent() {   // outer circle r 18 minus a disc r 13 centred 7 units right
  const out = [], d = Math.PI / 180;
  for (let t = 36; t <= 324; t += 12) out.push([18 * Math.cos(t * d), 18 * Math.sin(t * d)]);
  for (let t = 305; t >= 55; t -= 12) out.push([7 + 13 * Math.cos(t * d), 13 * Math.sin(t * d)]);
  return out;
}
function randomPolygon() {
  for (;;) {
    const pts = S.orderPoints(Array.from({ length: 5 + Math.floor(Math.random() * 6) }, () => {
      const a = Math.random() * 2 * Math.PI, r = 8 + Math.random() * 12; return [r * Math.cos(a), r * Math.sin(a)];
    }));
    if (S.polygonArea(pts) > 150) return pts;
  }
}
const SHAPES = {
  circle: ['Circle', poly(24, 16)],
  ellipse: ['Ellipse', ring(24, a => [21 * Math.cos(a), 11 * Math.sin(a)])],
  triangle: ['Triangle', poly(3, 19)],
  square: ['Square', [[-14, -14], [14, -14], [14, 14], [-14, 14]]],
  rectangle: ['Rectangle', [[-21, -10], [21, -10], [21, 10], [-21, 10]]],
  pentagon: ['Pentagon', poly(5, 18)],
  hexagon: ['Hexagon', poly(6, 18)],
  heptagon: ['Heptagon', poly(7, 18)],
  octagon: ['Octagon', poly(8, 18, -Math.PI / 8)],
  star5: ['Star 5', star(5, 20, 8)],
  star6: ['Star 6', star(6, 20, 10)],
  star8: ['Star 8', star(8, 20, 11)],
  cross: ['Cross', [[-5, -18], [5, -18], [5, -5], [18, -5], [18, 5], [5, 5], [5, 18], [-5, 18], [-5, 5], [-18, 5], [-18, -5], [-5, -5]]],
  L: ['L', [[-14, -19], [-4, -19], [-4, 8], [16, 8], [16, 18], [-14, 18]]],
  T: ['T', [[-19, -18], [19, -18], [19, -8], [5, -8], [5, 19], [-5, 19], [-5, -8], [-19, -8]]],
  U: ['U', [[-17, -18], [-7, -18], [-7, 7], [7, 7], [7, -18], [17, -18], [17, 18], [-17, 18]]],
  crescent: ['Crescent', crescent()],
  blob: ['Blob', ring(32, a => { const r = 16 + 3 * Math.sin(3 * a) + 2 * Math.cos(5 * a + 1); return [r * Math.cos(a), r * Math.sin(a)]; })],
  zigzag: ['Zigzag', [[-21, -3], [-10, -12], [0, -3], [10, -12], [21, -3], [21, 7], [10, -2], [0, 7], [-10, -2], [-21, 7]]],
  random: ['Random', null],
};

// ---- the visual shape chooser -----------------------------------------------------------

const RANDOM_ICON = [[-12, -15], [7, -18], [19, -3], [9, 16], [-7, 11], [-19, 1]];
const iconPath = pts => pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ') + 'Z';
const shapeIcon = k => `<svg viewBox="-24 -24 48 48"><path d="${iconPath(SHAPES[k][1] || RANDOM_ICON)}" fill="currentColor" fill-opacity=".12" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/></svg>`;
const shapeLabel = k => SHAPES[k][0].toLowerCase();

let shapeKey = null;
$('shapeGrid').innerHTML = Object.keys(SHAPES).map(k =>
  `<button data-shape="${k}" title="${shapeLabel(k)}" aria-label="${shapeLabel(k)}">${shapeIcon(k)}</button>`).join('');

function setShapeKey(k) {
  shapeKey = k;
  document.querySelectorAll('#shapeGrid button').forEach(b => b.classList.toggle('on', b.dataset.shape === k));
  $('startNote').textContent = uiMode === 'shape' && k ? shapeLabel(k) : '';
}

function chooseShape(k) {
  preview(SHAPES[k][1] || randomPolygon());
  setShapeKey(k);
}

// ---- growth character: the 6 simulator parameters (V6.3+ models take them as `params`) -----

const PARAMS = [   // key, label, range in the simulator, default
  ['minDistance', 'Min distance', .35, .6, .5],
  ['repulsionRadius', 'Repulsion radius', 1.5, 2.6, 2.0],
  ['brownianRange', 'Brownian range', .04, .30, .15],
  ['repulsionForce', 'Repulsion force', .025, .09, .05],
  ['attractionForce', 'Attraction force', .02, .08, .05],
  ['alignmentForce', 'Alignment force', 0, .05, .01],
];
const DEFAULTS = PARAMS.map(([, , lo, hi, d]) => (d - lo) / (hi - lo));
const pv = new Float32Array(DEFAULTS);            // normalised 0..1, the order the model expects
let hasParams = false;
const real = i => PARAMS[i][2] + pv[i] * (PARAMS[i][3] - PARAMS[i][2]);

// the simulator always uses the same values the network sees (defaults when the model has no params)
function applySim() {
  PARAMS.forEach(([k, , , , d], i) => { S.P[k] = hasParams ? real(i) : d; });
}

const FRIENDLY = { foldSize: 1, wildness: 2 };     // friendly slider -> PARAMS index

function syncParams() {
  for (const [id, i] of Object.entries(FRIENDLY)) $(id).value = Math.round(pv[i] * 1000);
  $('foldOut').textContent = real(1).toFixed(2);
  $('wildOut').textContent = real(2).toFixed(2);
  applySim();
}
for (const [id, i] of Object.entries(FRIENDLY)) $(id).oninput = e => { pv[i] = e.target.value / 1000; syncParams(); };
$('surprise').onclick = () => { for (let i = 0; i < pv.length; i++) pv[i] = Math.random(); syncParams(); };

// a start shape on the plate, ready to grow (press play)
function preview(pts) {
  reset(); mode = 'points'; points = pts.map(p => [...p]); drawInput(); showHint(false);
  setStatus('press grow or space'); refreshButtons();
}

function randomPoints() {
  const nv = +$('nPts').value, box = +$('spread').value;
  for (;;) {
    const n = nv < 3 ? 3 + Math.floor(Math.random() * 28) : nv;
    const pts = Array.from({ length: n }, () => [(Math.random() * 2 - 1) * box, (Math.random() * 2 - 1) * box]);
    if (S.polygonArea(S.orderPoints(pts)) > Math.min(30, box * box * 0.4)) return pts;
  }
}

// ---- controls ---------------------------------------------------------------------------

const ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.8v14.4L19 12z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4.2" height="14" rx="1"/><rect x="13.8" y="5" width="4.2" height="14" rx="1"/></svg>';
const ICON_RESTART = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12a7 7 0 1 0 2.1-5"/><path d="M5 4v4h4"/></svg>';
const SPEEDS = [0.5, 1, 2, 3, 4];
let uiMode = 'shape';      // shape | points | draw | random

const HINTS = {
  shape: ['choose a start shape', 'or click here to place points'], points: ['click to place points', 'three or more, then grow'],
  draw: ['draw a closed outline', 'it grows when you let go'], random: ['make random points', 'or click here to place your own'],
};
function showHint(on) {
  $('hint').style.opacity = on ? 1 : 0;
  [$('hintText').textContent, $('hintSub').textContent] = HINTS[uiMode];
}

// the four ways to start; the canvas takes clicks as points, or strokes in draw mode
function setUiMode(u) {
  uiMode = u;
  document.querySelectorAll('#modeSeg button').forEach(b => b.classList.toggle('on', b.dataset.mode === u));
  document.querySelectorAll('.pane').forEach(p => p.hidden = p.dataset.pane !== u);
  setMode(u === 'draw' ? 'draw' : 'points');
  if (u !== 'shape') setShapeKey(null); else setShapeKey(shapeKey);
}

function setMode(m) {
  mode = m; document.body.classList.toggle('drawing', m === 'draw');
  if (state === 'input') { if (m === 'draw') points = []; else stroke = null; drawInput(); showHint(!points.length && !stroke); }
  refreshButtons();
}

function setView(v) {
  view = v; document.querySelectorAll('#viewSeg button').forEach(b => b.classList.toggle('on', b.dataset.view === v));
  render('model'); render('sim');
}

function setStatus(s) { $('status').textContent = s; }

function refreshButtons() {
  const ready = !!session, hasStart = mode === 'draw' ? !!stroke : points.length >= 3;
  const g = $('grow');
  g.disabled = !ready || (state === 'input' && !hasStart);
  g.innerHTML = !ready ? 'loading…' : state === 'growing' ? ICON_PAUSE + 'pause' : state === 'paused' ? (ended ? ICON_RESTART + 'restart' : ICON_PLAY + 'resume') : ICON_PLAY + 'grow';
  g.classList.toggle('ready', ready && state === 'input' && hasStart);
  $('stepBtn').disabled = g.disabled || state === 'growing' || ended;
  $('random').disabled = !ready;
  $('ptCount').textContent = `${points.length} point${points.length === 1 ? '' : 's'}`;
  $('compareSwitch').classList.toggle('on', compare); $('compareTool').classList.toggle('on', compare);
  $('compareSwitch').setAttribute('aria-checked', compare);
}

function toggleCompare() {
  compare = !compare; document.body.classList.toggle('compare', compare); refreshButtons();
  if (start) {                       // restart from the same start so both sides line up
    const s = start, m = mode; reset(); mode = m;
    if (m === 'draw') stroke = s; else points = s.map(p => [...p]);
    begin();
  } else render('sim');
}

function saveImage() {
  const a = ctxs.model.canvas, b = compare ? ctxs.sim.canvas : null, W = a.width, pad = 48;
  const c = document.createElement('canvas'); c.width = b ? 2 * W + pad : W; c.height = W;
  const g = c.getContext('2d'); g.fillStyle = THEMES[theme].bg; g.fillRect(0, 0, c.width, c.height);
  g.drawImage(a, 0, 0); if (b) g.drawImage(b, W + pad, 0);
  const link = document.createElement('a');
  link.download = `neural-growth-step-${lanes.model ? lanes.model.step : 0}.png`; link.href = c.toDataURL('image/png'); link.click();
}

function setTheme(t) {
  theme = t; LUT = THEMES[t].lut; bucketColor = THEMES[t].buckets;
  if (t === 'light') document.documentElement.dataset.theme = 'light'; else delete document.documentElement.dataset.theme;
  try { localStorage.setItem('v6theme', t); } catch {}
  render('model'); render('sim'); drawInput();
}

document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  $('controlsTab').hidden = b.dataset.tab !== 'controls'; $('aboutTab').hidden = b.dataset.tab !== 'about';
});
document.querySelectorAll('#modeSeg button').forEach(b => b.onclick = () => { if (state !== 'input') reset(); setUiMode(b.dataset.mode); });
document.querySelectorAll('#shapeGrid button').forEach(b => {
  b.onclick = () => chooseShape(b.dataset.shape);
  b.onmouseenter = () => { $('startNote').textContent = shapeLabel(b.dataset.shape); };
});
$('shapeGrid').onmouseleave = () => { $('startNote').textContent = shapeKey ? shapeLabel(shapeKey) : ''; };
document.querySelectorAll('#viewSeg button').forEach(b => b.onclick = () => setView(b.dataset.view));
$('grow').onclick = begin;
$('stepBtn').onclick = () => { stepOnce = true; begin(); };
$('reset').onclick = () => { reset(); setShapeKey(null); };
$('clearPts').onclick = () => { reset(); };
$('random').onclick = () => preview(randomPoints());
$('nPts').oninput = e => $('nOut').textContent = +e.target.value < 3 ? 'any' : e.target.value;
$('spread').oninput = e => $('spreadOut').textContent = e.target.value;
$('speed').oninput = e => { const v = SPEEDS[+e.target.value]; stepMs = BASE_STEP_MS / v; $('speedOut').textContent = v + 'x'; };
$('compareSwitch').onclick = toggleCompare;
const toggleRaw = () => { rawOutput = !rawOutput; $('rawSwitch').classList.toggle('on', rawOutput); $('rawSwitch').setAttribute('aria-checked', rawOutput); };
$('rawSwitch').onclick = toggleRaw;
$('rawSwitch').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); toggleRaw(); } };
$('compareSwitch').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); toggleCompare(); } };
$('compareTool').onclick = toggleCompare;
$('saveTool').onclick = saveImage;
$('themeTool').onclick = () => setTheme(theme === 'light' ? 'dark' : 'light');
$('panelTool').onclick = () => document.body.classList.toggle('nopanel');
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || (e.target.tagName === 'BUTTON' && e.key !== 'Escape')) return;
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); begin(); }
  if (e.key === 'Escape') { reset(); setShapeKey(null); }
});

reset();
setUiMode('shape');
setStatus('loading model…');
refreshButtons();
loadModel();
