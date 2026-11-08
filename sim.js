// Differential growth simulator: the same rules and DEFAULT_PARAMS as code/v6_sim.py
// (split, prune, repulsion, attraction, alignment, Brownian). Repulsion uses a grid hash,
// the JavaScript equivalent of the KD-tree in Python: same force, only nearby pairs checked.

export const P = {
  maxDistance: 1.0, minDistance: 0.5, repulsionRadius: 2.0, brownianRange: 0.15,
  repulsionForce: 0.05, attractionForce: 0.05, alignmentForce: 0.01,
};
export const EVERY = 4;            // one model step = 4 simulator iterations
export const RES = 384, WIN = 54;  // demo canvas: 384 px over +/-54 units (same scale as training: 256 px over +/-36)

function split(pts) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    out.push(a);
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) >= P.maxDistance) out.push([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
  }
  return out;
}

function prune(pts, floor = 4) {
  if (pts.length <= floor) return pts;
  const kept = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const k = kept[kept.length - 1];
    if (Math.hypot(pts[i][0] - k[0], pts[i][1] - k[1]) >= P.minDistance) kept.push(pts[i]);
  }
  const l = kept[kept.length - 1];
  if (kept.length > floor && Math.hypot(l[0] - kept[0][0], l[1] - kept[0][1]) < P.minDistance) kept.pop();
  return kept;
}

function step(pts) {
  pts = prune(split(pts));
  const n = pts.length, R = P.repulsionRadius, dx = new Float64Array(n), dy = new Float64Array(n);

  // repulsion: every pair closer than R pushes apart, (R - dist) * force
  const grid = new Map();
  for (let i = 0; i < n; i++) {
    const key = Math.floor(pts[i][0] / R) * 73856093 ^ Math.floor(pts[i][1] / R) * 19349663;
    (grid.get(key) || grid.set(key, []).get(key)).push(i);
  }
  for (let i = 0; i < n; i++) {
    const cx = Math.floor(pts[i][0] / R), cy = Math.floor(pts[i][1] / R);
    for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) {
      const cell = grid.get((cx + ox) * 73856093 ^ (cy + oy) * 19349663);
      if (!cell) continue;
      for (const j of cell) {
        if (j <= i) continue;
        const vx = pts[i][0] - pts[j][0], vy = pts[i][1] - pts[j][1], r = Math.hypot(vx, vy);
        if (r >= R || r === 0) continue;
        const f = (R - r) * P.repulsionForce / r;
        dx[i] += vx * f; dy[i] += vy * f; dx[j] -= vx * f; dy[j] -= vy * f;
      }
    }
  }
  // attraction to both neighbours, alignment to their midpoint, Brownian jitter
  const out = new Array(n), b = P.brownianRange;
  for (let i = 0; i < n; i++) {
    const p = pts[i], a = pts[(i - 1 + n) % n], c = pts[(i + 1) % n];
    const x = p[0] + dx[i] + P.attractionForce * (a[0] + c[0] - 2 * p[0])
      + P.alignmentForce * ((a[0] + c[0]) / 2 - p[0]) + (Math.random() - 0.5) * b;
    const y = p[1] + dy[i] + P.attractionForce * (a[1] + c[1] - 2 * p[1])
      + P.alignmentForce * ((a[1] + c[1]) / 2 - p[1]) + (Math.random() - 0.5) * b;
    out[i] = [x, y];
  }
  return out;
}

// Yields the node list every `every` iterations, for as long as asked.
export function* grow(start, every = EVERY) {
  let pts = start.map(p => [p[0], p[1]]);
  for (;;) {
    for (let i = 0; i < every; i++) pts = step(pts);
    yield pts;
  }
}

// A drawn stroke -> closed curve with nodes about 1 unit apart (the simulator's maxDistance).
export function resampleClosed(pts, spacing = 1.0) {
  const c = [...pts, pts[0]], seg = [0];
  for (let i = 1; i < c.length; i++) seg.push(seg[i - 1] + Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]));
  const L = seg[seg.length - 1], n = Math.max(8, Math.round(L / spacing)), out = [];
  for (let k = 0, j = 0; k < n; k++) {
    const s = (k / n) * L;
    while (seg[j + 1] < s) j++;
    const t = (s - seg[j]) / Math.max(1e-9, seg[j + 1] - seg[j]);
    out.push([c[j][0] + t * (c[j + 1][0] - c[j][0]), c[j][1] + t * (c[j + 1][1] - c[j][1])]);
  }
  return out;
}

// ---- start points ---------------------------------------------------------

// Join points in angle order around their centre: a simple polygon, as in training.
export function orderPoints(pts) {
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return [...pts].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
}

export function polygonArea(p) {
  let s = 0;
  for (let i = 0; i < p.length; i++) { const j = (i + 1) % p.length; s += p[i][0] * p[j][1] - p[j][0] * p[i][1]; }
  return Math.abs(s / 2);
}

export function randomStart() {
  for (;;) {
    const n = 3 + Math.floor(Math.random() * 30), box = 8 + Math.random() * 14;
    const pts = orderPoints(Array.from({ length: n }, () => [(Math.random() * 2 - 1) * box, (Math.random() * 2 - 1) * box]));
    if (polygonArea(pts) > 40) return pts;
  }
}

// ---- the image the model sees (same as render_fill in Python) ----------------

const raster = (() => { const c = document.createElement('canvas'); c.width = c.height = RES; return c.getContext('2d', { willReadFrequently: true }); })();
const toPx = v => (v + WIN) * (RES / (2 * WIN));

export function renderFill(pts) {
  raster.fillStyle = '#000'; raster.fillRect(0, 0, RES, RES);
  raster.fillStyle = '#fff'; raster.beginPath();
  pts.forEach((p, i) => (i ? raster.lineTo : raster.moveTo).call(raster, toPx(p[0]), toPx(p[1])));
  raster.closePath(); raster.fill();
  const d = raster.getImageData(0, 0, RES, RES).data, out = new Float32Array(RES * RES);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4] > 127 ? 1 : 0;
  return out;
}
