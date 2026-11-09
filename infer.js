// Runs the network in a background worker so the page never freezes while it thinks.
// Messages in:  {type:'load', url, base, eps}  |  {type:'run', id, fill, prev, params, R}
// Messages out: {type:'progress', p} | {type:'loaded', ok, inputs, ep} | {type:'result', id, next, ms} | {type:'error', id, msg}

importScripts('vendor/ort/ort.webgpu.min.js');

let session = null, ep = '', modelBytes = null;

async function fetchModel(url) {
  const res = await fetch(url);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    postMessage({ type: 'progress', p: Math.min(0.99, got / total) });
  }
  const out = new Uint8Array(got);
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

async function create(which) {
  if (session) { try { await session.release(); } catch {} session = null; }
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [which], logSeverityLevel: 3,
    enableCpuMemArena: false, enableMemPattern: false,   // lower peak memory (phones)
  });
  ep = which;
}

async function runOnce(m) {
  const feeds = { fill: new ort.Tensor('float32', m.fill, [1, 1, m.R, m.R]) };
  if (session.inputNames.includes('prev')) feeds.prev = new ort.Tensor('float32', m.prev, [1, 1, m.R, m.R]);
  if (session.inputNames.includes('params') && m.params) feeds.params = new ort.Tensor('float32', m.params, [1, m.params.length]);
  const out = await session.run(feeds);
  const lg = out[session.outputNames[0]].data, next = new Float32Array(lg.length);
  for (let i = 0; i < lg.length; i++) next[i] = lg[i] > 0 ? 1 : 0;
  for (const k in out) out[k].dispose?.();
  return next;
}

onmessage = async ({ data: m }) => {
  if (m.type === 'load') {
    ort.env.logLevel = 'error';
    ort.env.wasm.wasmPaths = m.base + 'vendor/ort/';
    ort.env.wasm.numThreads = 1;
    try { modelBytes = await fetchModel(m.url); } catch (e) { postMessage({ type: 'loaded', ok: false }); return; }
    postMessage({ type: 'progress', p: 1 });
    for (const which of m.eps) {
      try { await create(which); break; } catch (e) { session = null; }
    }
    if (ep === 'wasm') modelBytes = null;   // no fallback needed: free the copy
    postMessage({ type: 'loaded', ok: !!session, inputs: session ? session.inputNames : [], ep });
    return;
  }
  if (m.type === 'run') {
    const t0 = performance.now();
    try {
      const next = await runOnce(m);
      postMessage({ type: 'result', id: m.id, next, ms: performance.now() - t0 }, [next.buffer]);
    } catch (e) {
      // the GPU path failed mid-run: switch to WASM once and retry
      if (ep !== 'wasm' && modelBytes) {
        try {
          await create('wasm'); modelBytes = null;
          const next = await runOnce(m);
          postMessage({ type: 'result', id: m.id, next, ms: performance.now() - t0, ep }, [next.buffer]);
          return;
        } catch (e2) { e = e2; }
      }
      postMessage({ type: 'error', id: m.id, msg: String(e && e.message || e) });
    }
  }
};
