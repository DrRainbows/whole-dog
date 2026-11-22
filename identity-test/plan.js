// Trial planning for the identity test, shared by the page (app.js) and the tests (test/plan.test.js).
// Rules M3: same-breed foils, a target clip is never reused for a judge, positions random,
// and the whole plan is reproducible from the judge's code.
(function (root) {
  function hashSeed(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function shuffle(arr, r) { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

  function plan(M, judge, known) {
    const r = rng(hashSeed(judge));
    const dog = (id) => M.dogs.find((d) => d.id === id);
    const clipsOf = (dogId, carrier) => M.clips.filter((c) => c.dog === dogId && c.carrier === carrier);
    const trials = [];
    const shortfalls = [];
    for (const carrier of shuffle(M.carriers, r)) {
      const targets = known.filter((id) => clipsOf(id, carrier.id).length);
      if (!targets.length) continue;
      const pools = Object.fromEntries(targets.map((id) => [id, shuffle(clipsOf(id, carrier.id), r)]));
      let made = 0;
      for (let i = 0; made < M.trialsPerCarrier && i < M.trialsPerCarrier * 3; i++) {
        const target = targets[i % targets.length];
        const clip = pools[target].shift();
        if (!clip) continue;
        const breed = dog(target).breed;
        const foilDogs = shuffle(M.dogs.filter((d) => d.breed === breed && d.id !== target && clipsOf(d.id, carrier.id).length), r)
          .slice(0, M.options - 1);
        if (foilDogs.length < M.options - 1) { shortfalls.push(`${carrier.label}: not enough ${breed} foils`); break; }
        const foils = foilDogs.map((d) => { const pool = clipsOf(d.id, carrier.id); return pool[Math.floor(r() * pool.length)]; });
        trials.push({ carrier: carrier.id, target, options: shuffle([clip, ...foils], r).map((c) => c.file) });
        made++;
      }
      if (made < M.trialsPerCarrier) shortfalls.push(`${carrier.label}: ${made} of ${M.trialsPerCarrier} trials (not enough clips)`);
    }
    return { trials, shortfalls };
  }

  const api = { plan, rng, shuffle, hashSeed };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WhichDogPlan = api;
})(typeof window !== 'undefined' ? window : globalThis);
