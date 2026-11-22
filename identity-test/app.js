// Forced-choice identity test. Protocol: decisions/001-first-build.md, rules M3 to M5.
// Each trial shows the target dog and same-breed foils, in random positions.
// A judge never sees the same target clip twice, so trials are independent samples.
// Loads manifest.js, or the file named in ?manifest=..., then starts.
// ?judge=CODE&known=dog1,dog2 skips the setup screen (for testing and for sending prepared links).
(function boot() {
  const params = new URLSearchParams(location.search);
  const s = document.createElement('script');
  s.src = params.get('manifest') || 'manifest.js';
  s.onload = () => start(params);
  s.onerror = () => {
    document.querySelector('main').insertAdjacentHTML('beforeend',
      '<p class="card">No clips yet: run make_manifest.py after adding clips (see README).</p>');
  };
  document.head.appendChild(s);
})();

function start(params) {
  const M = window.MANIFEST;
  const $ = (id) => document.getElementById(id);
  const KEY = 'which-dog-progress:' + (params.get('manifest') || 'manifest.js');
  $('lead').textContent = `You'll see ${M.options} short clips at a time and pick the dog you know. ` +
    'Some show only moving dots, some only sound. Go with your first impression.';

  const dog = (id) => M.dogs.find((d) => d.id === id);
  // Every trial for this judge is built up front, reproducibly from their code (plan.js).
  const plan = (judge, known) => window.WhichDogPlan.plan(M, judge, known);

  let state = null;

  function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* storage unavailable */ } }
  function load() { try { return JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) { return null; } }

  function renderSetup() {
    const box = $('knows');
    box.innerHTML = '';
    M.dogs.filter((d) => d.target).forEach((d) => {
      const l = document.createElement('label');
      l.innerHTML = `<input type="checkbox" value="${d.id}"> ${d.name}`;
      box.appendChild(l);
    });
    const update = () => {
      const judge = $('judge').value.trim();
      const known = [...box.querySelectorAll('input:checked')].map((i) => i.value);
      $('start').disabled = !judge || !known.length;
      if (judge && known.length) {
        const p = plan(judge, known);
        $('plan').textContent = `${p.trials.length} trials, about ${Math.ceil(p.trials.length * 12 / 60)} minutes.` +
          (p.shortfalls.length ? ' Note: ' + p.shortfalls.join('; ') + '.' : '');
      } else $('plan').textContent = '';
    };
    $('judge').addEventListener('input', update);
    box.addEventListener('change', update);
    $('start').addEventListener('click', () => {
      const judge = $('judge').value.trim();
      const known = [...box.querySelectorAll('input:checked')].map((i) => i.value);
      state = { judge, known, ...plan(judge, known), i: 0, results: [], started: new Date().toISOString() };
      save();
      showTrial();
    });
  }

  function showTrial() {
    $('setup').classList.add('hidden');
    $('done').classList.add('hidden');
    if (state.i >= state.trials.length) return finish();
    $('trial').classList.remove('hidden');
    const t = state.trials[state.i];
    const carrier = M.carriers.find((c) => c.id === t.carrier);
    $('progress').style.width = `${(100 * state.i) / state.trials.length}%`;
    $('trialMeta').textContent = `${state.i + 1} of ${state.trials.length} · ${carrier.label}`;
    $('prompt').textContent = `Which one is ${dog(t.target).name}?`;
    const grid = $('options');
    grid.innerHTML = '';
    const shownAt = performance.now();
    t.options.forEach((file, k) => {
      const el = document.createElement('div');
      el.className = 'option' + (carrier.audio ? ' audio' : '');
      el.tabIndex = 0;
      el.setAttribute('role', 'button');
      el.setAttribute('aria-label', `Option ${k + 1}`);
      if (carrier.audio) {
        el.innerHTML = `<div class="tag" style="position:static;color:inherit;text-shadow:none">Option ${k + 1}</div>`;
        const a = document.createElement('audio');
        a.src = file; a.controls = true; a.preload = 'auto';
        a.addEventListener('click', (e) => e.stopPropagation());
        el.appendChild(a);
        const pick = document.createElement('button');
        pick.textContent = 'This one';
        pick.addEventListener('click', (e) => { e.stopPropagation(); choose(k, shownAt); });
        el.appendChild(pick);
      } else {
        const v = document.createElement('video');
        v.src = file; v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true; v.preload = 'auto';
        el.appendChild(v);
        const tag = document.createElement('div');
        tag.className = 'tag'; tag.textContent = k + 1;
        el.appendChild(tag);
        el.addEventListener('click', () => choose(k, shownAt));
        el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(k, shownAt); } });
      }
      grid.appendChild(el);
    });
  }

  function choose(k, shownAt) {
    const t = state.trials[state.i];
    const file = t.options[k];
    const chosen = M.clips.find((c) => c.file === file);
    state.results.push({
      judge: state.judge, carrier: t.carrier, trial: state.i + 1, target: t.target,
      target_clip: t.options.find((f) => M.clips.find((c) => c.file === f).dog === t.target),
      chosen_dog: chosen.dog, chosen_clip: file, correct: chosen.dog === t.target ? 1 : 0,
      rt_ms: Math.round(performance.now() - shownAt), options: t.options.join('|'), n_options: t.options.length,
      time: new Date().toISOString(),
    });
    state.i++;
    save();
    showTrial();
  }

  function finish() {
    $('trial').classList.add('hidden');
    $('done').classList.remove('hidden');
  }

  function csv() {
    const cols = ['judge', 'carrier', 'trial', 'target', 'target_clip', 'chosen_dog', 'chosen_clip', 'correct', 'rt_ms', 'n_options', 'options', 'time'];
    const esc = (v) => `"${String(v).replace(/"/g, '""')}"`;
    return [cols.join(','), ...state.results.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n');
  }

  $('download').addEventListener('click', () => {
    const blob = new Blob([csv()], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `which-dog_${state.judge.replace(/[^a-z0-9_-]/gi, '')}_${state.started.slice(0, 10)}.csv`;
    a.click();
  });
  $('restart').addEventListener('click', () => { try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ } location.reload(); });

  const saved = load();
  const preJudge = params.get('judge');
  const preKnown = (params.get('known') || '').split(',').filter((id) => M.dogs.some((d) => d.id === id));
  if (preJudge && preKnown.length && !(saved && saved.judge === preJudge)) {
    state = { judge: preJudge, known: preKnown, ...plan(preJudge, preKnown), i: 0, results: [], started: new Date().toISOString() };
    save();
    showTrial();
  } else if (saved && saved.trials && saved.i < saved.trials.length) { state = saved; showTrial(); }
  else if (saved && saved.trials && saved.i >= saved.trials.length) { state = saved; finish(); }
  else renderSetup();
}
