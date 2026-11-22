// Protocol checks for the trial planner (rule M3). Run: node --test test/
const test = require('node:test');
const assert = require('node:assert/strict');
const { plan } = require('../plan.js');

function manifest() {
  const dogs = [
    { id: 'juno', name: 'Juno', breed: 'labrador', target: true },
    { id: 'lab_b', name: 'Lab B', breed: 'labrador' }, { id: 'lab_c', name: 'Lab C', breed: 'labrador' },
    { id: 'lab_d', name: 'Lab D', breed: 'labrador' },
    { id: 'nova', name: 'Nova', breed: 'malinois', target: true },
    { id: 'mal_b', name: 'Mal B', breed: 'malinois' }, { id: 'mal_c', name: 'Mal C', breed: 'malinois' },
    { id: 'mal_d', name: 'Mal D', breed: 'malinois' },
  ];
  const carriers = [{ id: 'gait-dots', label: 'trot' }, { id: 'react-dots', label: 'react' }];
  const clips = [];
  for (const d of dogs) for (const c of carriers) {
    const n = d.target ? 12 : 4;
    for (let i = 0; i < n; i++) clips.push({ file: `clips/${d.id}__${c.id}__${i}.mp4`, dog: d.id, carrier: c.id });
  }
  return { dogs, carriers, clips, trialsPerCarrier: 20, options: 4 };
}

const M = manifest();
const dogOf = (file) => M.clips.find((c) => c.file === file).dog;
const breedOf = (id) => M.dogs.find((d) => d.id === id).breed;

test('REQ-T1 every trial has exactly one option from the target dog and the rest from other dogs', () => {
  const { trials } = plan(M, 'J1', ['juno', 'nova']);
  assert.ok(trials.length > 0);
  for (const t of trials) {
    assert.equal(t.options.length, 4);
    assert.equal(t.options.filter((f) => dogOf(f) === t.target).length, 1);
    assert.equal(new Set(t.options.map(dogOf)).size, 4);
  }
});

test('REQ-T2 foils are always the same breed as the target', () => {
  const { trials } = plan(M, 'J2', ['juno', 'nova']);
  for (const t of trials) for (const f of t.options) assert.equal(breedOf(dogOf(f)), breedOf(t.target));
});

test('REQ-T3 a judge never sees the same target clip twice', () => {
  const { trials } = plan(M, 'J3', ['juno', 'nova']);
  const seen = new Set();
  for (const t of trials) {
    const tc = t.options.find((f) => dogOf(f) === t.target);
    assert.ok(!seen.has(tc), `repeated ${tc}`);
    seen.add(tc);
  }
});

test('REQ-T4 the target appears in every position about equally often', () => {
  const counts = [0, 0, 0, 0];
  for (let j = 0; j < 60; j++) {
    for (const t of plan(M, `J${j}`, ['juno', 'nova']).trials) counts[t.options.findIndex((f) => dogOf(f) === t.target)]++;
  }
  const total = counts.reduce((a, b) => a + b, 0);
  for (const c of counts) assert.ok(Math.abs(c / total - 0.25) < 0.04, `position counts ${counts}`);
});

test('REQ-T5 the plan is reproducible from the judge code and differs between judges', () => {
  assert.deepEqual(plan(M, 'same', ['juno']), plan(M, 'same', ['juno']));
  assert.notDeepEqual(plan(M, 'A', ['juno']).trials, plan(M, 'B', ['juno']).trials);
});

test('REQ-T6 a shortage of clips is reported, not hidden', () => {
  const small = { ...M, clips: M.clips.filter((c) => !(c.dog === 'juno' && c.carrier === 'gait-dots' && /__([5-9]|1\d)\.mp4$/.test(c.file))) };
  const { trials, shortfalls } = plan(small, 'J', ['juno']);
  assert.equal(trials.filter((t) => t.carrier === 'gait-dots').length, 5);
  assert.ok(shortfalls.some((s) => s.includes('5 of 20')));
});
