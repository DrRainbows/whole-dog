// Every number that defines one dog.
//
// "label" is the named feature shown in the slider condition (the vocabulary approach).
// The evolve condition never shows these names: the owner only chooses between whole dogs.
// Ranges and presets are estimates for medium-to-large dogs, to be refined against footage.
import { gauss } from './rng.js';

const P = (key, label, group, min, max, def, unit = '') => ({ key, label, group, min, max, def, unit });

export const GROUPS = {
  body: 'Build',
  gait: 'Gait',
  carriage: 'Carriage',
  soft: 'Ears and tail',
  idle: 'Standing and breathing',
};

export const PARAMS = [
  P('withers', 'Height at the withers', 'body', 0.45, 0.70, 0.58, 'm'),
  P('bodyRatio', 'Body length to height', 'body', 0.95, 1.30, 1.10),
  P('legScale', 'Leg length', 'body', 0.88, 1.12, 1.00),
  P('chestDepth', 'Chest depth', 'body', 0.42, 0.55, 0.48),
  P('heft', 'Heaviness', 'body', 0, 1, 0.5),
  P('headSize', 'Head size', 'body', 0.85, 1.20, 1.00),
  P('muzzle', 'Muzzle length', 'body', 0.80, 1.25, 1.00),
  P('earType', 'Ears: prick to drop', 'body', 0, 1, 0.6),
  P('earLen', 'Ear length', 'body', 0.70, 1.40, 1.00),
  P('tailLen', 'Tail length', 'body', 0.60, 1.25, 1.00),
  P('tailThick', 'Tail thickness', 'body', 0.60, 1.80, 1.00),

  P('trotSpeed', 'Trotting speed', 'gait', 1.6, 3.4, 2.4, 'm/s'),
  P('strideFreq', 'Strides per second', 'gait', 1.7, 2.9, 2.2, 'Hz'),
  P('duty', 'Time each foot spends on the ground', 'gait', 0.38, 0.56, 0.46),
  P('diagLag', 'Front foot lands after its diagonal hind', 'gait', -0.06, 0.08, 0.02),
  P('reachF', 'Front reach', 'gait', 0.80, 1.25, 1.00),
  P('driveH', 'Rear drive', 'gait', 0.80, 1.25, 1.00),
  P('liftF', 'Front foot lift', 'gait', 0.08, 0.32, 0.16),
  P('liftH', 'Hind foot lift', 'gait', 0.06, 0.24, 0.12),
  P('flickF', 'Paw fold in the air', 'gait', 0.3, 1.0, 0.7),
  P('cycleNoise', 'Step-to-step irregularity', 'gait', 0, 0.06, 0.015),
  P('asym', 'Left-right unevenness', 'gait', -0.08, 0.08, 0),

  P('bounce', 'Bounce', 'carriage', 0.01, 0.06, 0.03),
  P('pitch', 'Front-back rocking', 'carriage', 0, 4, 1.5, '°'),
  P('spine', 'Back flex', 'carriage', 0, 8, 2, '°'),
  P('neckAngle', 'Head carriage', 'carriage', 10, 65, 38, '°'),
  P('headBob', 'Head nod', 'carriage', 0, 8, 3, '°'),
  P('headStab', 'Head steadiness', 'carriage', 0, 1, 0.5),
  P('neckFreq', 'Neck stiffness', 'carriage', 1.5, 6, 3.2, 'Hz'),
  P('neckDamp', 'Neck damping', 'carriage', 0.25, 0.95, 0.55),
  P('scapSwing', 'Shoulder swing', 'carriage', 4, 18, 10, '°'),
  P('tailCarriage', 'Tail carriage', 'carriage', -45, 55, 5, '°'),

  P('tailStiff', 'Tail stiffness', 'soft', 1.0, 6.0, 2.5, 'Hz'),
  P('tailDamp', 'Tail damping', 'soft', 0.05, 0.6, 0.25),
  P('wagAmp', 'Wag size', 'soft', 0, 50, 20, '°'),
  P('wagFreq', 'Wag speed', 'soft', 1.5, 5.0, 3.0, 'Hz'),
  P('earStiff', 'Ear stiffness', 'soft', 1.2, 8.0, 3.0, 'Hz'),
  P('earDamp', 'Ear damping', 'soft', 0.05, 0.5, 0.2),

  P('breath', 'Breaths per minute at rest', 'idle', 12, 36, 22, '/min'),
  P('heatGain', 'Heats up when moving', 'idle', 0.4, 1.6, 1.0),
  P('coolRate', 'Cools down at rest', 'idle', 0.4, 1.6, 1.0),
  P('fidget', 'Weight shifts per minute', 'idle', 1, 14, 5, '/min'),
  P('lookRate', 'Head turns per minute', 'idle', 2, 24, 9, '/min'),
  P('stopSharp', 'Stop sharpness', 'idle', 0, 1, 0.5),
  P('squareUp', 'Squares up the feet after stopping', 'idle', 0, 1, 0.6),
];

export const KEYS = PARAMS.map((q) => q.key);
const BY_KEY = Object.fromEntries(PARAMS.map((q) => [q.key, q]));

export function defaults() {
  return Object.fromEntries(PARAMS.map((q) => [q.key, q.def]));
}

// Starting points only. The owner's choices move the dog away from these.
export const PRESETS = {
  generic: {},
  labrador: {
    withers: 0.57, bodyRatio: 1.12, legScale: 0.96, chestDepth: 0.50, heft: 0.75, headSize: 1.08, muzzle: 0.95,
    earType: 0.95, earLen: 1.05, tailLen: 0.95, tailThick: 1.7,
    trotSpeed: 2.3, strideFreq: 2.15, duty: 0.47, diagLag: 0.02, reachF: 0.95, driveH: 1.0, liftF: 0.14, liftH: 0.11,
    flickF: 0.6, cycleNoise: 0.02, bounce: 0.032, pitch: 1.8, spine: 2.5, neckAngle: 32, headBob: 3.5, headStab: 0.45,
    neckFreq: 2.8, neckDamp: 0.5, scapSwing: 9, tailCarriage: 5, tailStiff: 2.2, tailDamp: 0.22, wagAmp: 30,
    wagFreq: 2.6, earStiff: 2.0, earDamp: 0.15, breath: 20, heatGain: 1.15, coolRate: 0.85, fidget: 4, lookRate: 7,
    stopSharp: 0.4, squareUp: 0.4,
  },
  malinois: {
    withers: 0.62, bodyRatio: 1.0, legScale: 1.06, chestDepth: 0.46, heft: 0.3, headSize: 0.95, muzzle: 1.1,
    earType: 0.0, earLen: 1.15, tailLen: 1.1, tailThick: 0.8,
    trotSpeed: 2.9, strideFreq: 2.35, duty: 0.42, diagLag: 0.03, reachF: 1.15, driveH: 1.15, liftF: 0.18, liftH: 0.13,
    flickF: 0.8, cycleNoise: 0.012, bounce: 0.022, pitch: 1.0, spine: 2.0, neckAngle: 42, headBob: 2.0, headStab: 0.7,
    neckFreq: 4.2, neckDamp: 0.6, scapSwing: 12, tailCarriage: -10, tailStiff: 3.5, tailDamp: 0.3, wagAmp: 15,
    wagFreq: 3.5, earStiff: 6.5, earDamp: 0.3, breath: 24, heatGain: 0.9, coolRate: 1.1, fidget: 9, lookRate: 16,
    stopSharp: 0.8, squareUp: 0.8,
  },
};

export function preset(name) {
  return { ...defaults(), ...(PRESETS[name] || {}) };
}

// Genes are the same numbers scaled to 0..1, which is where evolution works.
export function toGenes(values) {
  return PARAMS.map((q) => (values[q.key] - q.min) / (q.max - q.min));
}

export function fromGenes(genes) {
  const out = {};
  PARAMS.forEach((q, i) => {
    const g = Math.min(1, Math.max(0, genes[i]));
    out[q.key] = q.min + g * (q.max - q.min);
  });
  return out;
}

// Which genes evolution may change: by default everything but the build.
export function maskFor(groups = ['gait', 'carriage', 'soft', 'idle']) {
  return PARAMS.map((q) => groups.includes(q.group));
}

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function mutate(genes, sigma, rng, mask) {
  return genes.map((g, i) => (mask[i] ? clamp01(g + sigma * gauss(rng)) : g));
}

// Blend two parents gene by gene, with a random mix for each gene.
export function crossover(a, b, rng, mask) {
  return a.map((g, i) => (mask[i] ? g + (b[i] - g) * rng() : g));
}

export function geneDistance(a, b, mask) {
  let s = 0;
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    if (!mask || mask[i]) {
      s += (a[i] - b[i]) ** 2;
      n++;
    }
  }
  return Math.sqrt(s / Math.max(1, n));
}

export function describe(key) {
  return BY_KEY[key];
}
