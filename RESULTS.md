# Results log

*This is the project's working lab notebook, kept in the order it was written. "You" and "your dogs" refer to the project owner and the two dogs the work is modelled on. Videos linked here are simulation renders; footage and trained weights are not in this repository (available on request).*

What was tried, what worked, and what didn't, with numbers. Newest first. Requirements and their status: [REQUIREMENTS.md](REQUIREMENTS.md). The system map: [SYSTEM.md](SYSTEM.md).

## 2026-11-21, evening: the first playable (Panda3D)

Engine: Panda3D for the prototype (your choice). Your longer-term preference is a custom C/C++ or Rust stack with a portable brain, so the game is built around a runtime with a clean boundary.

- **[game/dog_runtime.py](game/dog_runtime.py):** the dog at run time, with no JAX. One `step()` is one 20 ms step of the whole dog: brain, words, matcher, the dog-strength physics body, the reflex and the fall logic. It is numpy, MuJoCo and plain data, so it can move to a native stack.
- **[game/dog_game.py](game/dog_game.py)** ([game/play_dog.bat](game/play_dog.bat)): the game window.
  - It draws the physics dog from MuJoCo's own geometry every frame, with a person, a ground and a following camera.
  - The HUD shows what the dog is doing, its breathing and arousal, and the words it heard.
  - You type to the dog (Enter), and F1–F6 are quick words: sit, down, come, stand, jump, shake. F8 shoves it, and Tab switches the camera.

**Checked:**
- **Runtime against the research loop** (`online_dog.py --cpu`): the Malinois's words, same seed, same tracker.
  - Same brain decisions.
  - Falls at 12.6 s and 39.8 s (research loop: 12.6 and 40.2), back up at 13.7 s and 40.8 s (13.8 and 41.5).
- **Speed, native on Windows:** 2.8 ms per 20 ms step, 7× faster than real time on one core, ready in 5 s.
- **Self-test** (offscreen, every control a player has; `--selftest`): the Lab heard "platz" and declined (not her word), came on "come", and took "go to the fridge door and fetch me a beer" as a yes, all read by the grammar locally.
- **Demo** ([game/demo_mal_jump.mp4](game/demo_mal_jump.mp4), recorded offscreen): the Malinois in a down hears "platz", then an insisted "jump". She springs up, falls, the reflex has her up within a second, then "come" and "sit".
- **Both dogs in one world** ([game/world.py](game/world.py); `dog_game.py --dogs lab,mal`):
  - **Composition:** the world is built from copies of the trained body with MjSpec, names prefixed "lab/" and "mal/". Each dog reads and drives it through index maps from its own single-dog layout, so its observation, muscles, policy and reflex are exactly what it was trained with.
  - **Contact:** each dog's capsules touch the floor and the other dog, never itself.
  - **Physics step:** every dog's muscles act, then one shared physics step.
  - **Brains:** each dog has its own, and both read the same person.
  - **Words:** go to both dogs, or to one when they start with her name ("mal, down").
  - **Checks:**
    - a world of one reproduces the single dog (falls at 12.6 s and 39.8 s, up at 13.7 s and 40.8 s);
    - both dogs together take 6.1 ms per 20 ms step;
    - the self-test routes "mal, sit" to the Malinois only.
  - **Demo:** [game/demo_both.mp4](game/demo_both.mp4).
- **The breed colours** never showed before this: the swap looked for grey, and the body's default colour isn't grey. The Malinois is now fawn with a black mask and ears, and the Lab cream.
- **You are the person** (arrow keys). The brain reads your position from one shared array, so whatever it does for you or toward you follows you.
- **A grammar bug the game exposed:** "come", "here" and "heel" said alone weren't aimed at the speaker. The grammar only marked a link "to the person" when a word like "me" followed it, so the dog walked to the default spot (−3, 1) instead. The research replays never checked where the dog went.
  - Recall and handler-position cues now mean "to the one who said it" ([dog/grammar.py](dog/grammar.py) `TO_PERSON`).
  - Check: you walk 5 m away and say "come". Both dogs walk to you and stop 0.25–0.5 m away. A Lab lying on her side may still decline ("not now", 25%): that's her brain's choice.

## 2026-11-21, evening: the whole dog in real time on one CPU core

A game needs one dog at 50 Hz well inside 20 ms per step. The online loop on the GPU took 107 ms per step for one dog, because the GPU is built for thousands of dogs at once.

[body/train/dog_cpu.py](body/train/dog_cpu.py) runs the same body, muscles and trained policies on native MuJoCo and numpy, with no JAX:
- [body/train/export_policy.py](body/train/export_policy.py) writes each policy as numpy, plus the body's constants taken from the training environment itself, so the two can't drift.
- `online_dog.py --cpu` uses it for the tracker and the `scramble_up` reflex.

**Checked against the JAX environment** ([body/train/parity_cpu.py](body/train/parity_cpu.py), `anat_dog`):

| check | result |
|---|---|
| observation, same state | equal to 4 × 10⁻⁷ |
| policy, numpy against JAX, same input | equal to 4 × 10⁻⁷ |
| physics, same actions from the same start | trunks 0.4 mm apart after 10 steps, 2.6 cm after 50 (two engines, chaotic contact: expected) |
| the CPU dog on its own, 6 starts × 3 s of capture | 6 of 6 tracked to the end, as on the GPU |

**Speed:**
- **The dog alone** (observation, policy, five 4 ms physics steps with the muscles): 1.84 ms per 20 ms step.
- **The whole online loop** (brain, spoken words, motion matcher, reference window, dog, reflex, fall detector): **5.1 ms per 20 ms step**, real time with 4× to spare.

**Same behaviour:** the Malinois's words replayed on the CPU tell the same story:
- the insisted jump lags at 12.7 s, really falls at 13.2 s, and she's up at 14.2 s (GPU: 13.2 and 14.3);
- the second jump falls at 40.1 s, and she's up at 41.0 s.

**Many dogs at once.** With the dog on the CPU, sessions run side by side across the CPU's cores ([body/wsl_online_batch.sh](body/wsl_online_batch.sh), [dog/online_batch_summary.py](dog/online_batch_summary.py)): 20 one-minute sessions in one minute of wall time. Tracker `anat_tired`, reflex `scramble_up_dog3` (72M), 10 minutes per dog per condition:

| | real falls per minute | doing | back on its feet | lags per minute | walked per minute |
|---|---|---|---|---|---|
| Lab, alone | 0.1 | walking | 1 of 1, 1.1 s | 0 | 4.4 m |
| Malinois, alone | 0 | | | 0 | 6.3 m |
| Lab, a person present | 0.2 | walking (2) | 2 of 2, 0.9 s | 0.1 | 10.6 m |
| Malinois, a person present | 0.1 | running | 1 of 1, 1.1 s | 0 | 14.8 m |

- **The tracker that never practised tired** (`anat_dog`), with a person present: Lab 0.3 falls per minute, Malinois 0.1, all recovered.
- **In ordinary life** the dogs fall about once every 5–10 minutes of play, always while moving, and get up in about a second. A real dog walking doesn't fall, so the falls while walking are the next thing to remove.

**The baseline, larger** (50 one-minute sessions per dog, a person present, `anat_tired`):
- **Lab:** 0.24 falls per minute (12: walking 10, standing 1, trotting 1). 12 of 12 back on its feet, median 0.94 s.
- **Malinois:** 0.12 falls per minute (6: walking 3, running 2, trotting 1). 6 of 6 back up.

**Where they come from:**
- **The Lab's falls cluster in the first second of walking off slowly** (0.1–0.2 m/s), right after its own "bring gift" (grab, carry, present).
- **Two falls began as stumbles the online loop first read as lag.** One had already rolled 56°. The online fall threshold is now 45° of thorax roll: no stand, walk, trot, run, jump or sit frame of the capture rolls that far.
- **Slow walking is where it falls.** Before the Lab's 10 walking falls, the reference was walking at a median 0.32 m/s; half were below 0.3 m/s. The reference walks that slowly only 13% of the time (median 0.90 m/s), so falling is about 7× likelier there. Those moments are starting off and arriving.
- **Training never boosted them.** The transition boost counts stand, walk, trot and run as one posture, so changes of pace are never treated as transitions.

**Shoves, the test (fixed at 17:40, before the run).**
- **Run:** `anat_push` is `anat_tired` continued for 100M steps with random shoves (on average every 3 s, up to 0.2 body weights for 0.15 s), with capacity still 0.55–1.1 and the same references and settings.
- **S1, the claim:** in the same 100 social sessions (same seeds), real falls per minute, both dogs together, are fewer than `anat_tired`'s 18 in 100 minutes (one-sided conditional binomial test on the two counts, p < 0.05).
- **S2, validity:** the capture is at least 90% complete (`anat_tired`: 92.5%).

**Changes of pace, the test (fixed at 17:43, before the run), run alongside the shoves.**
- **Run:** `anat_pace` is `anat_tired` continued for 100M steps with starts just before a change of pace weighted 2×. A change of pace is the reference's speed changing by 0.4 m/s, or crossing 0.3 m/s, within the next second; that marks 32% of starts. Capacity stays 0.55–1.1, with no shoves, the same references and settings.
- **P1, the claim:** in the same 100 social sessions, fewer real falls than `anat_tired`'s 18 (one-sided conditional binomial test, p < 0.05).
- **P2, validity:** the capture is at least 90% complete.

**Result, both fixes** (the same 100 social sessions as the baseline):

| | real falls in 100 min | Lab | Malinois | p (fewer than 18) | capture complete |
|---|---|---|---|---|---|
| `anat_tired` (baseline) | 18 | 12 (walking 10) | 6 | | 92.5% |
| `anat_push` (shoves) | 15 | 6 (walking 3) | 9 (standing 4) | 0.36: **S1 fails** | 92.8%: S2 passes |
| `anat_pace` (pace changes) | 14 | 8 (walking 5) | 6 | 0.30: **P1 fails** | 93.4%: P2 passes |

- **Neither fix is shown to help.** Both cut the Lab's walking falls, but the totals don't move beyond chance, and the shoves made the Malinois fall more while standing.
- Every fall was recovered except one that came 0.9 s before its session ended.
- A larger comparison is running (exploratory, fresh seeds): 150 sessions per dog for each tracker.

**The consolidated dog (a build step, not an experiment), started 18:30.**
- `anat_final` is `anat_lig` (the ligament body; L1 and L2 pass) continued for 150M steps with practice tired (capacity 0.55–1.1; T1 passes), paired with `scramble_up_lig`.
- **Accepted if** the capture is at least 90% complete fresh and at least 85% at capacity 0.65, and the social batch doesn't fall more often than `anat_tired`'s.

## 2026-11-21, afternoon: a body with joints, ligaments and a dog's muscles

**Why.** After the grammar, physics still couldn't judge over-commanding:
- forced and routed changes fell equally often;
- practising sentences erased the one hint that they didn't.

Your first notes describe the body that is missing:
- "the only things you really have are a handful of joints with known movement constraints, and sure you can break a joint but its a near zero out of plane motion until broken";
- "the muscles are what protect that from the ligaments which carry the force the muscles command";
- the signals "going from the brain or solving locally at the limbs";
- "they will 'fall' if over commanded, not interpolate".

**What the old body actually had.** Every joint was three unlimited hinges about the BVH's own axes, measured from its straight-line rest pose. A standing dog was therefore a large rotation from zero, so hinges neared gimbal lock and capture noise wound them through whole turns. In the capture (refs/flat):
- the right shoulder hinge reaches −1065°;
- the right elbow reaches +617°;
- the raw stifle bends 72° out of its plane.

None of the joints had limits, which means none had ligaments.

**What was built.**
1. **Joints from the capture** ([body/anat.py](body/anat.py) → [body/anat.json](body/anat.json), [body/dog_anat.xml](body/dog_anat.xml)). For each joint:
   - **Rest:** the dog's mean standing rotation.
   - **Axes:** the principal axes of its motion, main swing first.
   - **Twist, settled by anatomy:**
     - markers fix where a bone points better than how it is turned, and in this body the turn moves nothing but the plane of the next hinge;
     - so the humerus and femur turn until the elbow and stifle bend in their own plane, the same way as when standing;
     - the tibia may turn up to 15°;
     - bones with no hinge below turn as little as possible;
     - the thorax's roll and the head's tilt are kept as captured.
   - **Ranges, the ligaments:** the capture's extent plus a margin:
     - main swing: 0.1–99.9 percentile, +15°;
     - out-of-plane hinges: 1–99 percentile, +10°.

   **Results:**
   - It is exactly the same motion: every body position matches to 0.00 mm.
   - **Single-axis share of each joint's motion:** elbows 98–100%, stifles 92%, carpus 83–94%, hocks 89–95%.
   - **Out-of-plane motion overall:** half that of the raw description.
   - **Excluded clip:** D1_ex03_KAN02_012 is left out. Its right elbow bends backwards for 22 s of ordinary walking, a fault in that clip.
2. **Muscles** ([body/train/dog_track.py](body/train/dog_track.py), `config.muscles`). At every physics step:
   - the policy's target reaches the muscle through an activation lag (the brain's signal);
   - the limb's own stiffness acts at once about that lagged target (the local reflex);
   - the torque is capped at the joint's strength, and the cap falls as the joint turns faster in the pull's direction (Hill);
   - the ligaments take whatever the muscles don't hold.

   Checked against the old motors: run through this code at the old strength, a body with no policy at all lasts 61.0 control steps against 61.4, with nothing non-finite and the same speed.
3. **Drive sense.** The policy sees where each muscle is aiming and how hard it pulls, relative to its strength: "actuation is itself a form of knowing where it is".
4. **References in the new joints** (runs/refs_anat).
   - **Held at a range:**
     - capture: 0.22% of hinge-frames, moving a paw more than 2 cm in 2.9% of frames;
     - motion-matched: 0.21%;
     - grammar sentences: 0.19%.
   - **Link test:** its 576 references are re-blended in the anatomical joints ([body/make_links.py](body/make_links.py) `--anat`), so "straight across" moves each joint along its own axes, inside its range.

**The test, fixed before any result.**
- **Runs:**
  - `anat_strong`: anatomical body with the old strength (instant motors, ±80 N·m limbs and ±200 N·m trunk, on every hinge). Trained from scratch for 600M steps on the capture, motion-matched and grammar references (196).
  - `anat_dog`: the same run's 300M-step checkpoint, continued for 300M steps on the same references with a dog's muscles (`body/muscles_dog.json`; table below, fixed before the run starts). Same hyperparameters, so both have the same total practice.
- **Evaluation** (both at 600M, reference anchored at 0.1, ending only on a real fall):
  - the capture;
  - the 24 held-out sentences;
  - the 576-reference link test.
- **Predictions:**
  - **P1, validity:** `anat_dog` tracks the capture with at least 75% of episodes complete. A dog did all of it. If this fails, the muscle table is wrong somewhere (the joints at their limit will say where), and P2–P4 can't be read.
  - **P2, the claim:** on `anat_dog`, forced changes the words forbid fall more often than forced changes they allow (one-sided Fisher, p < 0.05, on the regimes the policy handles: stand, walk, trot, sit, down, crouch), while routed changes show no such gap.
  - **P3, control:** on `anat_strong` that gap is smaller than on `anat_dog`.
  - **P4, strain:** on `anat_dog`, ligament strain after the ask is higher when the words forbid a forced change than when they allow it (one-sided Mann–Whitney, p < 0.05). This is a physical signature even where the dog doesn't fall.
- **Added at 16:04, before `anat_dog`'s link results existed.**
  - **Why:** evaluating `anat_strong` twice with the same policy and references changed the outcome of 102 of the 576 references. The physics on the GPU isn't exactly reproducible, and the forced "forbid > allow" p for all regimes moved from 0.12 to 0.03 between the two passes. A single pass is one sample of each reference's outcome.
  - **Second problem:** the pre-registered episode-level tests count the 4 references of a link (same two postures, same mode) as independent trials, which they aren't.
  - **The fix:** both runs repeat the link test with 8 start-noise seeds ([dog/link_test_repeated.py](dog/link_test_repeated.py)). Each link's fall rate pools its episodes. Forbid vs allow is tested by permuting the words' verdict across the 30 competent links (one-sided, the pre-registered direction). P3 is paired by link: the change in fall rate from `anat_strong` to `anat_dog`, compared between forbid and allow links.
  - **How it is reported:** the pre-registered single-pass tests (seed 0) as written, and next to them this repeated, link-level judgement, which is the one to believe.
  - **Also added:** whether a dog that stays up actually followed the ask (its pose match over its last second). Without it, a dog that quietly refuses a forced change can't be told from one that does it.

**The dog's muscles, fixed before the `anat_dog` run** ([body/dog_strength.json](body/dog_strength.json) → [body/muscles_dog.json](body/muscles_dog.json)). Strength is each joint's main swing, set against the old body. Out-of-plane hinges get a share of it:
- at the ball joints (hip, shoulder, shoulder blade, spine, neck, head): 50% for the side hinge and 30% for the twist hinge;
- at the hinge joints (elbow, carpus, stifle, hock): 20% for each, because the ligaments hold them there.

| joint | strength (N·m) | old body | basis |
|---|---|---|---|
| hip | 56 | 200 | measured: 1.4 × 40 N·m, the peak at a high-acceleration gallop in a 28.5 kg greyhound (Williams et al. 2009, J Exp Biol 212:550, Fig. 9) |
| stifle | 35 | 80 | measured: 1.4 × 25 (same study) |
| hock | 63 | 80 | measured: 1.4 × 45 (same study). Checked against muscle data: the gastrocnemius alone, at 925 N on a roughly 4.5 cm lever, gives about 42 N·m (Ellis et al. 2018, Front Bioeng Biotechnol 6:162, Table 2) |
| shoulder | 56 | 200 | estimate: as the hip |
| elbow | 40 | 80 | estimate: between stifle and hock (the triceps is large) |
| carpus | 30 | 80 | estimate: below the hock (palmar ligaments and flexor tendons carry most of stance) |
| shoulder blade (the sling on the thorax) | 80 | 200 | estimate: carries about 60% of the body weight standing |
| lumbar and thoracic spine | 100 | 200 | estimate |
| neck / head | 30 / 12 | 80 / 80 | estimate |

No inverse dynamics of a dog's forelimb, spine or neck at maximal effort were found, so those rows are estimates and labelled as such.

**Other settings:**
- **Activation lag:** 25 ms, between the standard 10 ms activation and 40 ms deactivation (Zajac 1989), since a target move activates one muscle of a pair and relaxes the other.
- **Speed:**
  - No per-joint gallop speeds are published, and real gallops ride on tendon recoil, which this body lacks.
  - So each joint's zero-force speed is 3× the fastest the capture's dog reached, ranging from 17 rad/s (shoulder blade) to 90 rad/s (shoulder). A dog did those speeds, so the body must be able to.
  - At the dog's own top speed its muscles keep 29% of their strength.

### The result

**Tracking:**

| | `anat_strong` | `anat_dog` |
|---|---|---|
| capture, episodes complete (P1) | 95.3% | **93.5%**: pass (needs ≥ 75%) |
| 24 held-out sentences, complete | 94.4% | 91.8% |
| muscles at their limit | (no limit) | 1.4% of physics steps |
| ligament strain per step, capture | 0.00100 | 0.00105 |
| link test, all 576 references: fell | 39% | 49% |

A dog's strength costs very little tracking. Strength binds in a few changes, not in ordinary motion.

**Pre-registered single pass** (seed 0, episodes as trials, competent subset):
- **P2:** on `anat_dog`, forced changes the words forbid fell 9/62 (15%) against 1/53 (2%) for those they allow, Fisher p = 0.016: **pass as written**. Routed: 6/62 (10%) against 1/55 (2%), p = 0.078. No significant gap, but not zero either.
- **P3:** the gap is 6.2 points on `anat_strong` against 12.6 on `anat_dog`: holds as a point estimate.
- **P4:** ligament strain after a forced ask, forbid against allow, median 0.0017 against 0.0009 per step, p = 0.0025: **pass as written**. But `anat_strong` shows the same (p = 0.0068), so it's a property of those links, not of the dog's muscles.

**Repeated and judged per link** (8 seeds, 30 links: the judgement to believe):

| | `anat_strong` | `anat_dog` |
|---|---|---|
| forced, words forbid: fell | 6.4% | 12.3% |
| forced, words allow: fell | 1.6% | 2.1% |
| gap, points (95% CI over links) | 4.9 (−2.9 to 15.8) | 10.2 (−0.5 to 23.0) |
| permutation p | 0.21 | 0.10 |
| routed: gap, p | 2.9, p = 0.31 | 5.8, p = 0.14 |
| on forbid links, forced minus routed | 1.8 points, p = 0.25 | 4.7 points, p = 0.19 |
| ligament strain after a forced ask, forbid vs allow | 0.0011 vs 0.0003, p = 0.065 | 0.0052 vs 0.0024, p = 0.066 |

- **P3, paired by link:** `anat_dog`'s gap is 5.3 points larger than `anat_strong`'s (CI −1.1 to 14.7), p = 0.28.
- **Strain:** at dog strength the ligaments carry about 4× as much after a forced ask. "The muscles protect the ligaments", and weaker muscles protect them less.

**Where the falls are** (forced, 32 episodes per link):

| link | the words | old strength | dog strength | dog, routed |
|---|---|---|---|---|
| crouch → trot | forbid | 0% | **38%** | **0%** |
| sit → crouch | forbid | 0% | **59%** | 53% |
| down → trot | forbid | 75% | 75% | 38% |
| down → walk | forbid | 25% | 25% | 25% |
| down → stand | allow | 22% | 25% | 25% |
| the other 25 links | | 0–3% | 0–5% | 0–6% |

- **crouch → trot:** at dog strength, 38% forced against 0% routed and 0% at the old strength. I first wrote this up as your sentence in physics ("they will 'fall' if over commanded"). The check below shows most of it was lag, not falling.
- **Rising from lying down** looked like the tracker's weakest skill: an allowed change (down → stand) falls 25% at either strength, routed or not. Per reference it is one start of four, failing on every seed. Whether it falls or lags is answered below.

**What a "fall" was (validity check, after the result).** The pre-registered fall is relative to the reference: the trunk tipped 46° off the reference's tilt, or the hips 0.2 m below the reference's. When the reference rises fast and the body lags, that counts as a fall although the dog is lying safely. down → walk's first reference "falls" 0.16 s after the ask, too soon to topple.

I rescored with an absolute fall instead: the thorax *rolled* more than 60° about the long axis, onto its side or back (`--term topple`).
- **Why roll and not tilt:** a sitting dog's pelvis tips 97° (median) and a bow pitches the thorax past 60°, but neither rolls. In the capture, roll past 60° occurs in 0% of stand, walk, trot, run, jump, sit and down frames, 5% of crouch frames and 82% of side frames.
- **One caveat:** sit → crouch "topples" 88–100% in both bodies, forced or routed. That crouch is a half-rolled lie (its reference rolls up to 61°), so a tracking dog crosses the threshold. It is a borderline case, not a finding.

| per link, 8 seeds | `anat_strong` | `anat_dog` |
|---|---|---|
| forced: words forbid vs allow | 13.9% vs 2.7% (gap 11.3, p = 0.063) | 14.3% vs 4.6% (gap 9.6, p = 0.16) |
| routed: words forbid vs allow | 12.3% vs 3.0% (gap 9.2, p = 0.12) | 11.7% vs 4.9% (gap 6.8, p = 0.24) |
| forced minus routed, forbid links | 1.7 points (p = 0.30) | 2.5 points (p = 0.06) |
| crouch → trot, forced / routed | 4% / 30% | **6% / 0%** |

- **Dog strength:** its gap is no larger than the old strength's (paired difference −1.7 points, p = 0.70).

**Verdict:**
- P1 passes.
- The pre-registered single-pass tests of P2 and P4 pass, and P3 holds as a point estimate.
- **Judged per link over 8 seeds, the claim as stated is not supported.** It says the words predict falls when a change is forced and not when it is routed. What holds under both definitions of a fall is narrower:
  - the links the words forbid are harder for this body, by about 10 points, forced or routed, at either strength (p 0.06–0.24 per link, 30 links);
  - routing barely protects;
  - the dog-strength doubling of the gap appears only under the relative definition, which counts lagging as falling.
- **Where the falls actually are:** a few start poses fail on every seed and in both modes. Those are rises from lying down (down → trot, down → walk, down → stand), not the words' missing links.

GR5 stays "not yet". The words mark hard changes; whether over-commanding makes this body fall where routing wouldn't is not shown.

**Where a dog's strength binds** ([body/train/diag_muscles.py](body/train/diag_muscles.py), the capture). The share of physics steps each muscle spends at its limit while the dog is up:
- elbow side-bend (ForeArm a2): 9.1–9.5%;
- stifle twist (Leg a3): 7.6–9.2%;
- hip twist (UpLeg a3): 3.6–4.5%;
- carpus side-bend: 1.2–3.6%.

No main-swing muscle is in the top 15. So the binding muscles sit on the out-of-plane hinges of the hinge joints, where a real dog has collateral ligaments rather than muscles: your "near zero out of plane motion until broken". In this body those hinges are weakly muscled (20% of the main swing). Their ranges are the capture's spread plus 10°, and an elbow's spread is almost zero, so the policy spends effort holding slack that a ligament would simply take.

**The ligament body (fixed at 16:25, before any result).** [body/make_lig_body.py](body/make_lig_body.py) writes `dog_anat_lig.xml` and `muscles_dog_lig.json`. At the elbow, carpus, stifle and hock, each out-of-plane hinge gets:
- a range of the capture's spread plus 2° instead of 10° (the ligament's slack). The elbow is now held to ±2°; the others keep their captured spread.
- a spring toward standing (30 N·m/rad) and damping: the ligament's give inside the slack.
- its muscle reduced to nothing.

The actuator stays, so `anat_dog`'s policy can continue on this body (`anat_lig`: 150M steps, same references and settings).

Predictions:
- **L1:** the capture is at least 91.5% complete (`anat_dog`: 93.5%).
- **L2:** the remaining muscles are at their limit less often than `anat_dog`'s 1.4% of physics steps.

The link and rush tests are reported against `anat_dog`, without predictions.

**Result** (`anat_lig`, 150M steps on the ligament body):
- **L1 passes:** the capture is 92.3% complete (`anat_dog` 93.5%), and the held-out sentences 90.7% (91.8%).
- **L2 passes as written:** the 38 muscles left are at their limit 0.78% of physics steps, against the 1.4% fixed beforehand.
  - The fair comparison is the same 38 muscles in `anat_dog`: 0.79%. So the remaining muscles work exactly as before.
  - What changed is that the 16 out-of-plane muscles that bound most are gone, and the ligaments hold those planes, as in a dog.
- **Ligament strain per step** rises from 0.0011 to 0.0101. That's by design, not a cost: the ranges are now the capture's spread plus 2°, and the joints rest against them.
- **Link test** (8 seeds, per link; no prediction): forced changes the words forbid fell 12.3% against 2.3% allowed (gap 10.0 points, p = 0.027). Under real topples: 16.1% against 2.6% (p = 0.051).
  - The gap is the same size as `anat_dog`'s (paired difference −0.2 points). Only its spread across links is tighter.
  - This is the first per-link p below 0.05, but it is one of many tests run tonight and wasn't predicted, so it doesn't settle GR5.

The ligament body tracks as well as the muscled one, with the out-of-plane planes held by ligaments as your notes describe. It is the more faithful body to carry forward.

### Tired: fatigue as a determiner of falls (the test, fixed at 16:10, before any result)
Your notes say "blood and breath are determiners". `brain.muscle_capacity()` already scales every muscle's strength online: fatigue, being winded, arousal. A tired dog has less margin, so more changes sit near the edge of what it can do, which is where the words should start to matter.

**Runs:**
- **`anat_tired`:** `anat_dog` continued for 150M steps with each episode's capacity drawn from 0.55–1.1. It isn't told its capacity; it can only feel it through its drive sense ([body/wsl_anat_tired.sh](body/wsl_anat_tired.sh)).
- **`anat_dog` at reduced capacity:** a dog that never practised tired. Reported alongside, with no predictions.

**Evaluation:** the link test (8 seeds) at capacity 1.0 and 0.65 (full fatigue in `brain.muscle_capacity()`), and the capture at both.

**Predictions for `anat_tired`:**
- **T1, validity:** the capture is at least 75% complete at capacity 1.0, and at least 60% at 0.65.
- **T2, the claim:** at 0.65, forced changes the words forbid fall more often than those they allow (per link, one-sided permutation p < 0.05).
- **T3, fatigue × words:** going from 1.0 to 0.65, the forced fall rate rises more on forbid links than on allow links (paired by link, one-sided permutation p < 0.05).
- **T4, routing protects:** at 0.65, on forbid links, routed changes fall less often than forced ones (paired by link, one-sided sign-flip p < 0.05).
- **Added at 16:35, before `anat_tired` finished:** T2–T4 are judged with the fall as fixed (relative to the reference), and reported alongside under the absolute topple.

**Result** (`anat_tired`, 150M steps with capacity drawn from 0.55–1.1; 8 seeds, per link):

| | fall as fixed (relative) | topple (thorax rolled), reported alongside |
|---|---|---|
| **T1** capture complete, capacity 1.0 / 0.65 | **92.5% / 90.0%: pass** (needs 75% / 60%) | |
| **T2** tired, forced: forbid vs allow | 19.5% vs 7.7%, gap 11.8, p = 0.13: **fails** | 21.5% vs 6.9%, gap 14.6, p = 0.059 |
| **T3** gap widening, 1.0 → 0.65, forced | +2.6 points, p = 0.27: **fails** | +4.7 points, p = 0.095 |
| same, routed (no prediction) | +2.9 points, p = 0.30 | **+8.2 points, p = 0.006** |
| **T4** tired, forbid links, forced minus routed | 3.5 points, p = 0.20: **fails** | 3.4 points, p = 0.20 |

- **Practising tired costs little and buys a lot.** Fresh, it tracks as well as before (92.5% against 93.5%). At 0.65 it completes 90.0% against 89.4% for the dog that never practised tired, with less ligament strain (0.0018 against 0.0022 per step).
- **The words under fatigue:** being tired makes the changes the words forbid disproportionately harder, by every measure. Under real topples that is clearest for routed changes (p = 0.006, an exploratory test among many). Routing still doesn't protect.
- **So the words mark changes that ask more of the body,** and fatigue exposes it. What the words do not do here is predict falls that routing avoids. T2–T4 fail as written.

**A dog that never practised tired** (`anat_dog` at capacity 0.65; reported with no predictions; scored with the pre-registered relative fall, which counts lagging as falling, so read with the validity check above):
- **Capture:** 89.4% complete (93.5% fresh). Muscles at their limit 4.7% of physics steps (1.4% fresh). Ligament strain doubled.
- **Link test, per link:** forced changes the words forbid fell 21.0%, those they allow 10.1% (gap 11.0 points, p = 0.14). Falls rose everywhere, and the gap didn't grow (paired p = 0.45).
- **Where:**

  | link | the words | forced, fresh | forced, tired | routed, tired |
  |---|---|---|---|---|
  | crouch → trot | forbid | 38% | **96%** | **0%** |
  | sit → crouch | forbid | 59% | 91% | 94% |
  | walk → trot | allow | 0% | **50%** | 6% |
  | stand → trot | allow | 0% | 28% | 0% |
  | sit → stand | allow | 0% | 25% | 25% |
  | trot → sit | forbid | 0% | 25% | 25% |

- **Rescored with real topples (thorax rolled onto its side or back), per link, 8 seeds:**

  | `anat_dog` | forced: forbid vs allow | routed: forbid vs allow |
  |---|---|---|
  | fresh | 14.3% vs 4.6% (gap 9.6, p = 0.16) | 11.7% vs 4.9% (gap 6.8, p = 0.24) |
  | tired (0.65) | 25.1% vs 12.4% (gap 12.6, p = 0.11) | 20.2% vs 8.0% (gap 12.2, p = 0.085) |

  Being tired roughly doubles toppling everywhere, and the words' gap persists, forced or routed. Routing protects little (4.9 points on forbid links, 4.5 on allow).
- **What this shows:** "forced" in this test means two things at once:
  - no path through the postures between, which is what the words speak to;
  - and *now*: the matcher jumps to the new regime and blends in 0.2 s.
- Tired, the rush alone fells the dog on changes the words allow (walk → trot). A tired dog asked to spring from a crouch into a trot falls nearly every time; through standing it never does.

**The rush test (fixed at 16:15, before any result).**
- **Design:** the same change from the same start, asked two ways ([body/make_links.py](body/make_links.py) `--rush`): forced (0.2 s blend) and eased (1 s blend). The 30 competent links × 4 starts each. `anat_dog` at capacity 1.0 and 0.65, and `anat_tired` at 0.65, with 8 seeds each.
- **R1, the rush:** at 0.65, on links the words allow, eased changes fall less often than forced (paired by link, one-sided sign-flip p < 0.05).
- **R2, the words:** eased, at 0.65, links the words forbid fall more often than links they allow (per link, one-sided permutation p < 0.05). With the rush removed, what remains is the words' "no path".

**Result: R1 fails, and the test is invalid.** Eased changes fell far *more* often than forced ones, fresh and tired:

| `anat_dog` | forced, forbid | forced, allow | eased, forbid | eased, allow |
|---|---|---|---|---|
| fresh | 7.9% | 1.8% | 42.3% | 20.1% |
| tired (0.65) | 10.4% | 9.2% | 52.1% | 34.4% |

- **Why the manipulation failed:** a 1 s inertialization blend moves the joints and the root on separate schedules, so the eased references aren't physically consistent:
  - worst tenth of eased references: a paw 5 cm into the floor (forced: 0 cm), and all four paws 9 cm off it (forced: 6 cm);
  - inertialization is meant for a few frames, not a second.
- **So the test confounds slowness with a broken reference.** R2 can't be read either.
- **What still stands:** the per-link evidence above, where a tired dog rushed from walk into trot falls 50% forced and 6% routed. The routed mode is the physically consistent slow change, since it plays a transition the capture contains.

### Getting up: `scramble_up`, a reflex of its own (GR9)
The lexicon, written from text on 19 November, already had it:
- **posture graph:** "fall → stand: scramble up, often followed by shake_off_dry";
- **`scramble_up`:** class R, a reflex (triggered, never chosen), "getting the feet back fast after a fall", written as trunk twist plus legs go, fast.

The capture shows a dog rolling off its side only five times (all side → belly), so a tracker can't learn getting up from it. A reflex needs no reference. In the anatomical joints every angle at zero is the standing pose, so `scramble_up` ([body/train/dog_right.py](body/train/dog_right.py)) aims its muscles at offsets from standing and learns the scramble:
- **Starts:** the dog is dropped in a fallen orientation with its limbs in a captured pose.
- **Reward:** upright × hips height, paws down, poses a dog has (nearest captured pose), little ligament strain, little effort. The sooner it is up, the more it collects.
- **Network:** small (256-128).

In the online loop ([body/train/online_dog.py](body/train/online_dog.py) `--reflex`), the fall detector hands it the body. When its feet are under it, the matcher resumes from the pose the body is actually in. If it is only righted onto its belly, the planned rise takes over (belly → stand, as the capture does it).

**How fast a real dog is, in the capture:**
- **Side to belly:** 1.1 and 1.3 s, in the two clean rolls.
- **Belly to feet:** a median of 0.27 s (0.22–0.34 s, 6 rises).

**What the reflex does: first test, corrected.** The first test of "on its feet" looked only at the pelvis: its tilt within 35° of standing and the hips at 75% of standing height. The video ([dog/scramble_up_compare.mp4](dog/scramble_up_compare.mp4)) showed both reflexes passing it in postures no dog stands in: hips up, front end curled down. I first reported 100% within 0.14 s for the old motors, and 75% and 63% for the dog's muscles. Those numbers were the loophole.

The proper standard has three parts:
- pelvis and thorax both within 35° of standing;
- hips and withers both at 75% of standing height or more;
- at least three paws on the floor.

Re-scored on it (2,048 falls; times are medians; both policies trained on the loose reward):

| how it started | old motors (40M steps): on its feet | dog's muscles (80M steps): on its feet within 3 s |
|---|---|---|
| stumbling (< 45°) | 95.6%, 0.10 s | 93.9%, 0.10 s |
| on its side (45–135°) | 94.5%, 0.70 s | **19%**, 1.56 s |
| on its back (> 135°) | 99.3%, 0.70 s | **4%**, 1.57 s |

**Properly measured, the old motors right the dog from its back in 0.7 s, 99% of the time.** That is fast but not impossible: the "0.14 s" was a contorted pose passing the loose test. **The dog-strength reflex had mostly learned to game the loose test.** Both are being retrained on the reward the proper standard implies:
- each of pelvis and thorax rewarded for uprightness × height;
- a stronger pull toward poses a dog actually has.

**Retrained on the proper reward: the dog's muscles now get up** (`scramble_up_dog3` at 72M steps, after 80M loose + 108M strict). Tested on 2,048 falls against the proper standard ([body/train/diag_right.py](body/train/diag_right.py)). Two checks are added: does it stay up after first getting there, and is it up at 3 s?

| how it started | dog's muscles: on its feet | stays up afterwards | up at 3 s | old motors: on its feet | stays up | up at 3 s |
|---|---|---|---|---|---|---|
| stumbling (< 45°) | 100%, 0.10 s | 99% | 100% | 95%, 0.10 s | 87% | 87% |
| on its side | **100%, 0.44 s** | 94% | 100% | 94%, 0.70 s | 84% | 88% |
| on its back | **100%, 0.46 s** | 94% | 100% | 99%, 0.70 s | 86% | 93% |

- **Better than the old motors on every measure.** Not by more strength: a reward that only counts a dog standing as a dog stands.
- **The video** ([dog/scramble_up_strict.mp4](dog/scramble_up_strict.mp4), old motors left, dog's muscles right, same five falls):
  - From its back, the dog-strength reflex rolls onto its side, then its belly, gets its legs under it and pushes up, all in about half a second, and ends standing square with its head up.
  - The old motors flip the body up in one jerk and stand tall and oddly.
- **Faster than the capture's dog.** The capture's two clean rolls took 1.1–1.3 s from side to belly; this reflex takes about 0.4 s. Those were relaxed rolls, and this is a scramble. Whether a startled dog is this quick is a question for footage of one.
- **Tired** (the brain's capacity 0.65 on every muscle; the reflex never practised tired, and the reflex environment now carries the capacity as the tracker's does):
  - on its feet from its side 99.8% and from its back 100%, in 0.58 s instead of 0.44 s;
  - still up at 3 s 99% of the time.

  A tired dog gets up more slowly, with no retraining: blood and breath set how fast.
- **On the ligament body** (`scramble_up_lig`: 40M more steps on `dog_anat_lig.xml`, with capacity drawn from 0.55–1.1):
  - fresh: 100% on its feet from its side or back in 0.44 s, still up at 3 s 100%;
  - tired (0.65): 100%, in 0.54–0.56 s, staying up 99.5% or more.

  This is the reflex to carry forward with the ligament body.

**Online: the spoken-words runs on the dog-strength body, with the reflex** (`anat_dog` + `scramble_up_dog3` at 72M; same words, seeds and starts as before; [body/wsl_online_words.sh](body/wsl_online_words.sh)).

**A real fall, not a lag.** The first replay showed the online fall detector had the link test's flaw: it measures the body against the reference, so a dog that can't keep up with a jump "falls" (tilt 0.15 rad, hips 0.2 m under a reference in the air) and the reflex grabs a dog that hasn't fallen. Now the body says why it is off the plan:
- **Falling:** the thorax rolled past 60°, or the hips dropped 0.2 m within 0.5 s. The reflex takes the body.
- **Lagging:** anything else. The plan re-syncs to the captured frame nearest the body and asks again from there.

| | real falls | back on its feet | lags re-planned |
|---|---|---|---|
| Malinois, 56 s | 2: both asked to jump (insisted from platz at 11 s; asked again at 38 s) | 2 of 2, in 1.1 s and 0.9 s | 1 (the first jump, 0.5 s before it really fell) |
| Lab, 60 s | 1: walking to the beer | 1 of 1, in 0.7 s | 0 |

**The insisted jump** ([dog/words_dog_strength_reflex.mp4](dog/words_dog_strength_reflex.mp4), Malinois on the right):
- **11.0 s:** in platz, the person says "jump!" insistently, and the brain takes the 12-bit link straight across.
- **12.2–12.6 s:** she springs straight up out of the down and rears.
- **12.7 s:** she can't keep up with the jump: a lag, re-planned, not a fall.
- **13.2 s:** she tips over, a real fall.
- **13.4–14.2 s:** the reflex rolls her under and stands her up. Breathing goes from 38 to 43 a minute and arousal from 1.0 to 1.4.
- **14.3 s:** she's standing, and the plan has the body again.

This is your "they will 'fall' if over commanded, not interpolate", followed by the lexicon's own "scramble up".

**GR9 passes:**
- **Physics:** 100% on its feet from its side or back on the proper standard, and still up at 3 s.
- **Online:** 3 of 3 real falls ended back on its feet within about a second, and the plan resumed.

- **How long it stays up:** the same at dog strength as at the old strength (62.6 vs 62.6 control steps).
- **Muscles at their limit:** only 1% of physics steps when quiet. Strength binds in fast and explosive motion, not in holding still.
- **Ligament strain:** 4.5× as much, as the weaker out-of-plane muscles let joints settle onto their ligaments.

## 2026-11-21: words to bodies, the dog's grammar

**What was asked:**
- Determine which words and strings of words name which actions, reflexive or planned.
- String them into new motions and examples.
- Price how unlikely each link is ("platz jump" against "go to the fridge door, open it by tugging the rope on the door handle, and fetch me a beer").
- Let the networks run on that.
- Do it in three modes: alone (endemic), guided by a person, and with a person present.

**What was built:** [dog/grammar.py](dog/grammar.py) compiles the lexicon ([dog-lexicon/](dog-lexicon/), written 19 Nov from text, before any data) into [dog/grammar.json](dog/grammar.json):
- **Entries:** 316 bodies, of which 84 are reflexive (R, X: triggered, never chosen), 38 planned (D, O, L) and 194 states.
- **Structure:** 79 requests with their accept-sets, 39 sequences (one step still unresolved: "send cue"), and the 28-edge posture graph.
- **Pricing:** every link is priced in bits from separate sources:
  - **body:** the words' posture graph, weighted by the capture;
  - **habit:** each dog's footage, as surprise relative to any dog;
  - **meaning:** sequences and drives;
  - **skill:** a trained cue, or a body put to a use it doesn't serve, such as a tug that opens a door;
  - **state:** hot, tired, keyed up.
- **Routes and readings:** a change the body has no link for is routed through the postures between; forced, it falls. Ambiguous words are settled by price.

One lexicon fix came from the README's own rule: shake-off now carries the word "shake".

### The words' posture graph against data (`grammar.py audit`, and the stricter cut below)
| changes counted | words allow | capture (41 min, 50 Hz) covered | your two dogs (767 changes, 1 Hz) covered | random graphs of the same size |
|---|---|---|---|---|
| all | 37 of 110 (34%) | **78%** (p=0.004) | **80%** (p=0.0002) | 33–34% median |
| postures only (no stand↔gait) | 25 of 98 | **50%** (p=0.04) | **65%** (p=0.0002) | 24–25% |
| postures only, without crouch | 23 of 78 | **60%** (p=0.03) | **69%** (p=0.0002) | 28–29% |

**The words carry real knowledge of how a dog's body connects postures, but not all of it.**
- **Pair level:** of the 37 changes the words allow, 35 were seen in one source or the other.
- **What the words miss:**
  - walking straight off from a sit or a down: capture 26 and 5, your dogs 8 and 15;
  - the in-between low stance the capture calls crouch;
  - side→stand in one second (your dogs, 8).
  These are "the moments between" that the first notes (19 Nov) said language is thin on.

The grammar plans on words and capture together, and every edge records which source gave it. The Gemini labels see 4 seconds per call and could lean toward plausible sequences. The capture's labels come from kinematics alone, so the capture column is the clean test.

### The two examples, priced (`grammar.py price`)
| said | price | where the bits are |
|---|---|---|
| "platz jump" (the Malinois, in platz) | 12.7 bits in one link | No direct link in the words or the capture (the Malinois's footage has it twice at 1 Hz, springing up in play). Routed platz → run → jump: body 6.4 bits + her habit 5.3. Forced straight across, it falls |
| "go to the fridge door, open it by tugging the rope on the door handle, and fetch me a beer" (the Lab, lying) | 26 bits over 8 links, about 3 per link | One dear link: tug as the means to open a door, 9.6 bits (meaning 5: a tug doesn't follow a walk; skill 4: a play body put to work). Getting up and going is 4 bits; the fetch is cheap |
| "shake" | wet: shake-off 1.5 bits; dry: give-paw 5.3 | Settled by price; a reflex asked for without its trigger costs 6 bits more |
| "down" | the hot Malinois sploots; keyed up, platz; the tired Lab rolls onto her hip | The dog picks among the accept-set by its body's state (10-requests.md: "the dog's pick shows its state") |

### Does the price predict what the physics dog can do? (`make_links.py`, `train/eval_links.py`, `dog/test_links.py`)
**Method:**
- For each of 72 ordered pairs of 9 regimes, 4 references each way. **Forced:** asked straight across; the matcher jumps and blends in 0.2 s. **Routed:** the matcher's planner takes the capture's own path.
- The best tracker (uni_mm2) attempts every one (anchor 0.1, ends on a real fall). 87 of the 576 fell before the change and were left out.

**Predictions, written before the run:**
1. Forced changes fall more than routed ones.
2. Forced changes the words forbid fall more.
3. Forced falls rise with the route's price.
4. Routed changes rarely fall.

**Result: all four predictions fail as stated.**

| | result |
|---|---|
| 1. forced vs routed | 36% vs 37%: no difference |
| 2. forced, words forbid vs allow | 39% vs 32% (p=0.28) |
| 3. forced fall rate vs route price | Spearman 0.14 (p=0.14) |
| 4. routed | 37% fall |

**What dominates is the policy's own weak skills.**
- Anything ending in a **run or jump** falls: forced 96–100%, routed 73–77%.
- Anything starting **on its side** falls when routed (81%): the side get-up is the known gap.
- **Forced side→stand and side→sit never fell, while routed always did.** The simulated dog popped from lying on its side to standing in a fraction of a second, which no 28 kg dog can do.

**The motors are the likely reason.** Each limb hinge has ±80 N·m and the spine ±200 N·m, three hinges per joint, with an instant PD response and no muscle activation delay. That is probably more than a real dog's limbs deliver; the true limits are to be taken from canine inverse-dynamics studies (trotting Labradors, Colborne et al. 2011; Headrick et al. 2014), not guessed. So the physics can muscle through changes a real dog would fall out of. **Until the body has a dog's strength, physics can't judge over-commanding**, and "falling as a valid regime" has nothing to bite on. Your first notes said as much: the muscles protect the ligaments, and those limits have to be known.

**A weak hint in the predicted direction.** On the regimes the policy handles (stand, walk, trot, sit, down, crouch):
- forced changes the words forbid fell 10 of 61 times, against 3 of 53 that the words allow (one-sided p=0.064);
- routed changes showed no difference (10/63 vs 8/55, p=0.52).
That is the pattern to expect if the words' missing links are physical.

**The strength test: the same policy and the same 576 references, with every motor's force limit cut to 60%** (`dog_mocap_mjx_strength60.xml`, `eval_links.py --xml`, `dog/compare_strength.py`). Regimes the policy handles:

| motor strength | forced: words forbid | forced: words allow | routed: words forbid | routed: words allow |
|---|---|---|---|---|
| full (±80 / ±200 N·m) | 10/61 (16%) | 3/53 (6%), p=0.064 | 10/63 (16%) | 8/55 (15%), p=0.52 |
| **60%** | **15/61 (25%)** | **3/53 (6%), p=0.005** | 12/63 (19%) | 8/55 (15%), p=0.35 |
| 40% | 16/61 (26%) | 6/53 (11%), p=0.037 | 17/63 (27%) | 9/55 (16%), p=0.12 (the body is now too weak for the policy everywhere: routed changes fall too) |

**The words' verdict predicts a fall only when the link is forced.** The same pairs routed show no such gap, so this isn't a matter of harder pairs. Weakening the body sharpens the effect: forced changes the words forbid go from 16% to 25% falls, while forced changes they allow stay at 6%.

The posture graph was written from text before any data. **It says which links a body can't take straight across, and a body with less spare strength shows it.** Forced versus routed overall still doesn't differ (all pairs: 37% vs 39% at 60%), because the policy's weak skills (runs, jumps, side get-ups) dominate both.

This was not retrained for the weaker body. The proper test is a tracker trained with dog-strength muscles (limits from canine inverse-dynamics tables, plus activation delay), rerun on the same references.

### New sentences as training data: the tracker practises the grammar's motion (uni_gram)
**Method:** fine-tuned from uni_mm2 for 150M steps (31 min) on the capture, the matcher library and the grammar library (refs/gram, 60 min of sentences, transition-weighted starts). Both trackers were then scored on identical sets, anchored, with episodes ending only on a real fall.

| test set | uni_mm2 | **uni_gram** |
|---|---|---|
| **24 unseen grammar-sentence episodes** (seeds never trained on) | 252.5 steps, 82.5% full | **260.9 steps, 86.6% full** (run 50% → 62% full; jump 0 of 2 → 1 of 2) |
| the capture alone (57 parts) | **270.0, 89.0%** | 265.3, 85.5% (down, crouch and rear-up slipped; trot improved) |
| the 576 link references, all reaching the change | forced 36%, routed 37% fell | forced 31%, routed 34% |

**Practice on sentences no clip contains helps on unseen sentences (+4 points), at a small cost on the raw capture (−3.5 points).** The usual fine-tune trade-off.

**The same practice erased the words' verdict on forced changes:**
- forced changes the words forbid now fall 1 of 61 times (2%), against 3 of 53 that the words allow (6%);
- before practice, the figures were 16% against 6%.

With motors this strong, practice teaches the body to get through changes a real dog couldn't. **So the earlier hint was the old tracker's inexperience, not physics.** "Falling when over-commanded" can't rest on a policy being unpractised. It needs a body with a dog's strength: torque limits from canine inverse-dynamics tables and muscle activation delay, a tracker trained on it, and then this same test. That is now the deciding experiment for the words-and-physics claim.

### Does the grammar know your dogs before seeing them? (leave one clip out, 797 changes, 50 clips)
| model | surprise per change |
|---|---|
| uniform | 3.58 bits |
| **the grammar (words + capture; never saw your dogs)** | **2.97 bits** |
| Markov on the dog's own other clips | 2.51 bits |
| the dog's clips shrunk toward the grammar | 2.63–2.73 bits (worse) |

**Zero-shot, the grammar predicts your dogs' next regime 0.6 bits per change better than chance. Their own footage does better still,** and pulling their counts toward the grammar hurts slightly: the words miss changes your dogs make often (down→walk, walk→down). The grammar's job is to parse, route and compose; each dog's habit has to come from that dog.

### Each dog's signature, in the grammar's bits (habit: this dog's surprise relative to any dog; links seen 6+ times)
| the Lab takes it more readily | Lab | Malinois | | the Malinois takes it more readily | Lab | Malinois |
|---|---|---|---|---|---|---|
| side → back | −1.2 (27×) | +2.0 (0×) | | crouch → down | −0.2 (1×) | −2.0 (7×) |
| back → side | −0.7 (25×) | +0.5 (0×) | | down → walk | +0.3 (4×) | −0.7 (11×) |
| stand → side | −2.4 (4×) | −1.1 (2×) | | sit → walk | +1.6 (1×) | +0.1 (7×) |
| down → side | −0.8 (12×) | +0.6 (5×) | | crouch → run | −1.3 (2×) | −2.1 (6×) |
| walk → trot | −1.5 (7×) | −0.3 (4×) | | walk → sit | +0.1 (2×) | −0.8 (5×) |

**The Lab rolls: onto her side and back again, in play. The Malinois springs: out of a crouch into a run, and straight off a down or a sit into a walk.** Same words, two dogs; individuality as how readily each takes each link.

### Free speech, read by Gemini Carbon into the lexicon's own ids (`grammar.py price "..." --carbon`)
Carbon only maps the words. The prices are still the grammar's, and parses are cached in `dog/carbon_parses.json`.
- **"go to the fridge door, open it by tugging the rope on the door handle, and fetch me a beer":** walk → *open* (a change to the world) with tug as its means → retrieve, to you. Same reading and same 26 bits as the rule parser.
- **"could you go see what's in the kitchen and bring back whatever you find":** the *hunt* request → the *bring gift* sequence, to you.
- **"hey, settle down, lie on your mat and stay there":** settle → place → stay.

The rule parser handles the everyday cues: "drop it" → release, "off" → stand, "perch" → forepaws up, "go find it" → search. The first parse errors were fixed on the way:
- A sport's reading of a plain word ("stand" as the herding cue) had won because it asked less of the body. Now what was meant is settled before how hard it is: sense counts 4× body bits.
- A reflex asked for without its trigger now costs as much as a fall.

### Spoken words, online, in physics (`train/online_dog.py --brain mal --start down --present --say ...`, policy uni_mm2)
**The Malinois, 56 s, a person present:**
- **"Platz":** yes.
- **"Jump", insisted (11 s):** 12 bits, a 75% yes. Taken **straight across** from platz, and she **fell** 0.8 s later, tilting 0.81 rad. Over-commanded, the body fell instead of sliding.
- **Then:** the fall left her on her side, and every get-up failed (every 3.8 s, 11 times). This is the known open gap: rising from the side.
- **Her brain lived on through it:**
  - "come": a 99% yes;
  - "down": routed through stand, then platz;
  - a plain "jump": routed through a run, then the jump;
  - "sit";
  - her own social sentences: *middle* at 18 s and 36 s, *weave* at 53 s.

The body stayed on the floor. **Words, prices, final say and the over-commanded fall all work end to end; recovery is the bottleneck.**

**The Labrador, 60 s, a person present: no falls, 40 m walked.**
- **"Get the beer":** yes, though the chance was only 36% (hardest link 6 bits). She follows the gaze, walks, trots to the spot, then grab → carry back → release.
- **Her own idea at 25 s:** *bring gift* (grab, carry, present to the person).
- **"Go to the fridge door, open it by tugging the rope on the door handle, and fetch me a beer":** not now. Hardest link 10 bits (the rope tug as a door-opener), a 27% chance.
- **"Shake" while trotting:** walks, stands, sits, gives the paw.
- **"Down":** platz.

Her "get" ought to be a near-certain yes, since you call it sloppy-good. The 36% comes from placeholder numbers: her request bias of −0.3 and no per-dog cue reliabilities yet. Those are yours to set.

Video of both runs, with what each brain heard and did: [dog/words_physics.mp4](dog/words_physics.mp4).

## 2026-11-20, evening: the Carbon labels completed

### Labels and tracking
| step | result |
|---|---|
| Gemini Carbon teacher labels, every remaining clip (`label_gemini.py watch`) | **All 68 clips labelled: 2,895 labelled seconds (was 1,389)**, including all 6 minutes of the long play session (IMG_2650). 389 requests, no failures. Three boxes came back with a coordinate as text ("585"); the loader now reads them as numbers |
| Tracking (DeepLabCut) | **All 68 clips.** One transcode from 19 Nov was truncated (interrupted write); redone |

### Re-run on all labels (before → now; "before" re-scored with the same code on the earlier labels)
| what | before (32 clips) | now (67 clips) | verdict |
|---|---|---|---|
| Body regime on clips never seen (DINOv2 + linear head) | 54%, macro-F1 0.34 (majority 36%) | **59%, macro-F1 0.38** (majority 27%) | Better, against a harder baseline (the new clips are more varied) |
| Mouth and head, clips never seen | 66%, F1 0.41 (majority 38%) | 55%, F1 0.39 (majority 33%) | **Worse**: the new clips act differently (water, retrieves) |
| Activity, clips never seen | 65%, F1 0.32 (majority 55%) | **67%, F1 0.40** (majority 50%) | Better |
| Within labelled clips (filling the seconds between labels) | 76 / 81 / 86% | 75 / 79 / 84% | Unchanged |
| Which dog, DINOv2, clips never seen | 90.5%; 93.3% when sure | 90.5%; 92.5% when sure (94% of crops) | Unchanged. Hardest: IMG_9234 71%, the long wrestling session 75% |
| Which dog, the image classifier that names the tracks (ResNet-18), clips never seen | 80%; 86% when sure | **87.5%; 91.6% when sure (88% of crops)** | Better. Every clip now has teacher labels, where it scored 96% |
| Every tracked detection renamed with it (`dogid apply`), checked against the teacher's per-second visibility | | **65 of 68 clips agree.** Three lose a dog the teacher sees: IMG_0536 (the Malinois, seen 100% of seconds, named 3%), IMG_0895 (the Malinois, 71% → 0%), IMG_1122 (the Lab, 100% → 0%) | The brain reads the teacher's labels, not the tracks, so it is unaffected. Fix when tracks are needed: name each track by overlap with the teacher's boxes (every clip has them now) instead of by appearance |
| Routers, held-out transitions | 482 transitions. Markov −1.90; neural −3.50 | **797 transitions. Markov −1.76**; Markov + activity −1.93; neural −2.34 (still overfits) | More data helps the simple router most |
| Telling the dogs apart from their transitions alone | 50% (chance) | **60%** (80 sequences) | A first, weak signature |
| Per-second hazard router (2,895 s) | (1,892 s) | Time already spent in the regime is still the largest gain (−0.803 → −0.761 per second); the other dog −0.755; exertion −0.751; panting nothing | As before |
| **Generated rest vs real rest** (now 247 s of the Lab and 242 s of the Malinois at rest) | not measurable | Real dogs at rest lie down **94–96%** of the time. The learned generator alone: total variation 0.82 (Lab) and 0.68 (Malinois); the per-second hazard router: 0.46 and 0.53 (lying about half the time) | **The learned router can't do rest**: 8 transitions per dog in all that resting. The brain's designed rest priors stay in charge (trust 0.14) |
| Generated play vs real play | Lab 0.08, Malinois 0.11 | Lab 0.04, Malinois 0.11 | Play is right |
| The brain's router (`export_router` → `router_linear.json`) | play 146/137 transitions (trust 0.74/0.73), explore 7/26 | **play 269/272 (trust 0.84), explore 40/43 (0.44/0.46), retrieve 23/6** | The brain leans on the dogs themselves more, everywhere but rest |

### The fatigue test (`learn/fatigue.py`, IMG_2650: 6 minutes of wrestling, both dogs)
| dog | vigour by minute (running, trotting, jumping, rearing, tugging, mouthing) | trend | panting |
|---|---|---|---|
| Lab | 0.71, 0.47, **0.35**, 0.58, 0.62, 0.60 | −0.014/min (p 0.68) | peaks at 30% in minute 3, 20% in minute 6 |
| Malinois | 0.81, 0.65, 0.78, 0.67, 0.87, 0.87 | +0.017/min (p 0.51) | about 0 |

**Neither dog slows over 6 minutes.** The Lab takes a breather in minute 3 (vigour 35%, panting 30%, rest 35%) and goes back to full play: a fast exertion loop (burst, pause, recover), not slow fatigue. The brain's designed fatigue (time constant 5 min) would already have shortened their bouts by about 15% by minute 6 and never lets a dog recover while playing. **Changed: fatigue's time constant is now 20 min** (`dog/brain.py`). Open, for you to decide: should fast exertion trigger short breathers mid-play, as the Lab's minute 3 suggests? One session per dog is thin evidence; it points one way.

## 2026-11-20

### Machine and pipeline
| change | effect |
|---|---|
| Transcoding moved entirely to the GPU video engines (NVDEC, scale_cuda with conversion to 8-bit nv12, NVENC). Display rotation applied by hand on the scaled frames, and the rotation flag cleared | All 68 clips transcoded quickly. Output matches the earlier CPU transcodes: same orientation, mean pixel difference 11–14 against 77 for a wrong turn |
| iPhone footage is 10-bit HDR (HLG, BT.2020), which the H.264 encoder rejects | Fixed by converting to 8-bit inside the GPU scaler |
| Tracker (DeepLabCut SuperAnimal) batch 8 → 64, detector batch 8 → 32, TF32 on; CPU encoders no longer compete | About 14 frames/s, against 6 before |
| Footage work folder moved to `X:` behind a junction | C: stays free; paths unchanged |

### Naming the two dogs in each frame
| approach | result | verdict |
|---|---|---|
| Torso brightness against a per-clip median | 43% agreement with the teacher (worse than chance) | Failed: light changes and close play |
| Image classifier (ResNet-18 fine-tuned on 2,053 teacher crops), tested on clips it never saw | 80% (86% when confident) | Weak across clips: a backlit Lab at a window scored 27% |
| Same classifier, held-out seconds within clips | **96%; 97.6% when confident (95% of crops)** | Works wherever the teacher has labelled the clip. 36 clips await labels (Gemini quota) |
| DINOv2-base features (frozen) + linear head, clips it never saw | **90.7%; 94.1% when confident (88% of crops)** | **Best on new clips.** The backlit window clip goes from 27% to 85%. Weakest: the 6-minute wrestling session (76%), where the dogs overlap. Voting along each tracked path (a dog doesn't change identity mid-track) should lift it further: not yet tested |

### Recognising regimes (what the dog's body, mouth and attention are doing)
| approach | body regime | mouth and head | context | verdict |
|---|---|---|---|---|
| Majority class (baseline) | 36% | 38% | 55% | |
| Keypoints only, small temporal network, new clips | 24% | 37% | 30% | **Failed.** Handheld footage looks down at the dogs, and from above a standing dog's keypoints look like a lying dog's |
| Image network (ResNet-18, 1 frame), held-out seconds within clips | **73%** | **79%** | **84%** | Works for filling the seconds between teacher labels |
| Same network, clips it never saw | 39% | 44% | 54% | Doesn't generalise yet: 32 clips are too few |
| DINOv2-base features (frozen) + linear head, held-out seconds within clips | **76%** | **81%** | **86%** | Slightly better than the fine-tuned network |
| Same, clips it never saw | **54%** | **66%** | **65%** | **Works best on new clips.** Macro-F1 0.34 / 0.41 / 0.32: common classes carry it, rare ones (rear-up, swim, chew) still weak |
| Same with motion: frames 0.2 s before and after, plus the change across them, clips it never saw | 54% | 68% | 67% | **Doesn't help** (one frame in the same run: 55% / 66% / 67%; macro-F1 slightly worse). With 32 clips, four times the features only overfit. The limit is the number of labelled clips, not seeing motion: more Carbon labels are the lever |

### Routers (what comes next), trained on the teacher's per-second sequences
482 segments across both dogs, held-out clips, log-likelihood per transition (higher is better).

| router | log-likelihood | top-1 accuracy |
|---|---|---|
| Base rates | −2.19 | |
| Markov: next regime from the current one | −1.90 | 47% |
| Markov + context, full tables | −2.08 | Overfits |
| Markov + context, interpolated toward Markov | −2.06 | 47% |
| Neural router (small MLP with a learned per-dog vector) | −3.50 | 38%. Overfits badly at this data size |
| **Linear router: current regime, its duration, activity, the other dog's regime, distance, panting, which dog** | **−1.83** | 45%. Best so far |

**Telling the dogs apart from their transitions alone:** 50%, which is chance.

**Where they do differ is what they do.** The Lab spends far more time on her back and side (the underdog in wrestling); the Malinois stands and walks. When the router generates behaviour, play looks like them: the Lab rears up, rolls and runs, while the Malinois mostly stands (69%). Rest looks wrong: the generated dogs still walk a quarter of the time, because there are too few labelled rest seconds for the activity to steer the router.

### Physiology as a determiner, tested on the footage (`learn/physio.py`)
The router reworked to run per second, in two parts: **when** the current regime ends (a hazard) and **what** comes next. Physiology enters as two exertion integrators computed from the dog's own recent regimes: fast (breath and heart, tens of seconds) and slow (fatigue, minutes). Data: 1,892 labelled seconds, 381 regime switches, held-out clips.

| model (per second) | log-likelihood per second |
|---|---|
| Markov per dog (constant chance of switching) | −0.865 |
| Each dog's own regime durations (what the generator uses now) | −0.878 |
| Linear: regime, activity, dog | −0.854 |
| **+ time already spent in the regime** | **−0.799** |
| + the other dog | −0.798 |
| + panting seen | −0.800 |
| + exertion state | −0.795 (any time constant from 10 s to 20 min: the same) |

| question | result |
|---|---|
| Does the time already spent in a regime matter? | **Yes, the largest gain.** The chance of leaving a regime depends on how long the dog has been in it |
| Does recent exertion predict panting? | **No.** Held-out AUC 0.40, below chance: panting differs by clip (heat, play before the camera started), not by what the clip shows |
| Does exertion throttle sniffing? | The direction is right: sniffing is 28% at walking exertion and 3% at high exertion. The activity already explains it (AUC 0.97), and exertion adds nothing measurable |
| Does exertion help predict what comes next? | Barely (−0.798 → −0.795) |
| Does generated rest look like rest now? | **No.** There are only 90 labelled seconds of the Lab at rest and under 60 of the Malinois. Play is right with the current generator (off by 8% of time for the Lab, 11% for the Malinois) |

**Verdict:** physiology can't be learned from these clips. They are short (about 40 s), they start mid-activity, and what happened before the camera started drives breathing. Breath and blood stay designed determiners, as the first notes have them, built from canine physiology and then checked against the footage: panting after long play, play slowing over minutes. The test of play slowing over minutes needs the 6-minute play clip (IMG_2650), which is labelled only for its first 111 seconds: on hold for Carbon.

### Requests in the footage's audio (`learn/requests.py`)
Speech transcribed locally (faster-whisper; nothing left the machine), request words matched to the teacher's labels to measure, per dog, how often a request is followed by the asked-for change, and how fast.

| step | result |
|---|---|
| All 68 clips, small model with a speech filter | 296 words in total, speech in 22 clips. 16 request words in 7 clips, 3 of them in labelled clips |
| Six retrieve clips again, larger model, no filter | More words (39 in one clip that first gave none), still almost no requests: "okay", "here", "go" |
| Per-dog compliance | Not measurable: 4 usable events |

**Verdict:** these videos are filmed mostly in silence, so they can't teach the request layer. How each dog answers a request (how often, how fast, from which state) has to come from you, the person who knows them, and then be checked in the identity test.

### Physics body
| step | result |
|---|---|
| Dog model built from the 41-minute MANN capture skeleton (21 joints, 60 hinges, 28 kg) | Captures replay exactly; our own forward kinematics and MuJoCo's agree |
| WSL2 GPU stack | JAX 0.12.1 on the 3090, MuJoCo/MJX 4.0.1, Brax 0.14.2. Playground 1.0 dropped the old environment API, so the tracking environment is written directly against MJX and Brax |
| Smoke test (256 simulated dogs, random policy) | Runs. First version let a fallen dog sink through the floor; now every body part collides with the ground (falling is a real regime). 99% of dogs numerically sound in long uncontrolled falls; non-finite states now end the episode |
| Tracking references | Trot ×2, walk, sit, lie, gallop. Correction: renders show D1_003 is the sit and D1_005 is lying down then rising (the names were swapped earlier). The "gallop" (D1_010_KAN01_002) holds its paws 5–7 cm above the floor most of the time: suspect as a physics reference |
| The whole capture named in the footage's regimes (`catalog_mocap.py`, [MOCAP_CATALOG.md](body/MOCAP_CATALOG.md)) | Every frame labelled from hip height, trunk pitch and roll, speed, and paw contacts (diagonal pairs landing together means a trot). Checked against renders of six clips. 41 min (after the repair and model v2 below): stand 910 s, walk 650 s, sit 267 s, down 235 s, crouch 194 s, jump 75 s, run 52 s, side 52 s, trot 20 s, rear-up 3 s. **796 transitions of 42 kinds**, including all the first slice needs: lying down, getting up, sitting, rising, rolling onto the side. Missing: play wrestling (on the back, rearing), which is most of what your two do together |
| Regimes in, motion out: motion matching over the named capture (`motion_match.py`, the kinematic half of DReCon, Bergamin et al. 2019) | **Works.** Every 0.2 s the controller searches the whole capture for the frame that best continues the current pose toward the asked regime, speed and heading, then blends. A scripted first slice (lying, rolling onto the side, lying, asked to come, rising, walking to the person, sitting, lying down) plays the asked regime 90–96% of the time and ends within 1 m of the person. Its fast hinge motion (12% of frames) is below the raw capture's own rate while walking (21%): matching adds no pops. Getting up is a real captured rise (front end first, then hind legs, about 0.6 s), not a pop. Video: [first_slice_mm.mp4](body/first_slice_mm.mp4) |
| Composed motion as a physics reference: two faults found by checking speeds | (1) Per-clip unwrapping put the same pose a whole turn apart in different clips (10 joints; the tail drifts through several turns), so blending across clips spun joints: up to 377 rad/s. Fixed: each part's angles re-centred to the principal branch, and the matcher's output kept continuous with its previous frame. (2) Overlapping jumps restarted the blend from a raw capture frame, not from what was shown: a 25 cm hip jump in one frame. Fixed with **inertialization** (Bollo 2016): each jump starts exactly from the last shown pose and the difference fades. Result: root speed max 12.5 → 3.0 m/s, pops (> 0.3 rad in a frame) 177 → 68, no hip jumps |
| Model v3: a passive tail | The tail sits near gimbal lock in any hinge order and its angles drift through whole turns, so tracking it was noise. Now it has no motors: a spring toward neutral and damping, trailing the body (the first notes' "soft body as a passive trailing parameter"). 54 motors instead of 60; same pose layout, so every reference still works. Stable in a 2 s fall test |
| Tracking environment generalised | One policy can now learn many clips: the reference just ahead is the command (as in PHC and MaskedMimic); each episode ends at its own clip's end. The fall check is relative to the reference, so sit and lie don't count as falls |
| Which parts of the capture a flat-floor dog can follow (`split_flat.py`) | First pass: 22 of 51 clips seemed to hold the dog off the floor for up to 40 s, which I took for platforms; cutting them left 30 min. **That was mostly wrong.** The capture has 1–4-frame paw teleports (marker swaps), and the floor had been set from those frames, so whole clips hovered |
| Capture repair (`make_reference.py`) | Glitch frames (a paw over 12 m/s or the pelvis over 7 m/s) are rebuilt from the good frames on either side, each joint along its shortest angular path; the floor is set from the 5th percentile of the lowest paw. Worst paw speed in the gallop drops from 43 to 8.7 m/s. After repair (and model v2 below), **39.7 min in 57 flat-floor parts** (stand 910 s, walk 649 s, sit 267 s, down 235 s, crouch 193 s, side 52 s, run 50 s, trot 19 s). Only 1.3 min is cut: two clips really do hold the dog off the floor (35 s and 6 s) |
| Physics backend speed, same dog model (1,024 dogs, physics steps per second) | MJX's JAX backend: 20,000; with lighter contacts (13 colliding parts instead of 29): 22,000, so contacts weren't its bottleneck. **MuJoCo Warp: 275,000, about 13× faster**, and it compiles in about 1 s instead of 25. A 0.005 s timestep makes the joint motors fail (dogs collapse), so it stays 0.004 s |
| Warp's contact budget | Global across all simulated dogs, not per dog: with too small a budget every dog falls through the floor. Sized at dogs × 48; memory grows linearly (1.7 GB for 1,024 dogs, 2.4 GB for 2,048) |
| First trot training, JAX backend (stopped) | 9,000 steps/s. After 13 million steps (16 min) the dog stayed on the capture 43 control steps on average, up from 26, out of 300: learning, but too slowly to iterate |
| Same, Warp backend | Brax PPO runs unchanged. Compile 15 s instead of 95; 44,000 steps/s with only 512 dogs |
| Trot clip (stand → walk → trot, 9 s), Warp, 100 million steps, 18 min | The dog stays on the capture 70 control steps on average (1.4 s); no start lasts the full episode. From standing it holds 3.2 s and fails soon after the capture starts walking; started mid-walk, 1.3 s; mid-trot, 0.7 s. **Standing is learned, moving isn't.** Rendered side by side with the capture, the failure is the heading: the dog follows the legs for a second, then turns away |
| Why the heading fails: a bug | The policy saw the trunk orientation error as a raw quaternion difference. A quaternion and its negative are the same rotation, and the capture's quaternions flip sign 20 times in this clip (its root orientation sits near 180°), while the simulation's never do. At each flip the policy's input jumped while the real error was near zero. Fixed: the policy now sees the reference orientation relative to its body as two columns of a rotation matrix, which is continuous and has no sign. Every clip in the library has such flips (up to 38) |
| A second hidden fault: gimbal lock in the capture's joints | Every capture joint is three hinges in one order (Z, X, Y). Across the capture, 11 of 20 joints come within 65–89° of that order's singular pose (99th percentile), including every major leg joint, the head and the tail. There the hinge angles spin at up to 98 rad/s while the limb barely moves. The physics policy is scored on hinge angles, so it was punished on those joints even when the leg was right, and asked to follow targets no motor could reach |
| Fix: each joint's hinge order chosen to stay clear of its singular pose (`joint_orders.json`, model v2) | Each joint's rotation re-expressed in its best order (left forearm 75° → 27° from the singular pose, right lower leg 79° → 12°, tail tip 85° → 24°), angles unwrapped over time, root quaternion kept continuous. Identical poses: 0.0000 mm difference in every body position. Over the flat library the worst hinge speed drops **from 98 to 28 rad/s**, and frames with any hinge over 30 rad/s **from 8.5% to 1.3%**. v1 kept as `dog_mocap_mjx_v1.xml` and `refs_v1/` |
| Did the two fixes speed up learning? (trot clip, same settings, control steps on the capture during training) | Modestly. At 49 million steps: buggy orientation input 40, fixed input 40, fixed input on the gimbal-free model **44**. At about 100 million: 57 (buggy), 52–56 (v2). Differences of this size are within run-to-run noise. Neither fault was the main brake, though the v2 model also stops the solver's "iterations limit" warnings |
| Exploration noise | Plain normal distribution at noise 0.3: **collapsed** (episodes fell from 32 to 9 steps): unbounded actions. Bounded at 0.3: no early advantage over the default. Not the brake |
| Are the motors too weak? (capture replayed with motors only, no policy) | **No.** Stiffer motors help standing (43 → 93 control steps at 4× gain) but not walking (31 → 36) or trotting (22 → 18). Replaying capture without feedback falls at any stiffness: what's missing is learned balance. The trained policy (about 56) beats bare motors (22–43) only modestly so far |
| **The real brake: the planning horizon** (trot clip, discount 0.95 vs 0.98, everything else equal) | At 50 control steps a second, a discount of 0.95 looks only about 0.4 s ahead, about one stride; a stumble is decided earlier. With 0.98 (about 1 s): at 39 million steps **57 vs 42** control steps on the capture; final evaluation **78 after 60 million steps vs 52 after 108 million**, and pose reward per step 0.82 vs 0.77. Half the samples, half again the result. DeepMimic's 0.95 was at 30 Hz, a 0.67 s horizon |
| **First universal tracker**: one policy for the whole flat library (39.7 min, 57 parts), discount 0.98, 300 million steps, 50 min | Control steps on the capture (of 300) from 4,624 evaluation starts, by regime at the start: **lying on the side 257 (72% last the full episode), lying down 251 (71%)**, standing 159 (27%), sitting 141 (25%), crouching 137 (20%), **walking 64 (3.5%)**, trotting 35, running 31 (0%). Training plateaued from 140 million steps: more steps alone won't fix it |
| Why walking fails | Not lag. Every walking episode ends with the trunk turning or tilting away (0.4–0.9 rad) and the dog drifting 0.2–0.34 m sideways, then going down: **lateral balance**. Standing alone is learned (a pure standing clip: all starts hold the full 6 s); the 27% for standing starts comes from standing that turns into walking within the episode |
| Helping walking balance (walking-heavy subset, 10 parts, 4 min; model v3; 100 million steps each) | Upgrade = the policy sees its paws and the capture's relative to its trunk, plus a third look-ahead frame 0.3 s out, plus a reward for matching the trunk's velocity (Peng et al. 2020 used one on this same capture). **From walking starts: 92 control steps vs 71 (+29%)**; training curve 133 vs 97 (+38%); sitting 164 vs 84. Adopted. Walking is still short of solved (2.9% of walking starts last the full 6 s) |
| **The decided first slice in physics** (the brain's composed motion, never seen in training) | The Malinois **lies at rest for the full 8 s**, **gets up on her own legs when called** (rising 0.8 s, standing 1.2 s, stepping out 2.4 s) and fails about two strides into the walk (2.5 s in). The Lab fails rolling onto her side, and also in the walk. Rest and getting up work in physics; walking is the gap. Video: `runs\body-runs\slice_mal\rollout.mp4` |
| Universal tracker again, with the walking upgrade (model v3, whole flat library, 300 million steps, two runs sharing the GPU, 97 min) | Mean control steps on the capture **147 vs 138** for the first universal tracker. Walking 75 (was 64), standing 174 (was 159), sitting 143, crouching 144, lying down 255, on the side 251, trotting 34, running 34. Training curve 136 at the end, still creeping up. A real but modest gain: the upgrade's +29% on the walking subset shrinks to +17% in the whole library |
| Is the network too small? (same run, 1024-512-256 instead of 512-256-128) | **No.** Learns more slowly per sample (training curve 100 vs 136 at 300 million steps, 130 min); the final deterministic evaluation is a wash: mean 152 vs 147, walking 69 vs 75, standing 191 vs 174, crouching 177 vs 144, sitting 126 vs 143 |
| The first slice in physics with the upgraded tracker | No clear change. The Malinois's walk after getting up holds 315 control steps instead of 128 from one start, but another start does worse (92 vs 123). The Lab is the same or worse. Walking is still the gap |
| Is walking lost to drift or to falls? (the library tracker re-evaluated under different rules, no retraining, 2,312 starts) | **Falls.** From walking starts: 73 control steps when straying from the capture ends the episode (as trained); 83 when only a real fall ends it (trunk tipped 0.8 rad off the capture's tilt, or hips 0.2 m low). **Anchoring** the reference to the dog every step, so drift can't build up (the kinematic and physics characters coupled, as in DReCon): 90 at 10% of the gap per step, 87 at 100%. A small help; balance is the problem |
| A hand on the trunk: residual force control (Yuan & Kitani 2020), walking subset | Six extra policy outputs push and twist the trunk (in the trunk's frame; up to half body weight, and half body weight × 0.3 m), with a small charge for using them. Early training curves (control steps on the capture, of 300): **full assist 239 at 10 million steps and 271 at 20 million; twist only 237 at 29 million and 253 at 39 million; no assist 37–62 at the same points.** Stopped early: these runs had the bug below, which hurts short episodes most, so the gap is overstated. Rerun with the fix |
| Do the references float the paws? (lowest point of each paw pad while it bears weight, whole flat library) | Yes, uniformly: median 1.6–2.2 cm above the floor when standing (the floor was set from each part's lowest 5% of pad positions). Walking: front pads 2.3 cm, hind 1.4 cm. A uniform float costs physics nothing (the dog just stands 2 cm lower); the front/hind difference is about 1° of trunk pitch. **Not the walking problem** |
| **A bug in every training run so far** | Brax's auto-reset puts back only the body and the observation of each simulated dog's first start when an episode ends (by failure or the 6 s limit), not the environment's own clock, so later episodes were scored against the wrong capture frames. Measured with random actions under Brax's own wrappers: **83% of steps were one-step episodes whose observation disagreed with the body** (worst gap 19.5); with the fix, 0 (worst 0.001). Evaluations were never affected (they run the bare environment). Fixed in `dog_track.py`; test: `train/test_restart.py`. Every training comparison above ran under the bug and is being rerun |
| **Walking, rerun with the fix** (walking subset, same settings, checkpoints at 39 million steps, deterministic evaluation) | **No assist: mean 212 control steps of 300 (48% last the full 6 s); from walking starts 201 (42% full)**, trotting 202, running 206, sitting 282. With the bug it was 92 from walking starts (2.9% full) after 100 million steps: the bug, not balance, was most of the walking problem. Training curve 67 → 183 → 232 → 244 at 10/20/29/39 million, still rising. **Twist assist: mean 275 (99.8% full); walking 271, all of them the full 6 s**; trotting 300, sitting 300. It works hard, though: on average a twist of 0.18 × body weight × 0.3 m (about 15 N·m) on the trunk. Switch it off and the same policy falls at once (walking 46, 0% full): it learned to lean on the hand |
| **Online: the brain drives the physics dog, and falling is a regime** (`train/online_dog.py --brain mal`) | Every 20 ms: the brain (`dog/brain.py`) asks for a regime and heading; the motion matcher turns it into capture frames 0.4 s ahead; the frames are anchored to where the physics dog is; the tracking policy follows them; breathing rides on the spine. A fall detector (trunk 0.8 rad off the reference, or hips 0.2 m low) switches to FALLEN: the matcher re-syncs to the capture's nearest lying frame, matched on paws and on how the trunk lies against gravity, starting exactly from the fallen body; after 0.8 s the dog is asked to stand; if it isn't up 1.5 s later, the rise failed and it tries again. The brain hears about falls (arousal jumps) and resumes its goal once the dog is up. **Run (the old library policy, 30 s):** the Malinois lies, accepts "come" at 4.3 s (72%), gets up in physics at 5.0 s, falls 0.3 s into the walk, lies, gets up: **6 completed rises, 2 failed**, every walk failing within half a second (the old policy can't walk). The lying match got the trunk within 16–30° in 7 of 9 falls; once the dog rolled onto its back, which the capture never does (no lying-on-the-back frames). 300 ms per step with the GPU shared by three trainings. Video: `runs\body-runs\online_plus\online_mal_brain.mp4` |
| Can the assist teach walking, then be taken away? (a fading "virtual assistant", after Yu, Turk & Liu 2018; walking subset, scored with no assist) | **No need, and it doesn't help.** Never assisted, 78 million steps: mean 256, **walking 254 (78% last the full 6 s)**. Assisted then removed (twist 0.5 for 39 million, 0.25 for 36 million, none for 36 million; 111 million in all): mean 244, walking 233 (61% full). The plain policy wins with 30% fewer samples; the assisted one has to unlearn leaning on the hand. (The planned 0.1 stage crashed, so the last step down was abrupt.) The trunk assist stays a design tool, for example to steady the dog where the game wants it steady, not a training need |
| **The universal tracker, rerun with the fix** (whole flat library, same settings, 300 million steps, 111 min) | **Mean 260 control steps of 300 (was 147); 78% of starts last the full 6 s (was 26%).** By regime at the start: **walking 247 (67% full; was 75 and 5%)**, standing 265 (80%), trotting 221 (87%), running 214 (74%), sitting 261 (85%), lying down 283 (94%), on the side 273 (80%), crouching 265 (76%). Most of the gain came early: the training curve was 203 at 28 million steps and 257 at 84 million, flat after that |
| Online again, with the fixed library tracker (56 million steps, training curve 239) | Getting up is now reliable: **8 rises completed, none failed**; the lying match got the trunk within 7–13° in 8 of 9 falls. Walking still fails about half a second in, every time. Offline on the brain-composed slice, the same checkpoint lies the full 8 s and holds two walking starts for 233 and 400 control steps (previously 31 and 315), but fails 1.9 s after the standing start, where the walk begins |
| Why composed motion is harder than the capture (paw speed while the pad is on the floor) | Capture clips: 90th percentile 0.25–0.34 m/s; 7–13% of on-floor frames over 0.3 m/s. **Motion-matched output: 1.1–2.0 m/s; 27–38% over 0.3 m/s.** The matcher's jumps and blends slide the paws, worst at transitions such as stand to walk. A physics paw can't slide, so the tracker meets references it never practised on |
| **The first slice in physics, online, decided by the brains** (fully trained fixed library tracker, 306 million steps; calls at 8, 14 and 20 s; 40 s; seed 3) | **Malinois: lies at rest, accepts the first call (71%), is up 0.7 s later, walks 3.8 m to the person and sits by her at 15.7 s. Zero falls.** The 56-million-step checkpoint fell every time on the same path; the training curve was flat between them but robustness was not. Lab: rolls onto her side at 2.6 s, declines the first call (35%), accepts the second (65%), is up at 15.7 s, then falls 0.5 s into each walk (3 real falls, rising from the side, which the capture rarely does). The router also called 7 false "failed rises": the dog was up at full height but 0.5 rad off the reference's tilt, just over the threshold, and was sent back to lying. Fixed: "up" now means "not fallen", the same test as the fall detector. About 95 ms per step with one training sharing the GPU. Video: `runs\body-runs\online_fix306\slice_fix306.mp4` |
| A library of the matcher's own output (`make_mm_library.py`), as DReCon trains its tracker | Each dog's brain runs in a mix of activities (play 35%, explore 20%, travel 15%, rest 15%, with a person 15%) and is called to random spots 3–8 m away every 4–10 s; the matcher's output is recorded with the regime it played. **60 one-minute episodes: walking 1,247 s, sitting 1,002 s, standing 754 s, lying 527 s, trotting and running 58 s**, with many stand-to-walk, sit and rise transitions. 2 min on the CPU |
| Tracker fine-tuned on capture + matcher output (from the fixed library tracker, 162 million steps, 31 min) | A small shift. On matcher output: mean 262, 77% full, walking 244 (was 253, 70%, 237). On the capture: mean 255, 74% full, walking 237 (was 260, 78%, 247). Online, the Malinois's slice is again perfect (0 falls, sits by the person at 15.4–15.6 s, about 40 ms per step with the GPU free); the Lab still falls when she sets off after rising from her side |
| **Why the Lab can't get up: regimes need their constraints** (your first notes: "named regimes related by constraints") | Tracing what the matcher played: asked to stand while lying on her side, it reached "down" 0.3 s later, then found no rise near her pose and stayed down for 0.9 s; asked to walk, it jumped from lying straight into a walking frame in 0.16 s, a blend between unrelated postures that no body can follow. The capture's own transition graph explains it: **from lying on the side the only exit is rolling onto the belly (5 examples); from the belly, dogs get up by walking off (5) or sitting first (6); straight down→stand happens once.** The matcher had been free to ignore this |
| The matcher, made regime-faithful (`motion_match.py`) | Three rules. (1) **A jump may only land in the regime already playing**, so a regime changes only by playing a transition the capture contains (a real roll, a real rise), steered there by the "regime 0.3 s ahead" feature. (2) **A small planner** over the capture's transition graph: when the asked regime is reached directly fewer than 3 times in the capture, ask for the next regime on its most probable path (side → down → walk → stand). (3) **A dog settles**: after entering a low posture (lying, sitting, crouching) the clip runs 0.5 s before another change starts. Kinematic tests: side → stand: rolls onto the belly at 0.28 s, settles, walks off at 0.94 s, stands at 1.22 s (hips 0.09 → 0.49 m); down → stand: walks off at 0.10 s, stands at 0.32 s; walk → sit → stand, stand → down → side, sit → walk → down all follow captured paths. Without rule 1, down → stand thrashed between down and walk every 0.2 s and never got above 0.32 m |
| The Lab online, with the regime-faithful matcher | Her reference now really rolls and rises, but **the physics can't follow the roll**: the tracker never learned it (5 rolls in 52 s of side-lying in the whole capture, and in training a roll that lags trips the orientation limit and ends the episode). The Malinois, who lies on her belly, is unaffected |
| The matcher library regenerated with the regime-faithful matcher; tracker fine-tuned on it, with **4× the starts in the second before a posture change** (stand/walk/trot/run count as one posture, so gait flicker isn't boosted; 14% of starts qualify) | Library: 80 one-minute episodes, all posture changes captured ones (26 side→belly rolls, 69 rises by walking off, 31 sit→down, hundreds of sit/stand changes). Tracker (from the previous one, 162 million steps, 28 min), whole combined library: **mean 271 of 300, 85% of starts the full 6 s**; walking 263 (78%), standing 277 (89%), sitting 282 (96%), lying down 275 (88%), on the side 259 (71%), trotting 246, running 211 |
| Can the tracker roll the dog at all? (the three captured side→belly rolls, from 1 s before) | Yes: no failure in 3 s from any of them, ending 10–35° from the capture. The captured rolls are modest (the trunk turns 6–34°) |
| The Lab online again (new tracker; fall detector forgiving 0.3 s of lag, so a body that lags a transition but follows it isn't called fallen) | Still falls at 17.1 s, as she sets off after rising from her side; the trunk is 0.88 rad off every reference frame of the last 0.3 s, so it is a real divergence. From that fallen pose she never gets up (6 failed rises). **The open piece is getting up from arbitrary fallen poses**, which the capture doesn't teach. Next: a recovery skill trained from random fallen states (PHC's fail-state recovery, Luo et al. 2023; "Learning to Get Up", Tao et al. 2022). The Malinois's slice stays perfect. Video, both dogs side by side: [dog/first_slice_physics.mp4](dog/first_slice_physics.mp4) |
| GPU memory | Three trainings at once fill 22.5 of 24.6 GB; a fourth crashed a curriculum stage with a CUDA illegal-address error inside Warp. Keep it to three |
| An untrained policy, per regime (smoke test) | Holds lying for 245 steps on average and sitting for 156, but standing only 49, walking 31, trotting 18. The joint motors give static poses almost for free; anything involving balance has to be learned |
| Version clash caught before GPU time | Brax 0.14.2's PPO calls a function JAX 0.12 removed; patched with a shim. Brax's evaluation wrapper adds a metric that the environment was dropping; fixed. The evaluation started the dog in the wrong pose; fixed. A tiny CPU training run checks the whole loop |

### The first slice, decided rather than scripted (`dog/brain.py`, `dog/first_slice.py`)
The pieces above assembled into one running dog, kinematic for now:

| part | where it comes from |
|---|---|
| What comes next, and when | The router learned from the two dogs, trusted in proportion to its data in the current activity (play: about 140 transitions per dog, so mostly learned; rest: 3–6, so mostly designed priors). Durations from the footage are cut short by clip ends, so only whole bouts are used |
| Breath and exertion | Designed: fast and slow exertion integrators; breathing rate rises with exertion and becomes panting; fatigue lengthens rest and shortens running. Breathing is laid over the spine joints, arousal over the tail |
| Requests and the final say | Designed: each call is weighed against arousal, fatigue and whether the dog is lying; the dog may decline; asking again raises arousal. **Per-dog numbers are placeholders for you to set** |
| Body | Motion matching over the named capture, steered live |

Run (seed 3): both dogs lie at rest; the person calls "come" at 20 s. **The Lab declines** (34% chance; she's on her side), **accepts the second call** (63%), gets up 1.4 s later, walks over and sits by the person. **The Malinois accepts the first call** (70%), is up in 0.7 s and sits by the person at 26.8 s. Video: [dog/first_slice_seed3.mp4](dog/first_slice_seed3.mp4)

### Held for Gemini Carbon (daily quota): done the same evening
Teacher labels for the remaining 36 clips and the rest of the 6-minute play session: completed; see "2026-11-20, evening" at the top.
