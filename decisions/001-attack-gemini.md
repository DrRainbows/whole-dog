### 1. PRIOR-ART CHECK

**Audit of cited claims**
- **EvoFIT vs. feature-based composites (Frowd et al., 2004–2015):** Factually accurate for eyewitness faces, **misapplied to P3.** EvoFIT beats feature assembly *only* under two conditions P3 violates: (1) EvoFIT searches a Principal Component Analysis (PCA) space derived from real faces, guaranteeing every vector is biologically plausible and low-dimensional, whereas P3 searches hand-authored procedural parameters; (2) eyewitnesses rely on degraded memory of a stranger seen days earlier. When the target stays in view (as with an owner who lives with the dog and holds phone video of it), direct feature adjustment does not suffer the memory-retrieval bottleneck EvoFIT was built to bypass.
- **Cutting & Kozlowski (1977):** Overstated. Viewers identified six walking friends from point-lights at **38% accuracy** (chance: 16.7%)—statistically above chance, not reliable identification (62% of guesses failed). Subsequent work (Loula et al., 2005) showed steady walking carries weak individual identity compared to expressive, reactive actions.
- **Troje (2002):** Misapplied. Troje extracted perceptual axes by running two-stage PCA on optical motion-capture data of 80 real walkers. It proves real motion data can be compressed into linear perceptual axes; it does *not* show that a hand-coded procedural rig spans individual identity.
- **Christiano et al. (2017):** Misrepresented structurally. The simulated hopper learned a backflip from ~900 pairwise human comparisons because those comparisons trained a **neural reward predictor**, against which reinforcement learning executed *millions* of simulated steps. P3 discards both the learned surrogate model and the inner optimizer, giving direct evolutionary breeding only 150–180 total evaluations (30 generations × 6 variants).
- **The Last Guardian (2016):** Accurate.
- **Black & White (2001):** Accurate.
- **Blumberg, MIT (1996–2002):** Accurate (*Silas T. Dog* c. 1995–1996 through the Synthetic Characters group's *Dobie* / *AlphaWolf* c. 2002).

**Prior art missed**
- **Interactive evolution of procedural character motion:** **Ventrella (1994)** (*Artificial Life IV*) and **Lim & Thalmann (2000)** applied interactive genetic algorithms directly to procedural character walk cycles chosen by a human viewer from simultaneous variants.
- **Human-fatigue ceiling in interactive evolution:** **Takagi (2001)** (*Proc. IEEE*, survey of Interactive Evolutionary Computation) established that human evaluators degrade from perceptual fatigue after **10–20 generations**, directly contradicting P3's 30-generation plan.
- **Gallery-based preference optimization of procedural animation:** **Brochu, Brochu, & de Freitas (2010)** (*SCA*, "A Bayesian Interactive Optimization Approach to Procedural Animation Design") and **Koyama, Sato, & Goto (2020)** (*SIGGRAPH*, "Sequential Gallery for Interactive Visual Design Optimization") present a user with small galleries of variants (including 6-option grids) to tune continuous procedural animation parameters using active Gaussian Process preference learning—doing what P3 proposes, without naive genetic drift.
- **Owner-specific 3D dog replicas from casual phone footage:** **Yang et al. (2022)** (*CVPR*, *BANMo*) and **Rüegg et al. (2022, 2023)** (*CVPR*, *BARC* and *BITE*) reconstruct articulated 3D shape and motion of *specific individual pet dogs* directly from casual monocular phone videos.
- **Dog-specific data-driven game/physics locomotion:** **Starke et al. (2018)** (*SIGGRAPH*, Mode-Adaptive Neural Networks for quadrupeds) and **Peng et al. (2020)** (*RSS*, imitating real dog motion capture in simulated quadrupeds).

---

### 2. SCORING ERRORS

**Specific errors**
- **Judge (P1, P4, P5: 4 → 5; +1 each):** Criterion 1 awards 5 if people or physics judge success. Blind human recognition tests (K3/K4) evaluate P1, P4, and P5 identically to P3. Docking every rival 1 point is pure thumb-on-the-scale.
- **P3 Cost (5 → 2; −3):** "What gets built" smuggles half of P2 into P3 ("paw placement fixed in the world, footfall timing, shoulder swing, head carriage and nod, neck follow-through, soft ears and tail with real lag, breath and heat with panting, how it stops, how it squares up its feet, and idle weight shifts"). Authoring a bug-free, coupled quadruped procedural rig expressive enough to hold *individual* quirks before running test #1 takes months, not days.
- **P3 Leverage (5 → 2; −3):** Optimizing a parameter vector for one canned, non-interactive clip (`stand, trot, stop, pant`) yields zero steering, zero environmental grounding, zero state transitions, and zero gameplay reactivity.
- **P3 Unknown (5 → 3; −2):** Both IEC human fatigue (Takagi, 2001) and Evolution Strategy sample limits in >20 dimensions are mathematically settled, not unknown.
- **P3 Far from default (5 → 4; −1):** Parametric procedural walk rigs are standard indie practice; only the evolutionary wrapper is non-standard.
- **P3 Works and matters (3 → 2; −1):** 30 generations of 6 picks without a surrogate model cannot converge in a 30–50 parameter space (see Section 4).
- **P3 Prior-art gap (4 → 2; −2):** Ventrella (1994), Lim & Thalmann (2000), Brochu et al. (2010), and Koyama et al. (2020) already built gallery-driven evolution/optimization of procedural animation.
- **P2 Far from default (2 → 1; −1) & Prior-art gap (3 → 1; −2):** Criterion 3 explicitly defines physics + RL as the industry default; Peng et al. (2020) and Isaac Lab quadrupeds saturate the prior art.
- **P4 Far from default (1 → 2; +1), Works and matters (2 → 4; +2), Cost (2 → 3; +1), Leverage (3 → 4; +1), Prior-art gap (1 → 2; +1):** Studio default uses optical mocap stages, not casual handheld pet videos. You *already own* the phone footage and an RTX 3090; running off-the-shelf animal pose trackers (SLEAP, DeepLabCut, BITE) extracts ground-truth individual kinematics faster than hand-coding a parametric quadruped rig, guarantees the real dogs' timing is in the data, and feeds directly into an interactive motion-matching controller.
- **P6 Leverage (3 → 2; −1):** A 0.5-second pose predictor is an evaluation metric, not a controllable game character.

**Recomputed table**
| | judge | unknown | far from default | works and matters | cost | leverage | prior-art gap | total |
|---|---|---|---|---|---|---|---|---|
| P1 lexicon engine | 5 | 2 | 2 | 2 | 2 | 3 | 2 | **18** |
| P2 body first | 5 | 3 | 1 | 3 | 2 | 4 | 1 | **19** |
| P3 recognition evolution | 5 | 3 | 4 | 2 | 2 | 2 | 2 | **20** |
| P4 footage forward | 5 | 3 | 2 | 4 | 3 | 4 | 2 | **23** |
| **P5 duet** | **5** | **3** | **3** | **3** | **4** | **4** | **2** | **24** |
| P6 prediction | 5 | 4 | 3 | 3 | 3 | 2 | 3 | **23** |

P3 drops from 1st (32) to 4th (20). P5 leads among listed candidates (24), followed by P4 and P6 (23).

---

### 3. THE BEST CASE AGAINST P3

People who know dogs—especially comparing a Labrador and a Belgian Malinois—do not recognize a *particular* dog primarily from steady-state trot posture; they recognize it from **attentional latency, impulse control, transition jerk, and handler check-in timing** (how fast the Malinois snaps orientation to movement and holds fixation vs. the Lab's softer latency, ribcage arc, and distracted sniffing).

An unlisted hybrid of P4 and P5—**P7: Footage-extracted reactive timing**—extracts 2D/3D keypoints from your existing phone videos (using DeepLabCut/SLEAP + BITE on the RTX 3090) to measure each dog's real movement modes, transition accelerations, and head-turn latencies, driving an interactive player-dog timing controller (P5). Scored on the seven criteria:
1. **Judge (5):** Tested blind on humans (K4) and held-out video error.
2. **Unknown (5):** Nobody knows whether canine individuality in a game lives in gait kinematics or in stimulus-response timing distributions.
3. **Far from default (4):** Studios mocap gait cycles; they do not fit individual pet attentional-latency distributions from casual video.
4. **Works and matters (4):** Uses real physical measurements of *those exact two dogs* instead of hoping a hand-coded oscillator rig contains them.
5. **Cost (4):** Keypoint extraction on existing phone footage runs on a 3090 in days, before writing any procedural gait engine.
6. **Leverage (5):** Produces an interactive dog that responds to the player and world—which *is* the game—rather than a non-interactive 5-second animation clip.
7. **Prior-art gap (4):** Reconstructing individual pet behavioral micro-timing from phone video for an interactive companion is unbuilt. **Total: 31.**

Even restricted to listed candidates, **P5 (24)** and **P4 (23)** beat P3 because both build an interactive controller first rather than a static movie parameterizer.

---

### 4. HOW P3 MOST LIKELY FAILS

**The dimensionality pincer.** P3 is trapped between two incompatible mathematical bounds:
1. **Expressiveness lower bound:** The 12+ listed behavioral categories (footfall timing, shoulder swing, head carriage/nod, neck lag, ear/tail dynamics, breathing/panting, stopping mechanics, foot squaring, idle weight shifts) require **30–50 continuous parameters** just to avoid robotic symmetry—and because they are hand-authored oscillators rather than a PCA basis of real dogs, the real dog's coupled biomechanical signature likely lies *outside* the manifold altogether.
2. **Optimization upper bound:** Picking 1 of 6 variants yields at most $\log_2(6) = 2.58$ noiseless bits per generation (64.6 bits in 25 generations). In Evolution Strategy theory, even an **oracle step-size $(1,6)$-ES with zero human noise on a perfectly isotropic sphere** reduces distance by at most $\exp(-c_{1,6}^2 G / 2D) = \exp(-20.07 / D)$ in $G=25$ generations. Reaching K1's $\le 0.35$ threshold is mathematically impossible above **$D = 19$** *without noise*, and collapses above **$D \approx 8\text{–}10$** once human perceptual noise and attentional masking (viewers can only compare 2–3 salient traits across 6 simultaneous clips) are included. The owner will spend 10 generations fixing gross traits (trot speed, tail height) while the remaining 30 parameters undergo random genetic drift, then hit perceptual fatigue as all 6 variants look equally artificial.

**Earliest cheap observation:** Today, in pure Python (10 minutes, zero 3D graphics), run K1 at $D = 35$ where the simulated human ranks the 6 variants using noisy observation of only the 3 largest-error visible features per clip (attentional bottleneck) and standard genetic crossover/mutation without oracle step-size knowledge. It will fail K1 immediately. Second, run pose tracking on 5 seconds of your *real* Lab and Malinois phone footage, normalize skeleton limb lengths so size doesn't leak breed, and check K4 on real tracked dots before building any rig at all.

---

### 5. KILL RULES

K1–K4 will pass a broken system because of four design bugs:

- **K1 (Weak baseline & fake human model):** In $D=30$, 150 samples of uniform random search reduce target distance by <10%, so "beating random search by $\ge 1.5\times$" is trivial, while stationary Gaussian noise misses human **attentional masking** (ignoring subtle traits while gross traits vary) and **non-Euclidean perceptual coupling**. Missing: K1 must benchmark naive breeding against Bayesian preference learning (CMA-ES or GP active preference) and direct slider coordinate descent.
- **K2 (IKEA effect & adaptation):** Asking the unblinded creator for a $\ge 5/7$ rating after 30 generations of sunk effort and visual adaptation to procedural motion measures cognitive dissonance and grading on a curve, not realism.
- **K3 (Fatal familiarity & pseudoreplication confounds):**
  1. *Creation recognition, not dog recognition:* After staring at and breeding a dog for 30 generations, the owner will pick it over an unseen breed default 20/20 times simply because **they recognize the animation they just spent an hour building**, even if it looks nothing like their real dog.
  2. *Statistical pseudoreplication:* Repeating the *exact same deterministic clip* 20 times with randomized left/right sides violates trial independence. Once the owner spots *any* single quirk distinguishing clip A from clip B on trial 2, they score 19/20; binomial $p < 0.05$ at 15/20 is completely invalid because $N=1$ evolved dog and $N=1$ trajectory, not 20 independent samples.
- **K4 (Breed-stereotype & morphology confounds, missing $N$):**
  1. *Labrador vs. Belgian Malinois is a breed test, not an individual test:* Even as point-lights, unless limb proportions and shoulder heights are warped onto an identical canonical skeleton, dot spacing immediately betrays breed geometry. Even on a normalized skeleton, evolving "generic high-head alert shepherd" vs. "generic relaxed retriever" lets friends match the two dogs at 100% accuracy using **breed stereotypes alone**, proving zero individual recognition. K4 must test each evolved dog against **other real dogs (or plausible synthetic foils) of the same breed**.
  2. *Unspecified power:* "$\ge 75\%$ of the time" on a 50%-chance binary match without specifying sample size $N$ allows 3 out of 4 friends ($p = 0.31$) to count as a pass.
- **Breakthrough comparison (Order confound):** Testing evolution vs. sliders with "same owner, same time limit" across $N=1$ owner and $N=2$ dogs is confounded by learning transfer—whichever tool goes second benefits from the owner having just spent 45 minutes studying the rig's capabilities and their dog's footage.

---

### 6. VERDICT

**No:** P3 front-loads months of hand-coded procedural rigging whose dimensionality exceeds what 30 noisy 6-variant generations can optimize, while testing non-interactive steady-state gait instead of the reactive timing that actually identifies a particular dog.

**Single change:** Before building any procedural rig or evolutionary loop, extract size-normalized 2D/3D keypoints directly from your existing Labrador and Malinois phone footage on the RTX 3090 and run a breed-controlled K4 on those real dot clips (steady gait vs. reactive transitions) to prove which kinematic signals actually carry individual identity, then build an interactive controller (P4/P5) around the signals that pass.
