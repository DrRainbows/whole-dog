# Decision 001: what to build first

Date: 2026-11-19. **Status: revised the same day.** An independent attack by Gemini Carbon ([001-attack-gemini.md](001-attack-gemini.md)) found factual errors that changed the ranking. The decision now in force is in **Revision 1** at the end. The original record is kept unchanged above it.

## The question
Which path has the best chance of a real breakthrough, judged by evidence instead of by whichever side argued last?

## The rule, fixed before scoring
Each candidate is scored 1–5 on seven criteria. The highest total wins, and a tie goes to the cheaper first test.

1. **Who judges success.** People or physics score 5; a model's opinion scores 1.
2. **Unknown outcome.** Is the result of the first test unknown to everyone, every model included?
3. **Distance from the default.** The default answer, from an AI or a studio, is physics plus motion capture plus reinforcement learning.
4. **Chance it works and matters.** Will the first test succeed, and does its result change the build?
5. **Cost to the first decisive test.** What exists now: one person, an RTX 3090, phone footage, AI coding help.
6. **Leverage.** If it works, how much of the game does it carry?
7. **Gap in prior art.** Has it been done?

## Candidates
| id | path |
|---|---|
| P1 | **Lexicon engine.** Named behaviors drive modules through a router. This was the original design. |
| P2 | **Body first.** A physics dog with balance recovery, breath, heat, fatigue and soft parts. The opposition's rivals and your first notes converge here. |
| P3 | **Recognition-driven evolution.** A dog built from parameters. The owner watches six variants move, picks the one most like their dog, and the system breeds new variants from that pick. Repeat. This is the method behind forensic face composites (EvoFIT), applied to movement. |
| P4 | **Footage played forward.** Motion matching over the dogs' own footage. |
| P5 | **Duet.** The dog and the player as one timing system. |
| P6 | **Prediction.** A model trained to predict the dogs' next half second, scored against how well people predict it. |

## Scores
| | judge | unknown | far from default | works and matters | cost | leverage | prior-art gap | total |
|---|---|---|---|---|---|---|---|---|
| P1 lexicon engine | 4 | 2 | 2 | 2 | 2 | 3 | 2 | 17 |
| P2 body first | 5 | 3 | 2 | 3 | 2 | 4 | 3 | 22 |
| **P3 recognition evolution** | **5** | **5** | **5** | **3** | **5** | **5** | **4** | **32** |
| P4 footage forward | 4 | 3 | 1 | 2 | 2 | 3 | 1 | 16 |
| P5 duet | 4 | 3 | 3 | 3 | 4 | 4 | 2 | 23 |
| P6 prediction | 5 | 4 | 3 | 3 | 3 | 3 | 3 | 24 |

**Bias warning.** I proposed P3, and these are my scores. That is why the record went to Gemini Carbon for an independent attack. If the attack shows a factual error that changes the ranking, the decision changes.

**Why P3 scores high, and what it rests on**
- **The owner's recognition is the objective.** No model's opinion of what a dog is enters the loop. That is what makes the path AI-probability-agnostic.
- **Recognition beats description for faces.** Witnesses can rarely describe a face but can recognize one. Composites evolved from a witness's choices are named far more often than composites assembled from described features (Frowd and colleagues, EvoFIT). Assembling from named parts is what the lexicon does for a dog.
- **People recognize individuals from movement alone.** In point-light studies, people identified friends from walking dots (Cutting & Kozlowski, 1977). Motion can be decomposed into a space where single directions change what people perceive (Troje, 2002).
- **Comparisons are enough to shape simulated movement.** A simulated robot learned a backflip from about 900 human comparisons (Christiano et al., 2017).
- **Nobody knows if it works for dogs.** No known product or study has owners steer a simulated animal toward their own animal. Nobody knows whether owners can do it, or whether choosing between whole dogs beats setting named sliders. Only the test can say.

**Prior art the design must not claim**
- A companion that treats commands as requests and keeps its own will: Trico in *The Last Guardian* (2016).
- A creature shaped by the player's feedback: *Black & White* (2001).
- An ethology-based virtual dog: Blumberg, MIT, 1996–2002.

## What gets built (P3, carrying parts of the others)
1. **A dog made of parameters.** The genome covers paw placement fixed in the world, footfall timing, shoulder swing, head carriage and nod, neck follow-through, soft ears and tail with real lag, breath and heat with panting, how it stops, how it squares up its feet, and idle weight shifts. These are the cheap parts of P2, so the owner can select on them from day one.
2. **The loop.** Six variants run the same clip (stand, trot, stop, stand and pant). The owner picks the closest one or two, and the next generation is bred from the picks.
3. **The control condition.** The same dog with named sliders. That is the vocabulary approach, scored with the same tests.
4. **Three views.** Each variant can be shown as the full dog, a silhouette, or dots only (movement without appearance).

The lexicon becomes the scoring sheet. Prediction (P6) becomes the metric once footage exists. Physics recovery (P2) goes under the same genome later.

## Kill rules, fixed before testing
| id | when | pass | if it fails |
|---|---|---|---|
| K1 | Today, in simulation | A simulated owner who picks by visible motion with noise brings the dog to ≤35% of its starting distance from a hidden target within 25 generations, in ≥80% of runs, beating random search by ≥1.5× on the same budget | Fix the loop before any person uses it |
| K2 | Owner pilot | After ≤30 generations the owner rates the best variant ≥5 of 7 on "moves like my dog" for at least one dog | If both dogs score ≤3, the space can't hold a dog: move the body to P6 or P2 |
| K3 | Owner, blind | In 20 forced choices between the evolved dog and the breed default, shown as dots with sides randomized, the owner picks the evolved dog ≥15 times (p < 0.05) | The evolution followed noise |
| K4 | Others, with footage | People who know the two dogs match each evolved dog to the right real dog's dot clip ≥75% of the time | The owner's recognition doesn't transfer |

**Breakthrough** means K3 and K4 pass, and the evolved dog beats the slider-built dog on them: same owner, same time limit. That would be recognition beating description, decided by people.

---

## Revision 1: the attack changed the ranking

### What the attack got right
Each point below is checked against the cited work or recomputed.

- **The method has been built before.** People choosing among simultaneous variants of procedural walk cycles: Ventrella (1994), Lim & Thalmann (2000). Galleries of variants tuned by preference learning: Brochu, Brochu & de Freitas (2010), Koyama, Sato & Goto (2020). Prior-art gap 4 → 2.
- **K1 was set to fail.** Picking one of six for 25 generations shrinks the distance to the target by at most about e^(−20/D), and that is for a noise-free (1,6) evolution strategy on an ideal problem. With D ≈ 34 evolvable genes the 35% target is out of reach; it needs D ≤ 19 even with no noise. Unknown outcome 5 → 3; works and matters 3 → 2.
- **I overstated the evidence.**
  - Cutting & Kozlowski's viewers named friends from walking dots 38% of the time, against 17% chance.
  - Later work (Loula et al., 2005) found expressive and reactive actions carry identity better than walking. The clip I planned, a steady trot, is the weakest carrier.
  - Christiano et al. trained a reward model on the comparisons and then ran millions of simulation steps. P3 has neither.
- **The kill rules had confounds.**
  - An owner who bred a dog for an hour recognizes their own animation.
  - One clip shown 20 times is one sample, not 20.
  - Labrador against Malinois tests breed, not the individual.
  - No sample size was set.

### What I don't accept
- **The cost.** "Months" to build the procedural rig is wrong; it's days.
- **Gemini's own candidate (P7).** It scored itself 31 in the same generous way I scored P3 32.

### Rescored, applying only the factual corrections
| | before | after |
|---|---|---|
| P3 recognition evolution | 32 | 23 |
| P6 prediction | 24 | 24 |
| P5 duet | 23 | 23 |
| P2 body first | 22 | 22 |
| P1 lexicon engine | 17 | 17 |
| P4 footage forward | 16 | 16 (Gemini: 23) |

No path now wins on score: four sit within two points of each other (22 to 24). The rule's tiebreaker decides: the cheaper first test.

The cheapest first test is one every path needs, and nobody knows its result. Both the attack and the earlier opposition round named it: **measure where a particular dog's identity lives, from real footage, with people as judges and same-breed dogs as foils.** Its result picks the path:
- If reactions carry identity, build the duet and prediction paths (P5/P6).
- If gait carries it, build footage forward (P4).
- If neither does, rethink.

### Decision in force
**Build the measurement first.**
1. **Track the body** in phone footage of the two dogs on the RTX 3090. Normalize size and heading, so a dog's build can't give the breed away.
2. **Make clips that each carry one thing:** movement only (dots) and sound only. Cut steady trotting separately from reactions to a clap or a toy swap.
3. **Measure timing.** Find the clap or whistle in the audio, then the dog's head-turn latency in the tracked keypoints. Measure stride timing and its variability.
4. **Let people decide.** People who know the dogs, and who built or tuned nothing, pick the dog from clips with same-breed foils.
5. **Build the game dog** around whichever carrier wins.

P3 survives only as a later tuning tool. It would use preference learning on a small basis fitted to real data, not naive evolution. The procedural rig ([dogspace/](../dogspace/)) is parked until the measurement says which body signals matter.

### Kill rules in force (they replace K1–K4)
| id | rule |
|---|---|
| M1 | **Pipeline.** Keypoints from a side-view trot are complete enough to measure stride timing: at least 80% of frames have all four paws tracked above the confidence cutoff. |
| M2 | **Timing.** Clap onsets are found within 20 ms on synthetic audio. Head-turn latency is recovered within 2 frames on synthetic tracks. |
| M3 | **Design.** At least 8 judges who know the dog and didn't build anything. At least 10 independent clips per dog, from different days. Each trial uses a clip not used before for that judge. The dog appears with 3 foils of the same breed (chance 25%). 20 trials per judge per carrier. |
| M4 | **A carrier holds identity** if group accuracy is ≥ 40% and above chance: exact binomial, one-sided, α = 0.05, Holm-corrected across carriers. |
| M5 | **The deciding comparison.** Reaction clips against steady-trot clips, paired across judges. The winner decides the build. |
