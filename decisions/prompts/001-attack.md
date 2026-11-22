You are an adversarial reviewer. Below is a decision record for a game project: a video-game dog that people who know dogs accept as a real, particular dog, built by one person with an RTX 3090, phone footage of two real dogs (a Labrador and a Belgian Malinois) and AI coding help. The author scored their own proposal (P3) highest. Your job is to find what is wrong, not to improve the prose.

# Decision 001: what to build first

Date: 2026-11-19. Status: decided. An independent attack by Gemini Carbon is filed in [001-attack-gemini.md](../001-attack-gemini.md).

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


Answer in these sections, concretely, without praise:
1. PRIOR-ART CHECK. For each cited claim (EvoFIT and feature-based composites; Cutting & Kozlowski 1977; Troje 2002; Christiano et al. 2017; The Last Guardian; Black & White; Blumberg's virtual dog), say whether it is stated accurately, and correct it if not. Then name any prior work the author missed that does what P3 proposes or something very close (interactive evolution or preference learning applied to animal or character motion, owner-specific pet replicas, motion style from user choices). Give authors and years.
2. SCORING ERRORS. Where is a score wrong, and by how much, with the reason? Recompute the totals with your scores.
3. THE BEST CASE AGAINST P3. Under the same seven criteria, make the strongest case that another candidate, or one not listed, should be built first.
4. HOW P3 MOST LIKELY FAILS. The most likely failure, why, and the earliest cheap observation that would reveal it.
5. KILL RULES. Are K1–K4 adequate? What is missing, what threshold is wrong, what confound will fool them (for example, the owner learning the variants rather than recognizing their dog)?
6. VERDICT. Build P3 first: yes or no, in one sentence, and the single change you would make to the plan.
