# whole-dog: vision

**A complete behavioral and biomechanical model of one specific animal, learned from everyday video.**

This document is for reviewers who want to know where this project goes if it works. **Exists** marks what is built and measured today (numbers are from [RESULTS.md](RESULTS.md)). **Proposed** marks research not yet done. Nothing proposed is claimed as a result.

## 1. The theoretical endpoint

Take one particular animal. Film it with a phone during ordinary life. From that footage alone, build a model that:

1. **Moves like that animal**: its proportions, gait, timing, strength and fatigue, in a physically simulated body that can fall, recover, push and be pushed.
2. **Behaves like that animal**: what it attends to, what it wants, how it responds to people and requests, when it declines, how it rests, how it plays.
3. **Is recognized as that animal** by the people who know it best, in a blinded test, at a rate measurably above chance.
4. **Predicts that animal**: given a situation, the model's next few seconds of behavior match the real animal's distribution better than a generic model of the species.

At the limit this is a **digital twin of a living being**: a causal, embodied, individual-specific model whose body, control and decision-making are each grounded in physics and data, and whose fidelity is judged by prediction and recognition, not by appearance.

## 2. Why it matters

**For science: embodied cognition.** Behavior is produced by a body in a world, not by a policy in a vacuum. A model that has to keep a 28 kg body upright while deciding whether to fetch is a testbed for questions that are hard to study in animals directly: how much of behavior is reflex versus plan, what a minimal decision layer must output, how fatigue and arousal shape choice, and what individual differences consist of.

**For AI safety and verification.** Embodied agents will act in the physical world. This project is built around practices that matter for trustworthy embodied AI:
- **Hierarchical control with a hard physics floor.** The decision layer can only request; physics and a learned reflex determine what actually happens. Falling is a regime with detection and recovery, not an exception.
- **Grounded language.** Words map through an explicit grammar to body regimes with priced transitions, so what the agent was asked, what it decided and what its body did are separately inspectable.
- **Pre-registered evaluation.** Tests are fixed before runs (for example the shove test), negative results are recorded, and claims are traced to requirements with verification evidence. This is the evaluation discipline embodied AI needs.
- **Parity between training and deployment.** The deployed CPU policy is checked against the training environment to 4 × 10⁻⁷, a concrete example of verifying that what was evaluated is what ships.

**For animal welfare and veterinary science.** An individual-specific biomechanical model learned from phone video could flag gait changes, fatigue or pain earlier than an owner notices, without sensors or a lab.

**For games and digital companions.** Recognizably individual, physically honest animals are an open problem in interactive media; this is a path that runs on a gaming PC.

## 3. The first-principles argument

1. People recognize a particular animal from motion and timing, often more than from appearance. Point-light studies show identity and state survive when only moving dots remain. So fidelity should be measured on motion, in a blinded recognition test, before investing in looks.
2. Motion is constrained by a body. A physics body with realistic mass, joint ranges and muscle strength removes whole classes of implausible motion for free and makes failures (stumbles, falls) real rather than scripted.
3. Behavior has a vocabulary that can be written down (postures, gaits, signals, sequences) and checked against data. Writing it first, then measuring what the data contains that the vocabulary misses, turns intuition into testable claims.
4. A motion library can only produce transitions it contains. Restricting regime changes to observed transitions, and planning routes over that graph, keeps behavior faithful while still composing new sequences.
5. Individuality is a set of parameters over shared machinery: thresholds, timing, strength, preferences. If that is right, a new individual should need minutes of footage, not hours. If it is wrong, the recognition test will show it.
6. Everyday video is the only data source that scales to arbitrary individual animals. Therefore the long-term pipeline must go from phone video to body, motion and behavior, with no motion-capture studio in the loop.

## 4. Staged research roadmap

### Stage 0: what works now (**Exists**)
- Physics body: 54 actuators, passive tail, 28 kg; anatomical joints with ligament ranges; dog-strength muscles with activation lag; MuJoCo, trained on one RTX 3090 with Brax and MuJoCo Warp/MJX.
- One tracking policy for a whole motion library: 78% of starts held for full 6 s episodes (85% after fine-tuning on the matcher's own output); 93.5% capture completion at dog strength.
- `scramble_up` reflex: up from side or back in 100% of trials, under 0.5 s; every real fall in social play recovered (median 0.94 s).
- Brain, grammar and motion matcher driving physics online; two dogs in one world; real time on one CPU core (5.1 ms per 20 ms step); a Panda3D playable.
- 316-entry behavior lexicon reduced to 84 modes on 16 body parts, checked against real motion and against blind lists from independent language models.
- Phone-video pipeline: DeepLabCut SuperAnimal tracking, normalization, gait and reaction features; an identity classifier on the owner's own clips (87.5% on unseen clips).
- Browser identity test with pre-registered statistics; validated on simulated judges.
- Requirements with IDs, verification methods and evidence.

### Stage 1: recognition (**Proposed**, next 3 months)
- Run the identity test with real judges who know the two dogs: which carrier (gait dots, reaction dots, pauses, sound) holds identity above chance.
- Remove walking falls (most remaining falls happen at slow walking speeds), add fatigue as a tested variable.
- Release a reproducible training recipe from the public motion-capture dataset.

### Stage 2: from phone video to body (**Proposed**, 6 to 12 months)
- 3D motion from casual monocular video (building on BARC, BITE and Animal Avatars style reconstruction), retargeted to the physics body.
- Fit individual proportions, strength and timing from video; train the tracker on the individual's own motion.
- Benchmark: how many minutes of an individual's footage are needed to reach a given recognition rate.

### Stage 3: behavior from observation (**Proposed**, 1 to 2 years)
- Learn the brain's parameters (drives, thresholds, preferences) from labeled everyday video, with the lexicon as a structured prior.
- Predictive evaluation: next-seconds behavior of the twin vs. the real animal in matched situations.
- Social coupling: the human-dog pair as the unit (timing, waiting, repair), tested against the alternative hypotheses recorded in [dog-lexicon/rivals](dog-lexicon/rivals/README.md).

### Long-term goal
A general pipeline from everyday video to individual-specific, physically grounded behavioral models of animals, and the evaluation methods (blinded recognition, prediction, physics plausibility) to know when such a model is faithful. A step toward digital twins of living beings and toward embodied cognition models that can be inspected and verified layer by layer.

## 5. Open research questions

1. **Where does identity live?** In gait, reactions, pauses, sound, or the coupling with a person? The identity test is designed to answer this.
2. **How small can the decision layer be?** Does a target, a distance rule and a few manner parameters suffice for most behavior, as the lexicon reduction predicts, or do intention and memory need more?
3. **Vocabulary vs. continuum.** Does a named-regime structure limit lifelikeness, compared with a continuous controller with no named entries?
4. **Strength and failure.** At realistic strength, which transitions does the body make hard, and do those match what real dogs avoid?
5. **Few-shot individuals.** How much individual footage is needed, and what transfers from the species model?
6. **Evaluation.** Can recognition and prediction tests become a standard benchmark for behavioral digital twins?

## 6. How others can build on it

- **Body models** (`body/*.xml`) are plain MuJoCo and can be used in other controllers or species.
- **The lexicon** is a structured, checkable description of dog behavior with a compiler (`dog/grammar.py`) and checks; it can be extended or adapted to other species.
- **The identity test** is independent of this build and can evaluate any animated or simulated animal.
- **Training and parity tools** show a full path from GPU training to a JAX-free CPU runtime.

Collaboration with animal-behavior, biomechanics and graphics researchers is especially welcome.
