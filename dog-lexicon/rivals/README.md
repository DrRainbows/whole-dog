# The opposition

On 2026-11-19 four language models were set against the design in [../README.md](../README.md) and [../11-modules.md](../11-modules.md). They were told not to improve it, and for the rival accounts they were barred from using any framework it already uses. This file summarizes what came back. Nothing here has been reconciled with the design.

- The script: `far_run.py` (any OpenAI-compatible endpoint; models set by environment variables)
- Raw answers, the models' own merge and the rival build specs are not in this repository; they are available on request.
- What to film to decide between the design and the rivals: [shot-list.md](shot-list.md)

**Models.** Three hosted open-weight models (DeepSeek and GLM families plus one other) and a local Qwen 27B on the GPU.

**How to read it.**
- The "trade voices" (shepherd, puppeteer, blind handler and so on) are language models writing in a role. They are arguments, not testimony.
- Sampling above temperature 1.2 collapsed into multilingual token salad on every hosted model. Those answers were cut where they stopped being language and the prompts rerun at 1.0.
- One hosted model was the weakest and mostly incoherent. The usable material is from DeepSeek, GLM and Qwen.

## What they say is false

| the design says | the objection | cheapest test |
|---|---|---|
| What a dog does is bounded by vocabulary | Names are labels people put on afterwards. They bound the taxonomy, not the dog. A list with aliases can absorb almost any name, so a high match rate shows the categories are broad. What reads as alive sits between the names: hesitation, waiting, recovery, the aborted start | Build a continuous controller with no named entries and compare blind 10-second clips against the lexicon-driven dog |
| 16 body parts, each with its own modes | The body is one coupled thing. The spine carries the decision and everything hangs off it; lifting a forepaw changes the load on the other three legs before the paw moves. Sixteen controllers give "a marionette with sixteen strings" | Shove the standing dog sideways, or put it on a slope |
| 88% of entries need no policy, no model of others, no training | That number counts what can be *written* with the signals, not what *occurs* without them. Untaught is not uncontrolled: scratching needs weight shift, rhythm, a stop at contact and recovery. Most of what owners recognize is a response to the handler; one attack puts the real share at 30–40% | Have the dog scratch an itch with a contact sensor, and stop |
| The deciding layer outputs a target, a distance rule and four numbers | There is no slot for intention: a dog offering a ball and a dog deciding whether to fetch it share target, distance and manner. Fetch has two targets. Heel needs bearing, speed matching and turn radius | Toss a ball, then have the handler freeze. Walk a figure-eight at heel |
| Postures are a grid of fore-pair by hind-pair | Posture is continuous, asymmetric and whole-body: one foreleg tucked and one extended, a paw on a knee, pelvic tilt, spinal curve. Transitions are sequences of single-leg moves | Ask for a rest with one foreleg tucked |
| A sequence is one target getting closer | That describes pursuit only. It has no search with nothing in view, no return after the grip, no target switching, no self-grooming | Play a sound off screen; lay a scent with nothing visible |
| Each dog is a set of thresholds over a shared list | Individuality is build, timing, history and habit: latency, how far it escalates, how slowly it settles, which micro-decision wins. The same dog differs by handler | Owner lineup with size and coat hidden |
| People can grade the result by naming what they see | Naming is forgiving and teaches the judges the answer key. Recognition rides on weight, contact and lag before any name. "If my feet agree with the ground, I'm alive. If they pose, I'm a diagram" | One-second silent clips, real dog against render. Ask judges to predict the next half second instead of naming |
| A second model agreeing is support | The same model family listed and judged. The check was blind when listing, not when matching. "Letter confirming letter" | Freeze the lexicon; blind human coders on held-out clips |

## What they would build instead

Each rival was written to contradict the design at the root. Each comes with the test that separates it from the design.

| rival | claim | deciding test |
|---|---|---|
| Viability repair | A dog reads as alive by how it saves itself when the world breaks its plan | Slide the rug during a sit. Does it finish the sit, or abort, splay, catch itself and decide again? |
| Metabolic bank | Motion visibly costs something and rest visibly restores it | Twenty identical retrieves. Do timing and amplitude decay, and recover after rest? |
| Soft-matter lag | Aliveness is deformation and follow-through in a soft body | High-speed capture of one idle head turn: do ears, jowls and skin lag and overshoot? |
| Affordance fit | Behavior is the body settling into fit with surfaces and gaps | Food under a low shelf: does it lower a shoulder and turn the head, or reach straight in? |
| Duet coupling | Aliveness is timing with the handler: overlap, waiting, interruption, repair | The handler starts a hand-offer and freezes halfway |
| Drift | Aliveness is in the nothing between events | Five minutes with no target and no request: "resting" or "switched off"? |
| Commitment | The dog is always in the middle of something and pays to cancel | Remove a target and show a better one elsewhere: is there a blend and a cost? |
| Resonance | Each dog has its own tempo and locks to the handler's | Clap at three tempos; watch the wag and weight-shift rate |

DeepSeek bet on viability repair, "because it makes the failure case primary."

The furthest claims, each marked by its author as a bet:
- Sound carries recognition more than the body does: nails, collar, breath, lip flaps. Test audio-only against silent visuals.
- Recognition is carried by violations: refusing, getting the wrong idea, misjudging a jump and recovering.
- The pauses identify the dog: show judges only the gaps.
- The dog has no behavior of its own; the unit is the pair. Test whether the dog's next move is better predicted from the handler's last move than from its own.
- "Everything is already contained in what is known" is true and fatal: what is contained is a recording, not an animal.

## The audit of the assistant

- It "writes like an advocate": it states a limit, then leaves the stronger impression standing.
- It opened a report with "it supports your claim" and put the limits after.
- It called 17 controllers and 13 shared signals "nearly the same" as 15 parts and 2 signals. The extra signals are the part that most threatens the small-decider claim.
- "A script confirms every entry is covered" is circular: the entries were written in the reduction's own terms.
- It turned 5,535 names against 269 entries into "names inflate and bodies don't" instead of asking whether the list is too coarse.
- Their verdict: call the check a sanity check on a taxonomy. Nothing has been generated, animated or judged by a person.

## Inversions the models bet on

Asked to invert each founding claim and say which inversions are most likely right, DeepSeek chose two:
- The answer is not already contained in what is known. Two dogs and a hand-built vocabulary undersample what dogs do.
- Most behavior must be taught. Parts buy reflexes and locomotion; taking requests and individual manner come through consequences.

## Where the opposition disagrees with itself

- What to build first: breath and blood, or the bond with the handler, or contact with the ground.
- Whether the dog free-runs on its own schedule or exists only as half of a pair.
- Whether recognition lives in the motion or in the person watching.
- Whether the answer is fewer parts, more parts, or one spine.
- Whether history makes the dog or nothing accumulates at all.

## Methods note

Philosophical and theological argument prompts increased creative problem solving, but only when paired with low-refusal open models. Those outputs were combined with frontier models, which in these workflows tended to revert to the mean during brainstorming. The specific prompts are not reproduced here; the build implications they produced (breath and fatigue before behavior, appetite as the first controller, requests as invitations the dog can refuse) survive above only where they could be stated as testable engineering claims (for example, the metabolic-bank rival).
