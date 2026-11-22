# Dog lexicon v0.1

Everything a dog does, written as what the body does, so every behavior the game dog needs has a name and every name has a body. Built 2026-11-19. It is a claim to be checked against footage of real dogs, not trusted as is.

## Status
Nothing here has been built, animated or judged by a person. The lexicon is a description of dog behavior drawn from text. On 2026-11-19 four language models were set against it. Their objections, their rival designs and the footage that would decide between them are in [rivals/README.md](rivals/README.md) and [rivals/shot-list.md](rivals/shot-list.md). They are not reconciled with this file, and several claims below are marked as disputed.

**Compiled and checked, 2026-11-21.** [dog/grammar.py](../dog/grammar.py) reads every table here into a grammar the dog runs on:
- words to bodies;
- reflexive (R, X) or planned (D, O, L);
- every link priced in bits;
- routes through the posture graph.

Against data, the posture graph (02), written from text before any footage, allows 34% of the possible changes between regimes. Yet it covers 78% of the changes in the 41-minute dog capture and 80% of the two dogs' labelled changes, where random graphs of the same size cover 33–34%. What it misses is walking straight off from a sit or a down, and the low in-between stance. Details are in [RESULTS.md](../RESULTS.md) (21 Nov). One fix came from this file's own rule 1: shake_off now carries the word "shake".

## Count
| file | what | entries |
|---|---|---|
| [01-postures.md](01-postures.md) | held body shapes | 35 |
| [02-posture-graph.md](02-posture-graph.md) | allowed transitions between postures | 28 edges |
| [03-locomotion.md](03-locomotion.md) | gaits, jumps, falls | 49 |
| [04-head-mouth-nose.md](04-head-mouth-nose.md) | attention, scent, holding, biting, licking | 50 |
| [05-ears-eyes-tail-coat.md](05-ears-eyes-tail-coat.md) | signals | 40 |
| [06-limbs-whole-body.md](06-limbs-whole-body.md) | pawing, digging, scratching, rolling | 30 |
| [07-voice.md](07-voice.md) | barks, growls, whines | 23 |
| [08-sequences.md](08-sequences.md) | ordered chains of entries | 39 |
| [09-body-upkeep.md](09-body-upkeep.md) | eating, elimination, sleep, physiology | 42 |
| [10-requests.md](10-requests.md) | what a handler can ask for | 79 |
| [13-added-from-blind-lists.md](13-added-from-blind-lists.md) | mating, birth, pups, fights, illness, dying | 47 |

Total: 316 body entries, 39 sequences built from them, 79 handler requests mapped onto them, 28 posture transitions.

## The reduction
[11-modules.md](11-modules.md) reduces the 316 body entries to 84 modes on 16 body parts, plus two signals sent to every part at once: manner (four numbers) and target. [12-recipes.md](12-recipes.md) writes every entry that way, and `python check_recipes.py` verifies and counts it.

| what an entry needs beyond modes, manner and target | entries | share |
|---|---|---|
| nothing | 194 | 61% |
| physics acting on the parts | 22 | 7% |
| one reflex trigger | 62 | 20% |
| a policy over time | 19 | 6% |
| an estimate of someone else's attention or intent | 8 | 3% |
| training to a cue | 11 | 3% |

**Disputed.** These classes are my own labeling of what each entry can be *written* with. They are not a measurement. The objection: being writable with the signals is not the same as happening without control. Scratching still needs weight shift, rhythm, a stop at contact and recovery, and most of what owners recognize is a response to the handler. An earlier version of this file said 88% of entries "should appear without being taught". That claim is withdrawn until something runs.

## What the blind list showed
On 2026-11-19 an open-weight model (DeepSeek V4.1 Flash, hosted) listed everything dogs do without seeing this lexicon, in 28 sections it chose itself, with nothing off limits. Its names were then matched to the entries here.

| | names |
|---|---|
| names it produced | 5,831 |
| not an action of the body (traits, symptoms, human actions) | 296 |
| body actions | 5,535 |
| already an entry here, under another name or in another place | 4,970 (90%), landing on 260 of the 269 entries v0 had |
| not here, but expressible with the modes v0 had | 505 |
| flagged as needing something the parts table lacked | 60 |

Reading the 60 by hand left eight real gaps in the parts table: `tongue.suck`, `air.snore`, `gut.strain`, `gut.belch`, `gut.gland`, and a part v0 had left out entirely, `sex` (erect, tie, wink). The rest were appearances (pallor, swelling, rigor) or garbled lines. The gaps sat where v0 had a hole: it described an adult pet's day and skipped mating, birth and pups.

v0.1 adds those eight modes, 47 entries, 6 sequences, 10 requests and one posture transition.

Read the 90% as an upper bound. The same model wrote the list and did the matching, under a rule that folds same-body names together, and that rule can hide entries people name separately (rooting had been folded into nose_push).

**Held-out test of v0.1.** A second model, a local Qwen 3.8 27B, listed dog behavior without seeing any of this. It produced 1,083 names, 1,022 of them body actions. Of those, 881 (86%) matched entries here, 128 were new entries the existing modes could express, and 13 were flagged as needing something new: diving and surfacing under water, squeezing through a gap, carrying a load, and discharges, which are appearances. Its general list was cut short at nine sections, so it covered less ground than DeepSeek's.

Both checks are language models agreeing with a list made by a language model. They show the list is consistent with what has been written about dogs. They are not evidence about dogs.

Asked blind for the smallest set of parts and shared signals, the first model gave 17 controllers, one per body part, and 13 shared signals. Six of the 13 are this lexicon's manner and target under other names: posture height, stiffness, arousal, valence, attention vector, social distance. Three are travel (gait phase, drive, turn). The other four are body states sent to every part: hunger, pain or heat load, scent gradient, reproductive state.

## Rules
1. **The entry is the body, not the word.** Words are aliases. If two words name the same body they share one entry; if one word covers different bodies it splits ("down" is platz, down_hip or sploot).
2. **One language.** The most precise word from any language is kept as an alias (platz, fuss, aus); English otherwise.
3. **Resolution.** Split two behaviors when people who know dogs call them different things or expect different next moves. Merge them when they don't. That rule keeps the list as small as it can be: platz, a hip-rolled down and a sploot pass it; ten kinds of tail wag mostly don't.
4. **Channels run at once.** Support (01–03), head and mouth (04), signals (05), limbs (06), voice (07). A dog carries with its mouth while heeling with its legs, ears on a sound, tail wagging. The router assigns every channel; it doesn't pick one behavior at a time.
5. **Channels share some resources.** The airway can sniff or pant, not both at full rate (Gazit & Terkel, 2003). A mouth that's carrying can't bark or pant fully, which is why retrievers drop the ball to pant. A forepaw that's digging or pawing isn't bearing weight.
6. **Postures connect only along the graph (02).** Every edge is its own motion and gets its own test.
7. **Requests are goals, not motions (10).** Each request lists the states that count as done, and the dog chooses among them by its own costs. The choice is visible: a hot dog splooted, a keyed-up dog went to platz.
8. **Scope.** Everything real dogs do, including pathological behavior (tagged EDGE). Not things no dog does without fakery.

## Columns
- **id**: canonical name.
- **body**: what the body does. This is the definition.
- **words**: aliases: commands, slang, other languages.
- **with**: A human-absent, H human-coupled, D coupled to another dog or animal.
- **why**: the drives that select the entry.

Drive codes: REST rest · CMF comfort, temperature, grooming · ING eating, drinking · ELM elimination · EXP exploring, scent · PRY predatory sequence · PLY play · SOC bonding, greeting, appeasement, care of pups · FEA fear, escape · DEF threat, guarding · WRK handler request · SEX · RFX reflex or physiology, not chosen · EDGE pathological or rare.

## One lexicon, many dogs
Every dog uses this same lexicon. What differs is a set of numbers per dog:
- how easily each entry fires;
- how much the dog prefers it over the alternatives;
- how reliably each cue word maps to each request.

Breed sets the starting values and the individual's footage sets the rest. Breed explains only about 9% of individual behavioral variation (Morrill et al., *Science*, 2022). Working breeds differ mostly in the predatory sequence (08): herders exaggerate eye, stalk and chase, and retrievers exaggerate grab and carry with a soft mouth (Coppinger).

Predictions to check against footage of the two dogs:
- **Labrador.** Greeting with an object isn't a behavior of its own. It's the greeting drive firing while her carry threshold sits near zero. If that's right, she also grabs objects in other excited moments: visitors arriving, the leash coming out.
- **Malinois.** Weaving and middle are handler-position control: the herding loop aimed at the handler. If that's right, she also tracks and heads off other people or dogs who move away from the group.

## Confirming it's complete
1. **Coverage.** Label footage against the lexicon, every channel, every second. Anything that doesn't fit becomes a new entry. Plot new entries against hours labeled; the curve should flatten. Estimate what's still unseen from the entries seen only once or twice (the Chao1 estimator; Fagen & Goldman, 1977, sized animal repertoires this way).
2. **Sufficiency (the clip test).** Knowing the current entries and the context, the system should predict the next move as well as experienced dog people do. Where people do better, an entry is missing; find it and name it. The cues people can't put into words are entries that don't have names yet.
3. **Independent lists.** Done once with DeepSeek (above). A second list from a local model (Qwen 3.8 27B) is the held-out test of v0.1.
4. **The exam.** Build the parts, the two signals and the drives, train no named behavior, and have people who haven't seen the build name what the dog does. An entry counts when they apply its name unprompted. **Disputed:** naming is forgiving and teaches the judges the answer key. Stronger tests are in [rivals/shot-list.md](rivals/shot-list.md): one-second silent clips against real footage, predicting the next half second, and telling the dogs apart by movement, sound or pauses alone.
