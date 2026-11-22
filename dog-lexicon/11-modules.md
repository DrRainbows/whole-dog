# Modules: the lexicon reduced

Every body entry in files 01–09 and 13 is built from three things:

1. **Modes** of a small set of body parts. Each part is its own small controller.
2. **Manner**: four numbers sent to every part at once.
3. **Target**: what the dog is attending to, also sent to every part at once.

[12-recipes.md](12-recipes.md) writes each entry this way. `check_recipes.py` verifies the recipes against this table and counts the result.

## Body parts and their modes
| part | mode | what it is |
|---|---|---|
| fore | stand | forelegs straight, bearing weight |
| fore | down | elbows and forearms on the ground |
| fore | fold | forelegs folded back under the chest |
| fore | lift | one or both forelegs held off the ground |
| fore | prop | forepaws resting on something raised |
| fore | reach | one or both paws go to a target: touch, bat, pin, hook, clasp |
| fore | stroke | repeated strokes against a surface: dig, scrape |
| fore | step | stepping, as part of travel |
| hind | stand | hind legs straight, bearing weight |
| hind | fold | folded under with the rump down |
| hind | hip | rolled to one side |
| hind | splay | stretched straight back |
| hind | squat | half folded and spread |
| hind | lift | one hind leg raised |
| hind | stroke | repeated strokes: scratch, kick back |
| hind | step | stepping, as part of travel |
| hind | thrust | explosive extension: the push of a leap |
| trunk | level | spine horizontal |
| trunk | upright | spine vertical |
| trunk | side | lying on one side |
| trunk | back | lying on the back |
| trunk | curl | spine bent into a ring |
| trunk | arch | back arched up |
| trunk | twist | fast rotation along the spine: shake-off, righting, wriggle |
| trunk | press | trunk pressed or rubbed against something |
| trunk | thrust | pelvic thrusting |
| head | aim | head pointed at the target |
| head | avert | head pointed away from the target |
| head | tilt | head rolled to one side |
| head | shake | fast side-to-side shake |
| head | push | nose or head pushed against something |
| head | rest | chin laid on something |
| head | rub | face dragged along a surface |
| eyes | fix | eyes on the target |
| eyes | avert | eyes off the target |
| eyes | switch | eyes moving between two targets |
| eyes | lids | how open the lids are: wide, soft, squint, closed, blink |
| eyes | brow | inner brows raised |
| ears | set | ear position, each ear on its own |
| ears | flick | quick twitch |
| tail | carry | height, sideways set and stiffness of the tail |
| tail | wag | side-to-side or circling swing |
| tail | balance | counterweight in turns, jumps and water |
| jaw | open | mouth opens, lets go |
| jaw | snap | fast close |
| jaw | grip | sustained hold at some pressure |
| jaw | work | rhythmic closing: chew, gnaw, nibble |
| tongue | lick | strokes of the tongue on a target |
| tongue | lap | scooping liquid |
| tongue | loll | tongue hanging out |
| tongue | suck | sustained suction on a teat, a blanket or the flank |
| lips | forward | lip corners short and pushed forward; the whiskers fan forward with them |
| lips | back | lip corners pulled back; the whiskers flatten with them |
| lips | lift | lips raised off the teeth |
| lips | slack | lips loose |
| air | breathe | quiet breathing |
| air | pant | rapid shallow breaths through the open mouth |
| air | sniff | short rapid breaths through the nose |
| air | yawn | one long deep breath with the jaw wide |
| air | blow | one forced breath out: sigh, huff, snort |
| air | spasm | sneeze, reverse sneeze, cough, hiccup, retch |
| air | snore | noisy breathing in sleep: rattle, whistle, pause and snort |
| voice | bark | pulsed |
| voice | growl | low, rough, sustained |
| voice | whine | high, clear, sustained |
| voice | howl | long, clear, rising and falling |
| voice | yelp | abrupt and high |
| voice | moan | low and clear |
| gut | swallow | swallowing |
| gut | void | passing urine, feces or gas |
| gut | heave | vomiting or bringing food back up |
| gut | drool | saliva flow |
| gut | strain | bearing down to expel: birth, hard stool |
| gut | belch | air released up from the stomach |
| gut | gland | anal-sac secretion released |
| skin | bristle | hair raised |
| skin | shiver | fine trembling |
| skin | twitch | local ripple of skin or muscle |
| skin | pupil | pupil size |
| sex | erect | penis swollen and extended |
| sex | tie | pair locked together at the genitals |
| sex | wink | vulva lifting and tightening in rhythm |
| move | go | travel: forward or back, sideways, turning. The gait falls out of speed and manner |
| move | leap | one ballistic jump |

## Manner: four numbers, sent to every part
| signal | range | what it changes |
|---|---|---|
| size | big ↔ small | height of the body, head and tail; raised hair; pitch of the voice (big is low) |
| aim | toward ↔ away | weight forward or back; ears, eyes, lips, head and path pointed at the target or away from it |
| tension | stiff ↔ loose | muscle tone everywhere: locked and still, or soft, curvy and bouncy |
| energy | fast ↔ slow | tempo, amplitude and repetition of whatever each part is doing |

One shape with three manners gives three behaviors. `fore.down hind.stand` is play_bow when toward, loose and fast; stretch_bow when loose and slow; prayer when stiff and slow.

## Target: one thing, sent to every part
none · self · object · place · person · dog · prey · scent · sound · two at once.

With the target comes a distance rule: close in, hold a distance, hold a position, or leave.

## The posture grid
Postures are the fore pair's state crossed with the hind pair's state.

| | fore.stand | fore.down | fore.lift | fore.prop |
|---|---|---|---|---|
| **hind.stand** | stand | play_bow, stretch_bow | stand_hind | forepaws_up |
| **hind.fold** | sit | platz, down_sphinx | sit_pretty | sit_prop |
| **hind.hip** | sit_hip | down_hip | – | – |
| **hind.splay** | stretch_rear | sploot | – | – |

The grid predicted an entry the list had missed: sit_prop, sitting with the forepaws on a lap or ledge. It has been added to 01.

## What this predicts
**All four are disputed** by the opposition in [rivals/README.md](rivals/README.md). In short: the body is one coupled thing, not parts with their own controllers; posture is continuous and asymmetric; closing on a target describes pursuit only; and the deciding layer also has to carry intention, several targets, bearing and speed. A second model asked for the smallest set of shared signals gave 13, not this file's two. Treat the list below as claims to test, not findings.

1. **Natural posture changes move one leg pair at a time.** Stand to sit changes only the hind pair; sit to platz changes only the fore pair. Changing both at once (the sport drop) has to be trained.
2. **A sequence is one target getting closer.** Each part joins when the target comes within its reach: eyes and ears first, then legs, then nose, then jaw, then forepaws, then gut. The predatory chain, a greeting and guarding a bone all come out in that order with nothing scripted. Breed is a gain on each link (Coppinger).
3. **Holding a distance instead of closing gives the rest.** Hold a distance while moving and you get herding's orbit. Hold a position on the handler and you get heel and middle. Hold loosely and you get shadowing. Hold a spot between two targets and you get guarding and splitting.
4. **The deciding layer's output is small.** At each moment: one target, one distance rule, four manner numbers.

## The exam
Build the parts, the manner signal, the target signal and the drives. Train no named behavior. Then have people who haven't seen the build name what the dog does. The lexicon is the answer key: an entry counts when they apply its name unprompted.

## Where the pieces come from
None of the pieces is new. Darwin (1872) described opposite moods as opposite settings across the whole body, which is the manner signal. Hildebrand (1965) showed gaits are one stepping pattern at different timings. Sherrington and Grillner found the pattern generators in the spinal cord. Hediger described the distance rules. Coppinger described breed as gains on the predatory chain. What hasn't been done is assembling them into one animal and grading it against a complete list of names it was never taught.
