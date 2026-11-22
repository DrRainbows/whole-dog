# Footage pipeline

Turns phone footage of a dog into what the identity test needs:
- dot clips (movement only), plain or redrawn on a shared skeleton;
- sound clips;
- clap and whistle times, and how fast the head turns after them;
- stride timing.

Rules and status: [../REQUIREMENTS.md](../REQUIREMENTS.md). Why this comes first: [../decisions/001-first-build.md](../decisions/001-first-build.md).

## Run
Use the environment in `.venv`, which holds DeepLabCut 3.0.2 on PyTorch with CUDA. It reaches your global packages, but a stub inside it hides the broken global TensorFlow. Your global Python is unchanged.

```
.venv\Scripts\python run.py all VIDEO.mp4 --out out
```
This writes, in `out/`:
- `NAME.tracks.npz`: 39 keypoints per frame;
- `NAME.dots.mp4` and `NAME.dots-canonical.mp4`;
- `NAME.wav`;
- `NAME.report.json`: tracking coverage, sound onsets, head-turn latencies, stride timing;
- DeepLabCut's labeled video, for checking the tracking by eye.

Tests: `python -m unittest discover -s tests -v` (10 tests on synthetic data with known answers).

## Filming
1. **Phone:** landscape, 60 fps, fixed (tripod or propped) at the dog's shoulder height, side-on, 3–5 m away. Plain background if you can; daylight.
2. **Trotting passes:** the dog trots past on a lead or to a person, at least 10 seconds in frame, in both directions. Ten or more passes, spread over at least 3 days.
3. **Reactions:** the dog standing or lying at rest. Someone out of sight claps once, or whistles, or drops a toy. Keep the camera running 3 s before and 3 s after. Ten or more, spread over days, and not more than three in a row (dogs habituate).
4. **Foils:** the same filming for at least 3 other Labradors and 3 other Malinois: friends' dogs, a training club. Without them the test can only show breed, not your dog.
5. **Footfall check:** for three trotting clips, note by hand the frame numbers where one front paw lands. That checks the automatic footfalls (M1b).

## Known limits
- Duty factor reads high when a paw slows smoothly before landing. Footfalls must be checked against hand marks (M1b).
- The stand-in clips (PLOS veterinary videos, CC BY 2.5, in `standin/`) are 12 fps and show a very slow walk. They proved the tracking but are too short and slow for stride statistics.
- The side view can't separate a head turn toward the camera from a nod; both register as head movement.
