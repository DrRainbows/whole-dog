# Which dog? (identity test)

People who know your dogs watch a few short clips at a time and pick your dog. Each clip carries one thing only: dots that move like the dog, or its sound. Every trial puts your dog among other dogs of the same breed. The results decide what the game dog gets built around (rules M3–M5 in [../decisions/001-first-build.md](../decisions/001-first-build.md)).

## Set up
1. Put clips in `clips/`, named `DOG__CARRIER__ANYTHING.mp4`.
   - Carriers: `gait-dots`, `react-dots`, `gait-canonical`, `react-canonical`, `pauses-dots`, `sound`.
   - Make them with `../footage/run.py`, trimmed to 3–5 s.
2. List the dogs in `dogs.json`. Mark your dogs `"target": true`; the others are foils.
3. Run `python make_manifest.py`. It warns when a dog has fewer than 10 clips for a carrier.
4. Open `index.html` in a browser. No server is needed; it runs from the folder.

## Run it
- Each judge enters a code, ticks the dogs they know, and works through the trials. That takes about 12 seconds a trial.
- At the end they download a CSV and send it to you.
- To send a prepared link that skips setup, use `index.html?judge=CODE&known=juno,nova`.
- Use judges who know the dogs and haven't seen anything from this project. Don't take the test yourself, because you've seen the clips.

## Score it
`python analyze.py results/*.csv` prints accuracy per carrier against chance and the M4 verdict for each. It also prints the M5 comparison between reaction and trotting clips.

## Test it
`node --test "test/*.test.js"` checks the trial planner:
- one target per trial;
- foils of the same breed;
- no reused target clips;
- balanced positions;
- reproducible plans;
- clip shortages reported.

`demo/` holds a three-dog demo built from the stand-in footage. Open it with `index.html?manifest=demo/manifest.js`.
