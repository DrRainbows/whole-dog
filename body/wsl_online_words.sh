#!/usr/bin/env bash
# Usage: wsl_online_words.sh TRACKER_RUN REFLEX_RUN TAG
# The two spoken-words runs of 21 Nov (the Malinois's insisted jump from platz; the Lab's beer and fridge sentences),
# replayed with the same words, seeds and starts on another body, with the scramble_up reflex taking over on a fall.
set -e
cd "$HOME/dogsim" && . .venv/bin/activate
cd "$(git rev-parse --show-toplevel)/body/train"
R=${DOG_WORK:-$(git rev-parse --show-toplevel)/runs}/body-runs
python online_dog.py "$R/$1" --seconds 56 --brain mal --start down --present --seed 3 --reflex "$R/$2" --tag "_words_mal$3" \
  --say "6:platz|11:jump:1.5|21:come|31:down|38:jump|47:sit" 2>&1 | grep -E "^ONLINE|Error|Traceback|WARNING" || true
python online_dog.py "$R/$1" --seconds 60 --brain lab --start stand --present --seed 3 --reflex "$R/$2" --tag "_words_lab$3" \
  --say "8:get the beer|26:go to the fridge door, open it by tugging the rope on the door handle, and fetch me a beer|40:shake|48:down" \
  2>&1 | grep -E "^ONLINE|Error|Traceback|WARNING" || true
