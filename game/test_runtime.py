"""Headless check of the game runtime against the research loop (online_dog.py --cpu), same dog, words and seed.

  body\\.venv\\Scripts\\python game\\test_runtime.py [TRACKER] [REFLEX]

The Malinois, starting in a down with a person present, hears the words of the 21 Nov run; prints her falls, rises,
lags and what she did, and the time per 20 ms step.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dog_runtime import DT, DogRuntime  # noqa: E402

RUNS = "runs/body-runs"
SAY = [(6, "platz", 0.0), (11, "jump", 1.5), (21, "come", 0.0), (31, "down", 0.0), (38, "jump", 0.0), (47, "sit", 0.0)]


def main():
    tracker = sys.argv[1] if len(sys.argv) > 1 else f"{RUNS}/anat_tired"
    reflex = sys.argv[2] if len(sys.argv) > 2 else f"{RUNS}/scramble_up_dog3_72M"
    t0 = time.time()
    rt = DogRuntime(tracker, reflex, dog="mal", seed=3, present=True, start="down")
    print(f"ready in {time.time() - t0:.1f} s", flush=True)
    says = list(SAY)
    t1 = time.time()
    n = int(56 / DT)
    for _ in range(n):
        while says and rt.t >= says[0][0] - 1e-9:
            _, text, urg = says.pop(0)
            rt.say(text, urg)
        rt.step()
    ms = (time.time() - t1) / n * 1000
    falls = [(f["t"], f["during"]) for f in rt.log["falls"] if f["what"] == "fall"]
    print("falls", falls)
    print("rises", rt.log["rises"], "| scrambles", rt.log["scrambles"])
    print("lags", [(x["t"], x["during"]) for x in rt.log["lags"]])
    print("events", [e for e in rt.recent_events(60) if any(w in e for w in ("person", "falls", "gets back", "yes", "not"))])
    print(f"{ms:.2f} ms per 20 ms step", flush=True)


if __name__ == "__main__":
    main()
