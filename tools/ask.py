"""Send one prompt file to an OpenAI-compatible model and save the answer.

Endpoints:
  gemini  Gemini Carbon through the Hermes proxy (start it with: hermes proxy start --provider nous)
  local   the local llama-server router (one model loaded at a time)

Usage:
  python ask.py gemini prompt.md answer.md [--model google/gemini-4-carbon] [--max 32000] [--temp 0.7]
  python ask.py local  prompt.md answer.md [--model qwen3.8-27b]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINTS = {
    "gemini": ("http://127.0.0.1:8645/v1/chat/completions", "google/gemini-4-carbon"),
    "local": ("http://127.0.0.1:8080/v1/chat/completions", "qwen3.8-27b"),
}


def clean(text):
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32).replace("�", "")


def ask(endpoint, prompt, model=None, max_tokens=32000, temperature=0.7, retries=3):
    url, default_model = ENDPOINTS[endpoint]
    body = {"model": model or default_model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=1800) as r:
                data = json.loads(r.read())
            text = clean(data["choices"][0]["message"].get("content") or "")
            if text.strip():
                return text, data.get("usage", {})
            last = "empty answer, finish=%s" % data["choices"][0].get("finish_reason")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint", choices=sorted(ENDPOINTS))
    ap.add_argument("prompt")
    ap.add_argument("out")
    ap.add_argument("--model")
    ap.add_argument("--max", type=int, default=32000)
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()
    prompt = Path(a.prompt).read_text(encoding="utf-8")
    text, usage = ask(a.endpoint, prompt, a.model, a.max, a.temp)
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(text)} chars; usage {usage}", file=sys.stderr)


if __name__ == "__main__":
    main()
