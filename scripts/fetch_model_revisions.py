#!/usr/bin/env python
"""Sweep every registry model for its HF revisions (branches + tags) -> JSON.

    cd ~/github/malign-logits && uv run python scripts/fetch_model_revisions.py
    uv run python scripts/fetch_model_revisions.py --refresh   # refetch everything

Writes data/model_revisions.json. RH's ask (2026-08-10): which of our models
carry training trajectories (stage/step branches), campaign-wide, as an
artifact rather than eight ad-hoc calls.

Discipline, per the campaign ledger:
- POPULATION AT RUN TIME: the model list is read from the registry when the
  script runs (the resumable-fill doctrine; a frozen list goes stale between
  writing and running). The list actually fetched is recorded in the artifact.
- IDEMPOTENT: models already fetched successfully are skipped on re-run, so a
  rate-limited run resumes for the cost of a scan. --refresh overrides.
- GRACEFUL BACKOFF: 429s and transient HTTP errors back off exponentially
  with jitter (base 5s, cap 300s, 6 tries) and NEVER classify as a terminal
  result -- a 429 recorded as "no revisions" is the [5260] fill-attempt
  defect. Errors that persist are recorded AS errors, with the status code.
- QUOTA COURTESY: ~1.2s between calls; the census sweep shares the
  per-account quota and parallelism makes it worse ([5295]).
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "model_revisions.json")

PAUSE = 1.2          # seconds between successful calls
BACKOFF_BASE = 5.0   # first retry delay
BACKOFF_CAP = 300.0  # max single delay
MAX_TRIES = 6


def is_rate_limit(exc):
    txt = str(exc)
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code == 429 or "429" in txt or "rate limit" in txt.lower()


def is_transient(exc):
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code in (500, 502, 503, 504) or "Timeout" in type(exc).__name__


def fetch_refs(api, model_id):
    """Return a result dict; raises only on unrecoverable programmer error."""
    delay = BACKOFF_BASE
    for attempt in range(1, MAX_TRIES + 1):
        try:
            refs = api.list_repo_refs(model_id)
            branches = sorted(b.name for b in refs.branches)
            tags = sorted(t.name for t in getattr(refs, "tags", []) or [])
            return {
                "ok": True,
                "n_branches": len(branches),
                "branches": branches,
                "tags": tags,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "attempts": attempt,
            }
        except Exception as e:  # noqa: BLE001 -- classified below, never silent
            if is_rate_limit(e) or is_transient(e):
                if attempt == MAX_TRIES:
                    return {"ok": False, "error": "rate_limited_or_transient",
                            "detail": str(e)[:200], "attempts": attempt,
                            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                sleep = min(delay, BACKOFF_CAP) * (1 + random.random() * 0.25)
                kind = "429" if is_rate_limit(e) else type(e).__name__
                print(f"    {kind}; backoff {sleep:.0f}s (try {attempt}/{MAX_TRIES})",
                      flush=True)
                time.sleep(sleep)
                delay *= 2
                continue
            # Hard errors (404 dead repo, 401/403 gated) are RESULTS, not retries.
            code = getattr(getattr(e, "response", None), "status_code", None)
            return {"ok": False,
                    "error": {401: "gated", 403: "gated", 404: "dead_repo"}.get(code,
                             type(e).__name__),
                    "detail": str(e)[:200], "attempts": attempt,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return {"ok": False, "error": "unreachable"}  # not reachable; loop returns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="refetch models already present in the output")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    models = sorted({m["model_id"] for m in reg["models"]})

    existing = {}
    if os.path.exists(OUT) and not a.refresh:
        try:
            existing = json.load(open(OUT)).get("models", {})
        except Exception:
            existing = {}

    api = HfApi()
    results = dict(existing)
    todo = [m for m in models
            if a.refresh or not results.get(m, {}).get("ok", False)]
    print(f"registry models: {len(models)} | already ok: "
          f"{len(models) - len(todo)} | fetching: {len(todo)}")

    for i, mid in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {mid}", flush=True)
        results[mid] = fetch_refs(api, mid)
        time.sleep(PAUSE)

    ok = [m for m, r in results.items() if r.get("ok")]
    multi = {m: r["n_branches"] for m, r in results.items()
             if r.get("ok") and r["n_branches"] > 1}
    out = {
        "_about": ("HF revisions (branches + tags) for every registry model. "
                   "Which families carry training trajectories, as an artifact. "
                   "Idempotent: re-run resumes; --refresh refetches."),
        "_producer": "scripts/fetch_model_revisions.py",
        "_registry_models_at_run": len(models),
        "_fetched_ok": len(ok),
        "_errors": {m: r["error"] for m, r in results.items() if not r.get("ok")},
        "_multi_revision_models": dict(sorted(multi.items(),
                                              key=lambda x: -x[1])),
        "_generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}: {len(ok)}/{len(models)} ok, "
          f"{len(multi)} models with >1 branch")
    for m, n in sorted(multi.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:5d}  {m}")


if __name__ == "__main__":
    main()
