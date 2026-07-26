"""Progress monitor for the f20x subject-beams run. Read-only, safe to run anytime.

    uv run python scripts/f20x_progress.py
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from malign_logits import MODEL_FAMILIES
from malign_logits.probe import _get_cache
from f20x_subject_beams import ALL_PROMPTS, EXCLUDE, N_BEAMS, DEPTH

QUESTIONS = set(ALL_PROMPTS.values())
MODES = ("raw", "chat", "chatml", "chat_nosys")


def running():
    try:
        out = subprocess.run(["pgrep", "-f", "f20x_subject_beams"],
                             capture_output=True, text=True).stdout.split()
        if not out:
            return None
        pid = out[0]
        et = subprocess.run(["ps", "-p", pid, "-o", "etime="],
                            capture_output=True, text=True).stdout.strip()
        return pid, et
    except Exception:
        return None


def main():
    fams = {k: f for k, f in MODEL_FAMILIES.items()
            if getattr(f, "base", None)
            and (getattr(f, "superego", None) or getattr(f, "ego", None))
            and k not in EXCLUDE}
    arms = {}
    for k, f in fams.items():
        aligned = getattr(f, "superego", None) or getattr(f, "ego", None)
        arms[(k, "base")] = f.base
        arms[(k, "aligned")] = aligned

    c = _get_cache()
    have = set()
    for key in c.iter_beam_keys():
        if not isinstance(key, dict):
            continue
        if key.get("prompt") in QUESTIONS and key.get("max_tokens") == DEPTH \
                and key.get("n_beams") == N_BEAMS:
            have.add((key.get("model"), key.get("prompt"), key.get("mode") or "raw"))

    per_arm = {}
    for (fam, arm), mid in arms.items():
        n = sum(1 for q in QUESTIONS for m in MODES if (mid, q, m) in have)
        per_arm[(fam, arm)] = n

    full = len(QUESTIONS) * len(MODES)          # 40 cells if all four modes apply
    done_cells = sum(per_arm.values())
    total_cells = full * len(arms)
    complete = sum(1 for v in per_arm.values() if v >= len(QUESTIONS) * 3)
    started = sum(1 for v in per_arm.values() if v > 0)

    proc = running()
    print(f"process   : {'RUNNING pid ' + proc[0] + '  elapsed ' + proc[1] if proc else 'NOT RUNNING'}")
    print(f"arms      : {complete} complete / {started} started / {len(arms)} total")
    print(f"cells     : {done_cells:,} / ~{total_cells:,} stashed "
          f"({100 * done_cells / total_cells:.0f}%)")
    print(f"beams     : ~{done_cells * N_BEAMS:,} generated")

    # Rate measured BETWEEN CALLS, not since process start: cells reclaimed from
    # the stash at startup arrive in seconds and would otherwise dominate.
    snap = "/tmp/f20x_progress_snapshots.tsv"
    now = time.time()
    hist = []
    if os.path.exists(snap):
        with open(snap) as fh:
            for line in fh:
                try:
                    t, c = line.split()
                    hist.append((float(t), int(c)))
                except ValueError:
                    pass
    hist = [h for h in hist if h[1] <= done_cells]  # reset if the stash shrank
    with open(snap, "a") as fh:
        fh.write(f"{now}\t{done_cells}\n")

    prior = [h for h in hist if now - h[0] > 60]
    if prior and proc:
        t0, c0 = prior[-1]
        dc, dt = done_cells - c0, (now - t0) / 60
        if dt > 0 and dc > 0:
            rate = dc / dt
            left = (total_cells - done_cells) / rate
            print(f"rate      : {rate:.1f} cells/min measured over the last "
                  f"{dt:.0f} min ({dc} cells)")
            print(f"eta       : ~{left / 60:.1f} h remaining")
        elif dt > 0:
            print(f"rate      : no new cells in the last {dt:.0f} min")
    else:
        print("rate      : run this again in a few minutes for a measured rate")

    print("\nin progress / recent:")
    for (fam, arm), n in sorted(per_arm.items(), key=lambda x: -x[1])[:6]:
        if n:
            print(f"  {fam:16s} {arm:8s} {n:2d}/{full} cells")
    todo = [f"{k}/{a}" for (k, a), n in per_arm.items() if n == 0]
    print(f"\nnot started: {len(todo)}" + (f"  e.g. {', '.join(todo[:5])}" if todo else ""))


if __name__ == "__main__":
    main()
