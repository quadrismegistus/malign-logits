#!/usr/bin/env python3
"""Harvest MEASURED per-checkpoint costs from twp_cloud run logs.

Writes `data/model_costs.json`, which `malign_logits.model_cost` reads. Every
number here is a wall-clock reading from a run that happened; nothing is
inferred from parameter counts.

WHAT IT READS, and why each field is trustworthy or not:

    [N/M] org/model  P/Q to do          the model boundary
    Fetching K files: 100%|..| [MM:SS<  download seconds -- REAL only under
                                        --purge, which deletes the HF cache
                                        after each model. Without --purge a
                                        second run reads a warm cache and the
                                        number is meaningless. The run's own
                                        argv is checked and the field is
                                        dropped if --purge was absent.
    Loading weights: 100%|..| [MM:SS<   weights -> GPU seconds
    P/Q  R p/s                          throughput; the LAST reading of a
                                        model is taken, because early readings
                                        include warmup

USAGE
    python scripts/harvest_run_rates.py data/raw/cloud_run_*/run*.log
    python scripts/harvest_run_rates.py --ssh 'root@ssh5.vast.ai:14480' \
                                        --remote-glob '/workspace/twp/run*.log'
    python scripts/harvest_run_rates.py --show      # print what is on file
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from malign_logits.model_cost import (COSTS_PATH, arch_class,  # noqa: E402
                                      load_costs, rate_for, rate_source)

MODEL_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)")
FETCH_RE = re.compile(r"Fetching \d+ files: *100%\|[^|]*\| *(\d+)/\1 \[(\d+):(\d+)<")
LOAD_RE = re.compile(r"Loading weights: *100%\|[^|]*\| *(\d+)/\1 \[(\d+):(\d+)<")
RATE_RE = re.compile(r"^\s*(\d+)/(\d+)\s+([\d.]+) p/s")


def parse_log(text, purge):
    """One log -> {model: {p_per_s, load_s, n_prompts}}. Last reading wins."""
    out = {}
    cur = None
    for line in text.splitlines():
        m = MODEL_RE.match(line)
        if m:
            cur = m.group(3)
            out.setdefault(cur, {"fetch_s": None, "weights_s": None,
                                 "p_per_s": None, "prompts_seen": 0})
            continue
        if cur is None:
            continue
        e = out[cur]
        mm = FETCH_RE.search(line)
        if mm and purge:
            e["fetch_s"] = int(mm.group(2)) * 60 + int(mm.group(3))
        mm = LOAD_RE.search(line)
        if mm:
            e["weights_s"] = int(mm.group(2)) * 60 + int(mm.group(3))
        mm = RATE_RE.match(line)
        if mm:
            e["p_per_s"] = float(mm.group(3))
            e["prompts_seen"] = int(mm.group(1))
    return out


def merge(acc, new, source):
    """Later observations win; every field records where it came from."""
    for model, e in new.items():
        cur = acc.setdefault(model, {})
        if e["p_per_s"] is not None:
            cur["p_per_s"] = e["p_per_s"]
            cur["p_per_s_prompts"] = e["prompts_seen"]
            cur["p_per_s_source"] = source
        fs, ws = e["fetch_s"], e["weights_s"]
        if fs is not None or ws is not None:
            cur["load_s"] = (fs or 0) + (ws or 0)
            cur["load_s_parts"] = {"fetch": fs, "weights": ws}
            cur["load_s_source"] = source
        cur["arch"] = arch_class(model)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="local run*.log paths or globs")
    ap.add_argument("--ssh", help="user@host:port to read logs from")
    ap.add_argument("--remote-glob", default="/workspace/twp/run*.log")
    ap.add_argument("--no-purge", action="store_true",
                    help="the run did NOT use --purge, so fetch times reflect "
                         "a warm cache and are DISCARDED rather than recorded")
    ap.add_argument("--out", default=COSTS_PATH)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.show:
        costs = load_costs(a.out)
        if not costs:
            print("no measurements on file at %s" % a.out)
            return 0
        print("%-58s %8s %8s %7s %s" % ("model", "p/s", "load_s", "arch", "src"))
        for m in sorted(costs):
            e = costs[m]
            print("%-58s %8s %8s %7s %s"
                  % (m[:58], e.get("p_per_s", "-"), e.get("load_s", "-"),
                     e.get("arch", "-"), e.get("p_per_s_source", "-")))
        print("\n%d models with measurements" % len(costs))
        return 0

    purge = not a.no_purge
    acc = load_costs(a.out)
    n_logs = 0

    for pat in a.logs:
        for p in sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []):
            with open(p, errors="ignore") as fh:
                acc = merge(acc, parse_log(fh.read(), purge), os.path.basename(p))
            n_logs += 1

    if a.ssh:
        host, _, port = a.ssh.partition(":")
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if port:
            cmd += ["-p", port]
        cmd += [host, "for f in %s; do echo \"@@@FILE $f\"; cat $f; done"
                % a.remote_glob]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print("ssh failed: %s" % r.stderr.strip()[:200], file=sys.stderr)
            return 1
        for chunk in r.stdout.split("@@@FILE ")[1:]:
            name, _, body = chunk.partition("\n")
            acc = merge(acc, parse_log(body, purge), name.strip())
            n_logs += 1

    if not n_logs:
        print("no logs read; nothing to do", file=sys.stderr)
        return 1

    measured = sum(1 for e in acc.values() if e.get("p_per_s"))
    by_arch = {}
    for m, e in acc.items():
        if e.get("p_per_s"):
            by_arch.setdefault(e["arch"], []).append(e["p_per_s"])

    print("read %d log(s); %d models now carry a measured rate" % (n_logs, measured))
    for arch in sorted(by_arch):
        v = sorted(by_arch[arch])
        print("  %-12s n=%-3d  min %.2f  median %.2f  max %.2f p/s"
              % (arch, len(v), v[0], v[len(v) // 2], v[-1]))
    if not purge:
        print("  fetch times DISCARDED (--no-purge: a warm cache is not a download)")

    if a.dry_run:
        print("dry run; %s not written" % a.out)
        return 0
    with open(a.out, "w") as fh:
        json.dump({"_meta": {"note": "MEASURED wall-clock from twp_cloud logs. "
                                     "Operational facts about a box and a "
                                     "scoring loop, not properties of the "
                                     "architectures.",
                             "logs_read": n_logs, "models_measured": measured},
                   "models": acc}, fh, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
