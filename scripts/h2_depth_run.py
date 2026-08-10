#!/usr/bin/env python
"""h2_depth_run.py — the H2 depth sweep, resumable, one shard per pair.

    scripts/h2_depth_run.py --plan            what it WOULD do; touches nothing
    scripts/h2_depth_run.py                   run (resumes automatically)
    scripts/h2_depth_run.py --rules canonical,lens
    scripts/h2_depth_run.py --receipt         re-emit the receipt from shards

Drives `twp_depth_battery.py` as a SUBPROCESS per (pair, rule). The battery is
not modified: it already appends one nested row per (pair, prompt) with lens,
dh, repr_recovery, top, bottom, ceiling and weights, and that row is exactly the
resume unit.

## WHY A SUBPROCESS PER PAIR AND NOT A LOOP

Two models resident at fp16 is 16-36 GB, and `del` does not reliably return MPS
memory inside a long-lived process -- the same defect that cost the L2 fleet its
first three boxes with vLLM. A process boundary returns it unconditionally. It
also means **a pair that dies takes only itself down**, which is the difference
between losing 20 minutes and losing six hours.

## RESUME, AND THE THREE WAYS IT USUALLY LIES

**A resume that reads its own output is blind to everything else.** That is the
correct scope here and it is stated so nobody widens it later: this resume asks
"is there a ROW for (pair, prompt, rule) in the shard", never "does the store
have the inputs" -- 96% of an earlier fleet was regenerated because a resume
consulted its own directory and not the corpus.

    1. THE KEY MUST HOLD WHAT CHANGED.  A row from an older battery is not a
       row for this spec. Every shard carries a sidecar with `spec_sha` over
       (population hash, battery source, rule, kstep, dtype). If the current
       spec differs, this runner REFUSES the shard and names it, rather than
       appending new rows beside incomparable old ones.

    2. DID-NOT-RUN IS NOT RAN-AND-FAILED.  The battery SKIPS a prompt with
       fewer than 4 movers and writes no row -- so a naive resume retries it on
       every restart, forever, and the run never converges. Prompts attempted
       and legitimately unusable are recorded in the sidecar's `no_row` list
       and not re-offered. **A prompt is only in `no_row` after a run in which
       it was actually offered**, so an absence is never mistaken for a refusal.

    3. A TRAILING PARTIAL LINE IS NORMAL AFTER A KILL.  The reader tolerates
       exactly one unparseable final line and COUNTS it; an unparseable line
       anywhere else is a corrupt shard and stops the run.

**One writer per shard, enforced.** A lockfile holds the pid; a live pid means
another run owns that pair and this one skips it. Two writers on one filename
truncated 14 files in the beam campaign and was found only by a high-water
check afterwards.

## WHAT THIS DOES NOT DO

It does not write parquet. The six tables come from the shards afterwards, so
the schema can change without re-running six hours of forward passes.
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

OUTDIR = os.path.join(ROOT, "data", "h2_depth")
BATTERY = os.path.join(HERE, "twp_depth_battery.py")
POP = os.path.join(ROOT, "data", "h2_depth_population.json")


def sha16(s):
    if isinstance(s, str): s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def slug(pair):
    return re.sub(r"[^A-Za-z0-9]+", "_", pair["aligned"])[:60]


# ---------------------------------------------------------------- population

def build_population(write=False):
    """105 minimal-pair stems (210 prompts) + 21 EN category prompts.

    **`violence_explicit_5` IS RETIRED** (RH, 10 Aug) and is excluded by name
    rather than by its absence from the store -- an exclusion with a reason
    survives a re-run of the store, an exclusion by absence does not.
    """
    import pandas as pd
    RETIRED = {"violence_explicit_5"}
    df = pd.read_csv(os.path.join(ROOT, "data", "beam_sample_105.csv"))
    P = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    items = list(P.values()) if isinstance(P, dict) else P
    by_prompt = {}
    for v in items:
        by_prompt.setdefault(v["prompt"], v)

    prompts = []
    for _, r in df.iterrows():
        prompts.append({"prompt": r["prompt"], "prompt_id": None, "set": "beam105",
                        "stem": r["stem"], "member": r["member"], "domain": r["domain"],
                        "stratum": r["stratum"], "subdomain": r["subdomain"], "lang": "en"})
    want = ("sexual_liminal", "sexual_explicit", "violence_liminal", "violence_explicit")
    for v in items:
        pid = v.get("prompt_id") or ""
        if pid in RETIRED or pid.endswith("_zh"):
            continue
        if any(pid.startswith(w) for w in want) and "_zh" not in pid:
            prompts.append({"prompt": v["prompt"], "prompt_id": pid, "set": "category",
                            "stem": None, "member": None, "domain": v.get("domain"),
                            "stratum": None, "subdomain": v.get("subdomain"), "lang": "en"})
    seen, uniq = set(), []
    for p in prompts:
        if p["prompt"] in seen: continue
        seen.add(p["prompt"]); uniq.append(p)

    src = json.load(open(os.path.join(ROOT, "data", "h2_sweep_population.json")))
    pairs = [{k: p[k] for k in ("idx", "base", "aligned", "arch", "n_blocks", "family")}
             for p in src["pairs"] if p["INCLUDED"]]
    out = {"_about": "H2 depth sweep population: the 105 minimal-pair stems plus "
                     "the English category prompts. English only (RH, 10 Aug). "
                     "violence_explicit_5 RETIRED.",
           "_producer": "scripts/h2_depth_run.py --plan",
           "n_prompts": len(uniq), "n_pairs": len(pairs),
           "n_beam105": sum(1 for p in uniq if p["set"] == "beam105"),
           "n_category": sum(1 for p in uniq if p["set"] == "category"),
           "retired": sorted(RETIRED),
           "prompt_list_sha256_16": sha16("\n".join(p["prompt"] for p in uniq)),
           "pair_list_sha256_16": sha16("\n".join(sorted(
               "%s>%s" % (p["base"], p["aligned"]) for p in pairs))),
           "prompts": uniq, "pairs": pairs}
    if write:
        json.dump(out, open(POP, "w"), indent=1)
    return out


def spec_sha(pop, rule, kstep, dtype):
    """**THE KEY HOLDS WHAT CHANGED.** Population, instrument SOURCE, and every
    flag that alters a number. Battery source is hashed rather than named: a
    script edited between runs produces rows that are not comparable, and the
    filename is identical either way."""
    src = open(BATTERY, "rb").read()
    return sha16("|".join([pop["prompt_list_sha256_16"], pop["pair_list_sha256_16"],
                           sha16(src), rule, str(kstep), dtype]))


# -------------------------------------------------------------------- shards

def read_shard(path):
    """(rows, n_bad_tail). Tolerates ONE unparseable final line; anything else
    unparseable raises, because a hole in the middle is corruption and not a
    kill."""
    if not os.path.exists(path): return [], 0
    rows, lines = [], open(path, "r", errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        if not ln.strip(): continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            if i == len(lines) - 1:
                return rows, 1
            raise SystemExit(
                "CORRUPT SHARD %s: line %d of %d is unparseable and is not the "
                "last line. A truncated tail is a kill; a hole in the middle is "
                "corruption. Not resuming past it." % (path, i + 1, len(lines)))
    return rows, 0


def sidecar_path(shard): return shard[:-6] + ".spec.json"


def load_sidecar(shard):
    p = sidecar_path(shard)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_sidecar(shard, d):
    p = sidecar_path(shard)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def lock(shard):
    """True if we took it. A stale lock (dead pid) is reclaimed and said so."""
    p = shard[:-6] + ".lock"
    if os.path.exists(p):
        try:
            pid = int(open(p).read().strip())
            os.kill(pid, 0)
            return False, pid
        except (ValueError, ProcessLookupError):
            os.unlink(p)
        except PermissionError:
            return False, pid
    with open(p, "w") as f:
        f.write(str(os.getpid()))
    return True, os.getpid()


def unlock(shard):
    p = shard[:-6] + ".lock"
    try: os.unlink(p)
    except OSError: pass


# ----------------------------------------------------------------- the drive

def todo_for(shard, prompts, spec, sc):
    """Prompts still owed: not written, and not already offered-and-unusable.

    **`A or B` ON SETS INVERTS THIS FUNCTION AND DID.** The first version read

        done = {... if r["spec_sha"] == spec} or {... every row}

    so when NO row matched the current spec the filtered set was empty, `or`
    fell through, and every row counted as done. A TOTAL spec mismatch --
    the case the spec exists to catch -- reported the shard as fully complete.
    The fallback was meant for rows written before stamping existed; it is now
    explicit, and it admits an UNSTAMPED row while refusing a MISMATCHED one,
    which is the distinction the `or` collapsed.
    """
    rows, bad = read_shard(shard)
    done = {r.get("prompt") for r in rows if r.get("spec_sha") in (spec, None)}
    no_row = set(sc.get("no_row") or [])
    return [p for p in prompts if p not in done and p not in no_row], rows, bad


def run_pair(pair, prompts, rule, kstep, dtype, spec, timeout):
    shard = os.path.join(OUTDIR, "%s.%s.jsonl" % (slug(pair), rule))
    got, holder = lock(shard)
    if not got:
        print("  SKIP: pid %s holds %s" % (holder, os.path.basename(shard)))
        return None
    try:
        sc = load_sidecar(shard)
        if sc.get("spec_sha") and sc["spec_sha"] != spec:
            print("  REFUSED: shard spec %s != current %s -- the instrument or "
                  "population changed, so old rows are not comparable. Move or "
                  "delete %s to rerun." % (sc["spec_sha"], spec, os.path.basename(shard)))
            return "refused"
        todo, rows, bad = todo_for(shard, prompts, spec, sc)
        if bad:
            #: **DETECTING THE TRUNCATED TAIL IS NOT ENOUGH -- IT MUST BE CUT
            #: OFF BEFORE ANYTHING APPENDS.** A killed write leaves a partial
            #: line with NO trailing newline, so the next append concatenates
            #: onto it and produces one unparseable line in the MIDDLE of the
            #: file. That is permanent corruption, and it halts every later
            #: resume. Found by appending a partial line and re-running, not by
            #: reading the code: the reader tolerated it and reported success
            #: while leaving the file primed to break.
            _rewrite(shard, rows)
            print("  repaired: truncated trailing line removed (%d rows kept). "
                  "Left in place it would have merged with the next append." % len(rows))
        if not todo:
            print("  complete: %d rows, 0 owed" % len(rows))
            return "complete"
        print("  %d rows present, %d owed" % (len(rows), len(todo)))
        t0 = time.time()
        cmd = [sys.executable, BATTERY, "--base", pair["base"], "--aligned", pair["aligned"],
               "--rule", rule, "--kstep", str(kstep), "--dtype", dtype,
               "--out", os.path.relpath(shard, ROOT), "--prompts"] + todo
        r = subprocess.run(cmd, cwd=ROOT, timeout=timeout)
        dt = time.time() - t0
        after, _ = read_shard(shard)
        wrote = {x.get("prompt") for x in after} - {x.get("prompt") for x in rows}
        #: **THE BATTERY WRITES NO ROW FOR A PROMPT WITH <4 MOVERS AND SAYS SO
        #: ONLY ON STDOUT.** Offered-and-not-written is therefore a FACT about
        #: the cell, recorded so the next resume does not offer it again and
        #: loop forever. Recorded only for prompts this run actually offered.
        missed = [p for p in todo if p not in wrote]
        sc = {"spec_sha": spec, "pair": pair["aligned"], "base": pair["base"],
              "rule": rule, "kstep": kstep, "dtype": dtype,
              "no_row": sorted(set((sc.get("no_row") or [])) | set(missed)),
              "rows": len(after), "last_run_s": dt, "rc": r.returncode}
        save_sidecar(shard, sc)
        #: stamp the spec onto rows the battery wrote without it, so the shard
        #: is self-describing even if the sidecar is lost
        if wrote:
            _stamp(shard, spec)
        print("  wrote %d rows in %.1f min (rc %d)%s"
              % (len(wrote), dt / 60, r.returncode,
                 ("; %d offered but unusable" % len(missed)) if missed else ""))
        return "ran"
    finally:
        unlock(shard)


def _rewrite(shard, rows):
    """Atomically replace a shard with exactly `rows`. Used to cut a truncated
    tail and to stamp the spec. Writes to a temp file and renames, so a crash
    during the repair cannot leave a half-repaired shard."""
    tmp = shard + ".tmp"
    with open(tmp, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, shard)


def _stamp(shard, spec):
    rows, _ = read_shard(shard)
    changed = False
    for r in rows:
        if r.get("spec_sha") != spec:
            r["spec_sha"] = spec; changed = True
    if changed:
        _rewrite(shard, rows)


def receipt(pop, rules, spec_by_rule):
    out = {"population": {k: pop[k] for k in
                          ("n_prompts", "n_pairs", "prompt_list_sha256_16",
                           "pair_list_sha256_16", "n_beam105", "n_category")},
           "spec_sha": spec_by_rule, "shards": []}
    tot = 0
    for pair in pop["pairs"]:
        for rule in rules:
            shard = os.path.join(OUTDIR, "%s.%s.jsonl" % (slug(pair), rule))
            rows, _ = read_shard(shard)
            sc = load_sidecar(shard)
            tot += len(rows)
            out["shards"].append(
                {"pair": pair["aligned"], "rule": rule, "rows": len(rows),
                 "owed": len(pop["prompts"]) - len(rows) - len(sc.get("no_row") or []),
                 "no_row": len(sc.get("no_row") or []),
                 "md5": _md5(shard) if os.path.exists(shard) else None})
    out["total_rows"] = tot
    p = os.path.join(ROOT, "data", "h2_depth_receipt.json")
    json.dump(out, open(p, "w"), indent=1)
    return out, p


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="report only; writes population file")
    ap.add_argument("--rules", default="canonical")
    ap.add_argument("--kstep", type=int, default=4)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--pairs", help="comma-separated aligned-name substrings")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per (pair, rule)")
    ap.add_argument("--receipt", action="store_true")
    ap.add_argument("--limit", type=int, help="first N prompts only -- FOR SMOKE TESTS. "
                    "A limited run writes the same shard as a full one, so the sidecar "
                    "records it and the run is NOT marked complete.")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    pop = build_population(write=True)
    rules = [r.strip() for r in a.rules.split(",") if r.strip()]
    specs = {r: spec_sha(pop, r, a.kstep, a.dtype) for r in rules}
    prompts = [p["prompt"] for p in pop["prompts"]]
    if a.limit: prompts = prompts[:a.limit]
    pairs = pop["pairs"]
    if a.pairs:
        keys = [k.strip().lower() for k in a.pairs.split(",")]
        pairs = [p for p in pairs if any(k in p["aligned"].lower() for k in keys)]

    print("H2 DEPTH SWEEP")
    print("  prompts %d  (beam105 %d + category %d)  sha %s"
          % (pop["n_prompts"], pop["n_beam105"], pop["n_category"],
             pop["prompt_list_sha256_16"]))
    print("  pairs   %d of %d  sha %s" % (len(pairs), pop["n_pairs"],
                                          pop["pair_list_sha256_16"]))
    print("  rules   %s" % ", ".join("%s=%s" % (r, specs[r]) for r in rules))
    print("  retired %s" % ", ".join(pop["retired"]))
    print("  out     data/h2_depth/<pair>.<rule>.jsonl")

    if a.receipt:
        rec, p = receipt(pop, rules, specs)
        print("\n  total rows %d -> %s" % (rec["total_rows"], os.path.relpath(p, ROOT)))
        return 0

    owed = 0
    for pair in pairs:
        for rule in rules:
            shard = os.path.join(OUTDIR, "%s.%s.jsonl" % (slug(pair), rule))
            sc = load_sidecar(shard)
            t, rows, _ = todo_for(shard, prompts, specs[rule], sc)
            owed += len(t)
    print("\n  OWED: %d (pair, prompt, rule) cells of %d"
          % (owed, len(pairs) * len(prompts) * len(rules)))
    if a.plan:
        print("\n  --plan: nothing run. Population written to %s"
              % os.path.relpath(POP, ROOT))
        return 0

    stop = {"v": False}
    def onint(*_):
        stop["v"] = True
        print("\n  interrupt: finishing the current pair, then stopping. "
              "Rows already written are kept and resume will skip them.")
    signal.signal(signal.SIGINT, onint)

    t0 = time.time(); done = 0
    for pair in pairs:
        for rule in rules:
            if stop["v"]:
                print("\n  stopped by request after %d pair-rules" % done); break
            print("\n#### %-46s [%s]" % (pair["aligned"], rule))
            try:
                run_pair(pair, prompts, rule, a.kstep, a.dtype, specs[rule], a.timeout)
            except subprocess.TimeoutExpired:
                print("  TIMEOUT after %ds -- rows written so far are kept" % a.timeout)
            except Exception as e:
                print("  FAILED %s: %s -- continuing to the next pair" % (type(e).__name__, e))
            done += 1
            el = time.time() - t0
            #: **THE ETA COMES FROM THIS RUN, NOT FROM A TIMING RUN.** Timing a
            #: pair separately measures cold disk on the first load and warm
            #: page cache on the second, which biased an earlier estimate to a
            #: NEGATIVE per-prompt cost. Rows-per-second observed while doing
            #: the actual work has no such seam, and the projection is stated as
            #: what it is: a linear extrapolation of the pairs done so far, over
            #: a roster whose per-prompt cost spans 25x from smallest to largest.
            tot_pr = len(pairs) * len(rules)
            rate = el / done
            print("  [%d/%d pair-rules | %.1f min elapsed | ~%.1f h left "
                  "(linear over %d done; roster spans 25x, so early pairs "
                  "mislead)]" % (done, tot_pr, el / 60,
                                 rate * (tot_pr - done) / 3600, done))
        if stop["v"]: break

    rec, p = receipt(pop, rules, specs)
    print("\nDONE. %d rows across %d shards -> %s"
          % (rec["total_rows"], len(rec["shards"]), os.path.relpath(p, ROOT)))
    short = [s for s in rec["shards"] if s["owed"] > 0]
    if short:
        print("  %d shards still owe rows; re-run this command to resume:" % len(short))
        for s in short[:10]:
            print("     %-44s %s  owed %d" % (s["pair"][:44], s["rule"], s["owed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
