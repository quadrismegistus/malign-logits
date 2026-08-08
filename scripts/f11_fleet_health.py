#!/usr/bin/env python
"""f11_fleet_health.py — box health, re-measured ETA, and whether credit covers it.

    scripts/f11_fleet_health.py            print a report
    scripts/f11_fleet_health.py --json     machine-readable

**LIVE STATE, NOT A LOCAL RECORD OF IT.** Instances come from the vast API, not
from `.vastai*.json` -- a state file is a snapshot of an intention at write time
and this project has already lost an hour to that. The state files are consulted
only for the ssh coordinates of boxes the API says are alive.

**THE ETA IS MEASURED EVERY RUN, NEVER CARRIED.** Rate comes from the box's own
file mtimes -- first finished model to last, divided by the count -- so a box
that slows down says so. A carried estimate is an answer, and answers do not get
re-derived.

**THE CREDIT CHECK IS THE POINT.** Projected cost is Σ(hours remaining × $/hr)
over live boxes. If credit does not cover it the run dies mid-model on a box
nobody is watching, which is the expensive way to find out.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REMOTE = "/workspace/f11_twp"
SPECS = {"box1_dense": 92, "box2_ssm": 10, "box3_70b": 2}
#: **THE DEFAULT STATE FILE NAMES NO BOX**, so roster inference by substring
#: returned None for box 1 and the report read "37/None models" -- a denominator
#: that could not fail a check because it was not there. Mapped explicitly.
STATE_ROSTER = {".vastai.json": "box1_dense",
                ".vastai.box2ssm.json": "box2_ssm",
                ".vastai.box3_70b.json": "box3_70b"}
#: a box with nothing left to do is BURNING MONEY, and that is the one state a
#: health check exists to catch. Not an error, so nothing else would flag it.
IDLE_ALERT_MIN = 10
#: a box may stall, a link may halve; ask for headroom over the point estimate
MARGIN = 1.35


class VastUnreadable(Exception):
    """The API could not be READ. NOT the same as the API saying zero."""


def vast_json(*a):
    """Parse vastai --raw output, or RAISE.

    **TWO DEFECTS IN ONE FUNCTION, AND THE SECOND MADE THE FIRST INVISIBLE.**

    `vastai show instances` prints a DEPRECATION BANNER on stdout before the
    JSON, so `json.loads(stdout)` raises. The caller wrapped that in
    `except Exception: insts = []` -- and an empty instance list renders as
    "0 live boxes, $0.00 to finish, credit $0.00 -> SUFFICIENT".

    **A HEALTH CHECK THAT CANNOT READ THE API REPORTED THAT EVERYTHING WAS
    FINE**, and it would have gone on reporting it to a 10-minute loop whose
    entire job is to notice when it is not. Empty output from a failed parse is
    indistinguishable from a clean finding -- the same shape as a glob that
    matches nothing and a checker that returns no rows.

    So: skip to the first JSON character, and RAISE rather than return empty.
    A caller that wants to tolerate this has to say so.
    """
    r = subprocess.run(["vastai"] + list(a), capture_output=True, text=True)
    if r.returncode != 0:
        raise VastUnreadable("vastai %s exited %d: %s"
                             % (" ".join(a), r.returncode,
                                (r.stderr or "").strip()[:120]))
    out = r.stdout or ""
    i = min([x for x in (out.find("["), out.find("{")) if x >= 0] or [-1])
    if i < 0:
        raise VastUnreadable("no JSON in `vastai %s` output: %s"
                             % (" ".join(a), out.strip()[:120]))
    try:
        return json.loads(out[i:])
    except Exception as e:
        raise VastUnreadable("unparseable `vastai %s`: %s" % (" ".join(a), e))


def ssh(st, cmd, t=30):
    r = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=%d" % t,
         "-p", str(st["ssh_port"]), "root@%s" % st["ssh_host"], cmd],
        capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def coords():
    """ssh host/port by instance id, from whichever state file names it."""
    import glob as g
    out = {}
    for p in g.glob(os.path.join(ROOT, ".vastai*.json")):
        if any(x in p for x in ("DESTROYED", "destroyed", "stale")):
            continue
        try:
            st = json.load(open(p))
        except Exception:
            continue
        if st.get("instance_id") and st.get("ssh_host"):
            out[str(st["instance_id"])] = (os.path.basename(p), st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    try:
        insts = vast_json("show", "instances", "--raw")
        credit = float(vast_json("show", "user", "--raw").get("credit", 0.0))
    except VastUnreadable as e:
        #: LOUD, AND NOT "SUFFICIENT". An unreadable API is a state to act on,
        #: not a quiet zero.
        print("*** CANNOT READ THE VAST API -- fleet state UNKNOWN ***")
        print("    %s" % e)
        print("    This is NOT a clean bill of health. Boxes may be running and"
              " billing.")
        if a.json:
            print(json.dumps({"error": str(e), "sufficient": None}, indent=1))
        return 3
    cmap = coords()

    rows, total_h, total_dph = [], 0.0, 0.0
    for i in insts:
        iid = str(i.get("id"))
        dph = float(i.get("dph_total") or 0)
        total_dph += dph
        name, st = cmap.get(iid, (iid, None))
        r = {"instance": iid, "state": i.get("actual_status"), "dph": dph,
             "name": name, "gpu": i.get("gpu_name"),
             "n_gpu": i.get("num_gpus")}
        #: which roster is this box on? derived from the state file's name
        key = STATE_ROSTER.get(name) or next(
            (k for k in SPECS if k.split("_")[0] in name), None)
        r["roster"] = key
        r["total"] = SPECS.get(key)
        if st and i.get("actual_status") == "running":
            #: **LABELLED FIELDS, NOT POSITIONAL ONES.** This split on
            #: whitespace across six values of varying width, and `tr '\n' ','`
            #: emits no trailing newline -- so the tmux list and the next
            #: command's output fused into ONE token, `chain,1`. The session
            #: check then read the disk figure, saw a digit, concluded "no
            #: sessions", and reported COMPLETE AND STILL BILLING for a box
            #: actively running its backfill. The 10-minute loop would have
            #: destroyed it.
            #:
            #: A positional parse is a claim about the width of every field
            #: before it. Labels do not make that claim.
            out = ssh(st, "echo N=$(ls %s/*.jsonl 2>/dev/null | wc -l); "
                          "echo FIRST=$(stat -c %%Y %s/*.jsonl 2>/dev/null "
                          "| sort -n | head -1); "
                          "echo LAST=$(stat -c %%Y %s/*.jsonl 2>/dev/null "
                          "| sort -n | tail -1); "
                          "echo NOW=$(date +%%s); "
                          "echo DISK=$(df -P / | tail -1 | awk '{print $4}'); "
                          "echo SESS=$(tmux ls 2>/dev/null | cut -d: -f1 "
                          "| tr '\\n' ' '); "
                          "echo DONE=$(grep -c 'ALL MODELS COMPLETE' "
                          "/workspace/f11.log 2>/dev/null || echo 0)"
                          % (REMOTE, REMOTE, REMOTE))
            if out:
                kv = {}
                for line in out.splitlines():
                    if "=" in line:
                        k, _, v = line.partition("=")
                        kv[k.strip()] = v.strip()
                try:
                    n = int(kv.get("N", 0))
                    r["done"] = n
                    r["_sessions"] = kv.get("SESS", "")
                    r["_complete"] = kv.get("DONE", "0") not in ("0", "")
                    r["disk_free_gb"] = round(int(kv.get("DISK", 0)) / 1e6, 1)
                    if n >= 2:
                        first, last = int(kv["FIRST"]), int(kv["LAST"])
                        now = int(kv["NOW"])
                        r["min_per_model"] = round((last - first) / 60 / (n - 1), 2)
                        r["stalled_min"] = round((now - last) / 60, 1)
                        rem = (r["total"] or n) - n
                        r["remaining"] = rem
                        r["eta_h"] = round(rem * r["min_per_model"] / 60, 2)
                        total_h = max(total_h, r["eta_h"])   # boxes run CONCURRENTLY
                        r["cost_to_finish"] = round(r["eta_h"] * dph, 2)
                    else:
                        r["note"] = "fewer than 2 models done; no rate yet"
                except Exception as e:
                    r["note"] = "parse: %s (%s)" % (type(e).__name__, kv)
            else:
                r["note"] = "ssh unreachable"
        rows.append(r)

    #: COMPLETE-BUT-ALIVE is its own state and it costs money silently
    for r in rows:
        if r.get("state") == "running" and (
                r.get("_complete") or
                (r.get("total") and r.get("done", -1) >= r["total"])):
            #: **A BOX RUNNING ITS BACKFILL IS NOT AN IDLE BOX.** The main
            #: roster's count hits its total while a chained session is still
            #: working in another directory, and a monitor that reads only the
            #: main count would tell the loop to destroy a box mid-run. The
            #: chain is asked about directly rather than inferred from a count.
            live = r.get("_sessions") or ""
            if "chain" in live or "f11" in live:
                r["alert"] = None
                r["note"] = ("main roster done; BACKFILL RUNNING (tmux: %s) "
                             "-- do not destroy" % live.replace("\n", " "))
            else:
                r["alert"] = ("COMPLETE AND STILL BILLING at $%.2f/h -- destroy it"
                              % r["dph"])
        elif r.get("stalled_min", 0) > IDLE_ALERT_MIN and \
                r.get("remaining", 0) > 0 and not r.get("_complete"):
            r["alert"] = ("no new model for %.0f min with %d left -- stalled?"
                          % (r["stalled_min"], r["remaining"]))

    #: **THE PROJECTION MUST COVER COMMITTED WORK, NOT VISIBLE WORK.** Each box
    #: carries an armed `chain` session that starts a hidden-state backfill the
    #: moment its main roster ends. Costing only the main roster reports
    #: SUFFICIENT right up to the instant the chain fires, then jumps -- and the
    #: credit alarm exists precisely to fire BEFORE that, not with it. A number
    #: that is right until the moment it matters is the shape of a check that
    #: cannot fail.
    for r in rows:
        if "min_per_model" not in r or not r.get("roster"):
            continue
        bf = os.path.join(ROOT, "data",
                          "f11_twp_backfill.%s.json" % r["roster"])
        if not (os.path.exists(bf) and "chain" in (r.get("_sessions") or "")):
            continue
        try:
            n_bf = len(json.load(open(bf))["spec"])
        except Exception:
            continue
        h = n_bf * r["min_per_model"] / 60
        r["backfill_models"] = n_bf
        r["backfill_h"] = round(h, 2)
        r["backfill_cost"] = round(h * r["dph"], 2)
        r["cost_to_finish"] = round(r.get("cost_to_finish", 0.0)
                                    + r["backfill_cost"], 2)

    proj = sum(x.get("cost_to_finish", 0.0) for x in rows)
    #: boxes without a rate still burn: charge them the wall-clock of the slowest
    unknown = [x for x in rows if x.get("state") == "running"
               and "cost_to_finish" not in x]
    proj_unknown = sum(x["dph"] * max(total_h, 0.5) for x in unknown)
    need = (proj + proj_unknown) * MARGIN
    ok = credit >= need

    alerts = [x["alert"] for x in rows if x.get("alert")]
    rep = {"alerts": alerts, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "credit": round(credit, 2), "burn_per_hour": round(total_dph, 3),
           "projected_bare": round(proj + proj_unknown, 2),
           "projected_with_margin": round(need, 2), "margin": MARGIN,
           "sufficient": ok, "boxes": rows}
    if a.json:
        print(json.dumps(rep, indent=1))
        return 0 if ok else 2

    print("FLEET  %s" % rep["utc"])
    for r in rows:
        head = "  %-22s %-9s %s x%s  $%.3f/h" % (
            r["name"][:22], r.get("state"), r.get("gpu"), r.get("n_gpu"), r["dph"])
        print(head)
        if "done" in r:
            print("      %s/%s models" % (r["done"], r.get("total", "?")), end="")
            if "min_per_model" in r:
                print("  %.2f min/model  ETA %.2f h  $%.2f  (idle %.1f min)"
                      % (r["min_per_model"], r["eta_h"], r["cost_to_finish"],
                         r["stalled_min"]), end="")
            print("   disk %.0f GB free" % r.get("disk_free_gb", 0))
            if r.get("backfill_models"):
                print("      + chained backfill: %d models, %.2f h, $%.2f "
                      "(INCLUDED above)"
                      % (r["backfill_models"], r["backfill_h"],
                         r["backfill_cost"]))
        if r.get("note"):
            print("      %s" % r["note"])
        if r.get("alert"):
            print("      *** %s" % r["alert"])
    print("\n  burn      $%.2f/h across %d live box(es)" % (total_dph, len(rows)))
    print("  to finish $%.2f bare, $%.2f at %.2fx margin"
          % (rep["projected_bare"], need, MARGIN))
    print("  credit    $%.2f   ->  %s" % (credit, "SUFFICIENT" if ok
                                          else "*** INSUFFICIENT ***"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
