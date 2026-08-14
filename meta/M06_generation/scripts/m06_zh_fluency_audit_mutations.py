"""Watch every check in the fluency audit FAIL, by breaking what it guards.

    uv run python meta/M06_generation/scripts/m06_zh_fluency_audit_mutations.py

`m06_zh_fluency_audit.py` reports five green checks. **A check observed only
passing is not a check.** registrar's category at [6161]: a vacuous test is
worse than a false stated reason, because it has a detector, the detector RUNS,
it runs green, and the green is exactly why nobody looks. Their own `ch.py`
test survived two re-aims before they found it could not fail at all.

So each mutation copies the real artifacts to a scratch tree, breaks ONE
property, and re-runs the audit expecting that check to go red. A mutation that
leaves everything green means the check is not testing what its name says.

    field-leak      write a `model` field into a batch item
    batch-cluster   re-batch so every model lands in one file
    key-order       reassign keys in model order
    rating-order    sort each batch by verdict, worst first
    drift-cancels   unbalance the per-model round mix

**THE TARGETED MUTATION IS THE POINT.** A mutation that breaks everything at
once proves nothing: it cannot distinguish a check that noticed its own
property from one that tripped on collateral damage. Each below is the
smallest edit that violates exactly one claim, and the harness reports which
OTHER checks moved so that collateral is visible rather than assumed.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
AUDIT = os.path.join(HERE, "m06_zh_fluency_audit.py")
FILES = ["zh_fluency_sample.json", "zh_fluency_sample_r2.json",
         "zh_fluency_verdicts.json", "zh_fluency_verdicts_r2.json"]
DIRS = ["zh_fluency_batches", "zh_fluency_batches_r2"]
SCORE = {"fluent": 3, "flawed": 2, "broken": 1, "not_chinese": 0}


def stage(tmp):
    for f in FILES:
        shutil.copy(os.path.join(OUTD, f), os.path.join(tmp, f))
    for d in DIRS:
        shutil.copytree(os.path.join(OUTD, d), os.path.join(tmp, d))


def batches(tmp, sfx=""):
    return sorted(os.path.join(tmp, "zh_fluency_batches%s" % sfx, f)
                  for f in os.listdir(os.path.join(tmp, "zh_fluency_batches%s" % sfx))
                  if f.startswith("batch_"))


def m_field_leak(tmp):
    p = batches(tmp)[0]
    items = json.load(open(p))
    items[0]["model"] = "01-ai/Yi-1.5-9B"
    json.dump(items, open(p, "w"), ensure_ascii=False)


def m_batch_cluster(tmp):
    """Re-batch round 1 so each model's passages land in one file."""
    truth = json.load(open(os.path.join(tmp, "zh_fluency_sample.json")))["truth"]
    items = []
    for p in batches(tmp):
        items += json.load(open(p))
    items.sort(key=lambda it: truth.get(it["key"], {}).get("model", ""))
    bs = batches(tmp)
    k = (len(items) + len(bs) - 1) // len(bs)
    for i, p in enumerate(bs):
        json.dump(items[i * k:(i + 1) * k], open(p, "w"), ensure_ascii=False)


def m_key_order(tmp):
    """Reassign round-1 keys so sorted key order IS model order."""
    sp = os.path.join(tmp, "zh_fluency_sample.json")
    doc = json.load(open(sp))
    truth = doc["truth"]
    order = sorted(truth, key=lambda k: (truth[k]["model"], k))
    remap = {old: "p%03d" % i for i, old in enumerate(order)}
    doc["truth"] = {remap[k]: v for k, v in truth.items()}
    json.dump(doc, open(sp, "w"), ensure_ascii=False)
    vp = os.path.join(tmp, "zh_fluency_verdicts.json")
    v = json.load(open(vp))
    for r in v["verdicts"]:
        r["key"] = remap.get(r["key"], r["key"])
    json.dump(v, open(vp, "w"), ensure_ascii=False)
    for p in batches(tmp):
        items = json.load(open(p))
        for it in items:
            it["key"] = remap.get(it["key"], it["key"])
        json.dump(items, open(p, "w"), ensure_ascii=False)


def m_rating_order(tmp):
    """Sort each round-1 batch worst-first, so position predicts the verdict."""
    v = json.load(open(os.path.join(tmp, "zh_fluency_verdicts.json")))["verdicts"]
    sc = {r["key"]: SCORE[r["verdict"]] for r in v}
    for p in batches(tmp):
        items = json.load(open(p))
        items.sort(key=lambda it: sc.get(it["key"], 0))
        json.dump(items, open(p, "w"), ensure_ascii=False)


def m_drift_cancels(tmp):
    """Unbalance the round mix: drop half of one model's round-2 firsts."""
    sp = os.path.join(tmp, "zh_fluency_sample_r2.json")
    doc = json.load(open(sp))
    truth = doc["truth"]
    victim = sorted({v["model"] for v in truth.values()})[0]
    drop, kept = 0, {}
    for k, v in truth.items():
        if v.get("role") == "new" and v["model"] == victim and drop < 7:
            drop += 1
            continue
        kept[k] = v
    doc["truth"] = kept
    json.dump(doc, open(sp, "w"), ensure_ascii=False)


MUTATIONS = [("field-leak", m_field_leak), ("batch-cluster", m_batch_cluster),
             ("key-order", m_key_order), ("rating-order", m_rating_order),
             ("drift-cancels", m_drift_cancels)]


def run(tmp):
    """Run the audit; a CRASH is a failure of the harness, not a red check.

    The first version discarded stderr and then json.load'd an artifact the
    audit had never written, reporting a JSONDecodeError instead of the
    traceback that caused it. A harness that hides why its subject died
    cannot distinguish "the check fired" from "the check exploded".
    """
    r = subprocess.run([sys.executable, AUDIT, "--outd", tmp, "--quiet"],
                       capture_output=True, text=True)
    out = os.path.join(tmp, "zh_fluency_audit.json")
    if not os.path.exists(out):
        raise SystemExit("audit wrote no artifact under %s\nrc=%d\n%s"
                         % (tmp, r.returncode, (r.stderr or "")[-1500:]))
    try:
        d = json.load(open(out))
    except Exception as e:
        raise SystemExit("audit artifact unparseable (%s)\nrc=%d\n%s"
                         % (e, r.returncode, (r.stderr or "")[-1500:]))
    #: collapse round:check -> check; a mutation on either round should kill it
    agg = {}
    for k, v in d["checks"].items():
        name = k.split(":", 1)[-1]
        agg[name] = agg.get(name, True) and v["pass"]
    return agg


def main():
    with tempfile.TemporaryDirectory() as base:
        clean = os.path.join(base, "clean")
        os.makedirs(clean)
        stage(clean)
        base_res = run(clean)
        print("UNMUTATED BASELINE: %s"
              % ("all green" if all(base_res.values())
                 else "NOT ALL GREEN -- %s" % base_res))
        if not all(base_res.values()):
            raise SystemExit("baseline is not clean; mutations are uninterpretable")

        print("\n%-16s %-8s %s" % ("mutation", "target", "checks that went red"))
        bad = 0
        for name, fn in MUTATIONS:
            tmp = os.path.join(base, "mut_" + name)
            os.makedirs(tmp)
            stage(tmp)
            fn(tmp)
            r = run(tmp)
            red = sorted(k for k, v in r.items() if not v)
            killed = name in red
            bad += not killed
            collateral = [k for k in red if k != name]
            print("  %-14s %-8s %-6s %s"
                  % (name, "KILLED" if killed else "SURVIVED",
                     "", ", ".join(red) or "none"))
            if collateral:
                print("  %-14s %-8s collateral: %s"
                      % ("", "", ", ".join(collateral)))
        print("\n%s" % ("EVERY CHECK WAS WATCHED TO FAIL"
                        if not bad else
                        "**%d CHECK(S) SURVIVED THEIR OWN MUTATION -- vacuous**" % bad))
        return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
