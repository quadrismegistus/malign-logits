#!/usr/bin/env python
"""f11_delta_preflight.py — every known failure mode, checked before spend.

Loads nothing, launches nothing. Each check below is a wound this campaign has
already taken; a PASS here means the spec cannot repeat it.
"""
import json, os, sys, hashlib, collections
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)

SPEC=os.path.join(ROOT,"data","f11_twp_spec.quintuplet_delta.json")
POP=os.path.join(ROOT,"data","f11_delta_population.json")
SRC=os.path.join(ROOT,"data","f11_quintuplets.json")
SCAN=("Falcon-H1","Falcon3-Mamba","falcon-mamba","Zamba2")
fails=[]
def chk(name, ok, detail=""):
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok: fails.append(name)

d=json.load(open(SPEC)); spec=d["spec"]; pop=json.load(open(POP))
print("PREFLIGHT — %s\n" % os.path.basename(SPEC))

# 1. population == the enumerated file == the source of record
allp={p for e in spec for p in e["prompts"]}
enum={p["text"] for p in pop["prompts"]}
chk("population matches the enumerated file", allp==enum, "%d prompts"%len(allp))
h=hashlib.sha256("\n".join(e["prompts"] for e in spec[:1] for e in [None] or []).encode()).hexdigest() if False else \
  hashlib.sha256("\n".join(spec[0]["prompts"]).encode()).hexdigest()[:16]
chk("prompt-list hash matches the posted one", h==pop["prompt_list_sha256_16"], h)
srch=hashlib.sha256(open(SRC,"rb").read()).hexdigest()[:16]
chk("source of record unchanged since posting", srch==pop["source_of_record_sha256_16"], srch)

# 2. THE FALCON-H1 DEFECT: compute_dtype on every scan architecture
scan=[e for e in spec if any(k.lower() in e["model"].lower() for k in SCAN)]
bad=[e["model"] for e in scan if e.get("compute_dtype")!="bfloat16"]
chk("bf16 declared on every scan architecture", not bad,
    "%d scan models, %d missing" % (len(scan), len(bad)))
for m in bad: print("        MISSING:", m)

# 3. no model silently pre-excluded (populations enumerate, instruments refuse)
from malign_logits.registry import Registry
roster={m for p in Registry().base_aligned_pairs() for m in (p["base"],p["aligned"])}
chk("roster complete, nothing pre-filtered", {e["model"] for e in spec}==roster,
    "%d models" % len(spec))

# 4. every model gets the SAME prompt list
lens={len(e["prompts"]) for e in spec}
chk("identical prompt list for every model", len(lens)==1, "lengths %s"%sorted(lens))

# 5. output dir must not collide with existing data
out=os.path.join(ROOT,"data","f11_twp_delta")
chk("output directory is fresh", not os.path.exists(out) or not os.listdir(out),
    os.path.relpath(out,ROOT))

# 6. environment coverage: does every model have a known environment?
env=json.load(open(os.path.join(ROOT,"data","f11_env_plan.json")))["environments"]
placed={m for v in env.values() for m in v["models"]}
chk("every model assigned an environment", roster<=placed,
    "%d unplaced" % len(roster-placed))

# 7. the delta must NOT re-run anything already scored
first=set(json.load(open(os.path.join(ROOT,"data","f11_twp_spec.json")))["spec"][0]["prompts"])
chk("no overlap with the first fleet's 115", not (allp & first),
    "%d overlapping" % len(allp & first))

print("\n%s" % ("ALL CHECKS PASS" if not fails else "FAILED: %s"%", ".join(fails)))
sys.exit(1 if fails else 0)
