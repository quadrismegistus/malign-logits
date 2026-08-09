#!/usr/bin/env python
"""f11_l2_preflight.py — name the known-bad checkpoints BEFORE a box loads one.

    scripts/f11_l2_preflight.py --pairs 6,10,17

**THIS EXISTS BECAUSE I DID NOT READ `data/model_load_environments.json`.** I
built an exclusion list from memory of a previous fleet, ran one checkpoint with
a blocking record, and wrongly excluded nine whose only recorded failure was on
a device this fleet does not use. `f11_l1_logits.py --preflight` already did
this; copying that pattern was the instruction and I skipped it.

**AN ENVIRONMENT TAG IS NOT A CAUSE, and this is the part a naive filter gets
wrong in both directions.** A failure recorded under `local_mps` may be
device-specific (OLMoE's integer `histc`, absent on MPS, fine on CUDA) or
device-INDEPENDENT (mpt-7b's repo is simply gone; deepseek, croissant and Teuken
mangle the prompt in the TOKENIZER, which no card changes). So this reads the
`cause` and classifies, rather than trusting the environment field.
"""
import argparse, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,ROOT)

#: causes that travel with the CHECKPOINT, whatever the box
PORTABLE = ("not a valid model identifier", "repo is gone", "gated repo",
            "deletes", "destroys the prompt", "normalises", "DELETES",
            "encode('a b')", "tokenizer")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pairs"); a=ap.parse_args()
    rec=json.load(open(os.path.join(ROOT,"data","model_load_environments.json")))
    per={}
    for o in rec["observations"]:
        per.setdefault(o["model_id"],[]).append(o)
    from malign_logits.registry import Registry
    pairs=Registry().base_aligned_pairs()
    if a.pairs:
        pairs=[pairs[int(i)] for i in a.pairs.split(",")]
    #: **THE CORPUS OUTRANKS THE RECORD.** A checkpoint with a complete file on
    #: disk demonstrably works in THIS environment, whatever any prior
    #: observation says. Without this the preflight blocks
    #: OLMo-2-0425-1B-DPO -- from which we hold 3,940 verified passages -- on a
    #: torch-floor observation the current profile already satisfies. A
    #: predictor contradicted by evidence is not a conservative predictor, it
    #: is a wrong one.
    import glob
    proven={os.path.basename(f)[:-len(".gen.jsonl")].replace("__","/")
            for f in glob.glob(os.path.join(ROOT,"data","f11_l2","gen","*.gen.jsonl"))
            + glob.glob("/Volumes/chambers/malign-l2/gen/*.gen.jsonl")
            if sum(1 for _ in open(f))==3940}
    #: fixes the CURRENT profile already applies -- a floor met is not a failure
    APPLIED=("torch >= 2.6","torch>=2.6","sentencepiece","protobuf")
    block=[];fixable=[]
    for p in pairs:
        for m in (p["base"],p["aligned"]):
            if m in proven: continue
            for o in per.get(m,[]):
                if o.get("outcome")=="loads": continue
                cause=(o.get("cause") or ""); fix=(o.get("fix") or "")
                if any(k in cause or k in fix for k in APPLIED):
                    fixable.append((m,cause[:50])); break
                portable=any(k.lower() in cause.lower() for k in PORTABLE)
                env=o.get("environment")
                if portable or env not in ("local_mps","grid_v3_box_initial"):
                    block.append((m,env,cause[:64],"PORTABLE" if portable else "env"))
                    break
    if proven: print("PROVEN BY CORPUS (skipped): %d checkpoints" % len(proven))
    if fixable:
        print("FIXABLE by the profile's package floors (%d), NOT blocked:" % len(fixable))
        for m,c in fixable: print("  %-44s %s" % (m[:44],c))
        print()
    if block:
        print("BLOCKED (%d) — do not send these to a box:" % len(block))
        for m,e,c,k in block: print("  %-44s [%s/%s] %s" % (m[:44],k,e,c))
    else:
        print("no blocking record for these pairs")
    return 0
if __name__=="__main__": sys.exit(main())
