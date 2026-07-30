"""The canonical model file: one row per model, typed edges beside it.

    uv run .venv/bin/python scripts/build_model_registry.py --dry-run
    uv run .venv/bin/python scripts/build_model_registry.py

Writes `data/model_registry.json` on the pattern of `prompt_categorisation.json`
(RH's instruction, docket [983]-[988]).

WHY IT EXISTS. Model facts lived in five places and disagreed: MODEL_FAMILIES,
a JSON cache of it, registry.py's NICKNAMES, cjk_coverage.csv, and the docket.
The cache was written on 26 June and `Registry.__init__` reads it IF IT EXISTS,
so a stale file permanently outranked the live taxonomy -- 59 models against
112, covering 41 of the 103 in the frozen spec. That is a cache-invalidation
bug wearing a knowledge gap's clothes, and this file is regenerated rather than
read-if-present for exactly that reason.

MEASURED FIELDS NAME THEIR PRODUCERS ([984].1). Nothing here takes a fresh
ad-hoc read where a sweep exists. `_provenance` maps every measured field to the
script and artifact that produced it, WITH the library versions the sweep ran
under, because tonight established that tokenizer measurements are
version-dependent -- transformers refuses .bin below torch 2.6, and the same
checkpoints load on one machine and not another.

THREE SOURCE KINDS, and the distinction is the point:

    declared    a fact about OUR taxonomy      family, position, architecture
    derived     computed from a declared fact  stage from the model name
    measured    a fact about the MODEL         vocab_size, weights_format, cjk_tier

An architecture claim was asserted from a spec read and was wrong; a stage field
says `dpo` for all four archangel arms because it was derived from a family
label rather than from the name that carries the method. Marking which fields
describe our declarations and which describe the models is what separates those
two failures from each other.

STATUS CARRIES ITS REASON AND ITS SCOPE. `"100 of 103"` becomes a QUERY over
`status`, not a sentence someone must remember to write -- and every EXCLUDED
row carries `pending_repair`, because tonight needed the difference between
"excluded" and "excluded until the repair pass" twice in one evening.

TWO LISTS. A model has several outgoing edges at once -- its training parent AND
its size sibling AND its data-ablation twin -- so relations cannot be fields on
the row. Direction is declared per relation type in `_schema`, because an edge
label that does not state its convention invites a silent inversion.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES, PATH_DATA  # noqa: E402
from malign_logits.registry import NICKNAMES  # noqa: E402

OUT = os.path.join(PATH_DATA, "model_registry.json")
# HAND-CURATED FACTS NO ARTIFACT CAN REGENERATE -- org_type, country, scale.
# The first run of this builder DESTROYED them: the file it overwrote was both a
# stale derived cache AND a store of hand-entered values, and "regenerate, never
# read-if-exists" is right for the first and fatal for the second. They were
# recovered from git and now live in their own file, which is merged and never
# rebuilt. `country` is not incidental -- the Chinese-family contrast and the
# country confound both read it.
CURATED = os.path.join(PATH_DATA, "model_curated.json")
SPEC = os.path.join(PATH_DATA, "grid_spec.json")

# ---- measured inputs: (artifact, producer script, key column) ----
SWEEPS = {
    "weights": ("weights_audit.csv", "scripts/weights_audit.py"),
    "cjk": ("cjk_coverage.csv", "scripts/build_cjk_coverage.py"),
    "bos": ("bos_resolution.csv", "scripts/bos_resolution_sweep.py"),
    "roundtrip": ("tokenizer_roundtrip.csv", "scripts/tokenizer_roundtrip_sweep.py"),
}

POSITIONS = ("base", "ego", "superego", "reinforced_superego",
             "reasoning", "reasoning_base")

# POSITION IS OUR STRUCTURAL SLOT; STAGE IS THE PROCEDURE THAT PRODUCED IT.
# They are not the same field and collapsing them is what made all four
# archangel arms read `dpo`.
POSITION_STAGE = {"base": "base", "ego": "sft", "superego": "dpo",
                  "reinforced_superego": "rlvr", "reasoning": "reasoning",
                  "reasoning_base": "base"}

# DERIVED FROM THE MODEL NAME, which carries the method the family label lost.
# archangel_sft-kto_pythia2-8b is a KTO model; the position field says superego
# for all four, and superego->dpo then erases the one variable the cell exists
# to vary.
#
# `\b` DOES NOT WORK HERE and the first version of this line silently did
# nothing: the ids read `archangel_sft-kto_pythia2-8b`, and `_` is a word
# character, so there is no boundary between `kto` and `_`. All four arms kept
# reading `dpo` and the fix looked applied. A negative lookahead for a letter is
# the correct assertion.
METHOD_IN_NAME = re.compile(r"sft-(dpo|kto|ppo|slic)(?![a-z])", re.I)

# DECLARED. Not inferable from any artifact we hold, and asserting it from a
# spec read is the error this column exists to make visible.
ARCHITECTURE = {
    "tiiuae/falcon-mamba-7b": "ssm",
    "tiiuae/falcon-mamba-7b-instruct": "ssm",
    "tiiuae/Falcon3-Mamba-7B-Base": "ssm",
    "tiiuae/Falcon3-Mamba-7B-Instruct": "ssm",
    "tiiuae/Falcon-H1-1.5B-Base": "hybrid",
    "tiiuae/Falcon-H1-1.5B-Instruct": "hybrid",
    "tiiuae/Falcon-H1-7B-Base": "hybrid",
    "tiiuae/Falcon-H1-7B-Instruct": "hybrid",
    "RWKV/rwkv-4-7b-pile": "linear_attn_rnn",
    "RWKV/rwkv-raven-7b": "linear_attn_rnn",
    "allenai/OLMoE-1B-7B-0125": "moe",
    "allenai/OLMoE-1B-7B-0125-SFT": "moe",
    "allenai/OLMoE-1B-7B-0125-DPO": "moe",
    "allenai/OLMoE-1B-7B-0125-Instruct": "moe",
}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def load_csv(fn):
    p = os.path.join(PATH_DATA, fn)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return {r["model"]: r for r in csv.DictReader(fh)}


def params_from_name(mid):
    s = mid.lower()
    m = re.findall(r"[-_/](\d+(?:\.\d+)?)b(?![a-z0-9])", s) or \
        re.findall(r"(\d+(?:\.\d+)?)b(?![a-z0-9])", s)
    if m:
        return f"{m[-1]}B"
    m = re.findall(r"(\d+)m(?![a-z0-9])", s)
    return f"{m[-1]}M" if m else ""


def build():
    spec = json.load(open(SPEC))
    spec_rows = spec["spec"] if isinstance(spec, dict) else spec
    in_spec = {r["model"] for r in spec_rows}
    sw = {k: load_csv(v[0]) for k, v in SWEEPS.items()}
    cur = (json.load(open(CURATED))["models"]
           if os.path.exists(CURATED) else {})

    rows, relations = {}, []
    for fam_key, fam in sorted(MODEL_FAMILIES.items()):
        arms = {p: getattr(fam, p, None) for p in POSITIONS}
        base_id = arms.get("base")
        for pos, mid in arms.items():
            if not mid:
                continue
            r = rows.setdefault(mid, {"model_id": mid})
            # A MODEL CAN SIT IN TWO FAMILIES (a shared base). The row records
            # the first family that claims it and the rest become edges, rather
            # than the row silently taking whichever family sorted last.
            if "family" not in r:
                w = sw["weights"].get(mid, {})
                cj = sw["cjk"].get(mid, {})
                bo = sw["bos"].get(mid, {})
                rt = sw["roundtrip"].get(mid, {})
                method = METHOD_IN_NAME.search(mid)
                r.update(
                    nickname=NICKNAMES.get(mid, ""),
                    family=fam_key, position=pos,
                    stage=(method.group(1).lower() if method
                           else POSITION_STAGE.get(pos, "")),
                    org=mid.split("/")[0], params=params_from_name(mid),
                    architecture=ARCHITECTURE.get(mid, "transformer"),
                    tokenizer_class=rt.get("tokenizer", ""),
                    vocab_size=int(cj["vocab_size"]) if cj.get("vocab_size") else None,
                    cjk_tier=cj.get("tier", ""),
                    cjk_chars=int(cj["cjk_chars"]) if cj.get("cjk_chars") else None,
                    weights_format=w.get("weights_format", ""),
                    index_present=w.get("index_present", ""),
                    needs_torch=w.get("needs_torch", ""),
                    bos_stratum=bo.get("stratum", ""),
                    bos_resolver=bo.get("resolver", ""),
                    loader_override=(bo.get("loader", "")
                                     if bo.get("loader") not in ("", "AutoTokenizer")
                                     else ""),
                    in_grid_spec=mid in in_spec,
                    status="ACTIVE", exclusion_reason="", pending_repair=None,
                )
                r.update(cur.get(mid, {}))          # curated wins; never derived
            else:
                relations.append({"parent": mid, "child": fam_key,
                                  "relation": "also_member_of"})
        # training-axis edges, from the family's own slots
        for pos in ("ego", "superego", "reinforced_superego", "reasoning"):
            if base_id and arms.get(pos):
                relations.append({"parent": base_id, "child": arms[pos],
                                  "relation": f"{POSITION_STAGE[pos]}_of"})
    return rows, relations, sw


def lateral(rows, relations):
    """The edges ModelFamily has no slot for -- and the schema's payoff."""
    by_base = defaultdict(list)
    for mid, r in rows.items():
        for fk, fam in MODEL_FAMILIES.items():
            if fam.name == r["family"] or fk == r["family"]:
                if getattr(fam, "base", None):
                    by_base[fam.base].append(mid)
    # same_base_as: the separator cell becomes a query instead of four reads
    for base, members in by_base.items():
        sibs = sorted(set(members) - {base})
        for i, a in enumerate(sibs):
            for b in sibs[i + 1:]:
                relations.append({"parent": a, "child": b,
                                  "relation": "same_base_as"})
    # data_ablation_of: the pair that can say WHICH training data
    for mid in rows:
        if "no-" in mid and "Tulu" in mid:
            full = re.sub(r"-no-[a-z]+-data", "", mid)
            if full in rows:
                relations.append({"parent": full, "child": mid,
                                  "relation": "data_ablation_of"})
    return relations


def main(a):
    rows, relations, sw = build()
    relations = lateral(rows, relations)

    # ---- EXCLUSIONS ARE PASS-SCOPED ([984].3) ----
    # A model requiring torch >= 2.6 is not defective; it was unloadable ON THE
    # BOX THE v3 GRID RAN ON, which had 2.5.1. Writing that as a property of the
    # model would be the mistake this field exists to prevent -- tonight every
    # exclusion looked permanent and every one was a version floor.
    n_exc = 0
    for mid, r in rows.items():
        if not r.get("in_grid_spec"):
            continue
        w = sw["weights"].get(mid, {})
        if w.get("needs_torch") == "2.6":
            mixed = w.get("weights_format") == "mixed"
            r["status"] = "EXCLUDED"
            r["exclusion_reason"] = (
                "safetensors shards present but index absent; falls back to the "
                ".bin index, refused below torch 2.6"
                if mixed else
                "bin-only checkpoint; refused below torch 2.6")
            r["pending_repair"] = True
            r["excluded_from"] = "grid_v3"
            n_exc += 1

    # dedupe; an edge asserted twice is not two edges
    seen, uniq = set(), []
    for e in relations:
        k = (e["parent"], e["child"], e["relation"])
        if k not in seen and e["parent"] in rows and e["child"] in rows:
            seen.add(k)
            uniq.append(e)

    doc = {
        "_schema": {
            "note": ("Regenerate; never read-if-exists. A cache that can outrank "
                     "its source is how 59 models shadowed 112 for five weeks."),
            "fields": {
                "family": {"source": "declared", "from": "MODEL_FAMILIES"},
                "position": {"source": "declared",
                             "values": list(POSITIONS)},
                "stage": {"source": "derived",
                          "values": ["base", "sft", "dpo", "kto", "ppo", "slic",
                                     "rlvr", "reasoning", "instruct"],
                          "note": ("from the MODEL NAME where it carries the "
                                   "method, else from position. The four "
                                   "archangel arms differ only by method and a "
                                   "position-derived stage called all four dpo.")},
                "architecture": {"source": "declared",
                                 "values": ["transformer", "ssm", "hybrid",
                                            "linear_attn_rnn", "moe"]},
                "vocab_size": {"source": "measured"},
                "cjk_tier": {"source": "measured"},
                "weights_format": {"source": "measured",
                                   "values": ["safetensors", "bin", "mixed",
                                              "none", "unknown"]},
                "index_present": {"source": "measured",
                                  "note": ("sharded checkpoints only; decides "
                                           "loadability where format does not")},
                "needs_torch": {"source": "measured",
                                "note": "transformers refuses .bin below 2.6"},
                "status": {"source": "declared",
                           "values": ["ACTIVE", "EXCLUDED"]},
                "pending_repair": {"source": "declared",
                                   "note": ("EXCLUDED rows only. 'excluded' and "
                                            "'excluded until the repair pass' "
                                            "are different facts.")},
            },
            "relations": {
                "direction": "child is the <relation> of parent",
            "field_names": ["parent", "child", "relation"],
                "values": ["sft_of", "dpo_of", "rlvr_of", "reasoning_of",
                           "same_base_as", "data_ablation_of", "also_member_of"],
                "symmetric": ["same_base_as"],
            },
        },
        "_provenance": {
            "built_by": "scripts/build_model_registry.py",
            "head": sh("git rev-parse --short HEAD"),
            "measured_from": {
                k: {"artifact": f"data/{v[0]}", "producer": v[1],
                    "rows": len(sw[k])} for k, v in SWEEPS.items()},
            "measurement_context": {
                "note": ("tokenizer and weight measurements are library-version "
                         "dependent; the sweeps ran locally, the v3 grid ran on "
                         "transformers 5.14.1 / torch 2.5.1"),
            },
        },
        "models": [rows[k] for k in sorted(rows)],
        "relations": sorted(uniq, key=lambda e: (e["relation"], e["parent"])),
    }

    print(f"models {len(doc['models'])}   relations {len(doc['relations'])}")
    print(f"  in the frozen spec: {sum(1 for r in doc['models'] if r['in_grid_spec'])}"
          f" / {len(json.load(open(SPEC))['spec'])}")
    import collections
    print(f"  stages: {dict(collections.Counter(r['stage'] for r in doc['models']))}")
    arch = {r["model_id"].split("/")[-1]: r["stage"]
            for r in doc["models"] if "archangel_sft-" in r["model_id"]}
    print("  archangel (the separator cell -- these MUST differ):")
    for k, v in sorted(arch.items()):
        print(f"    {k:<38}{v}")
    if len(set(arch.values())) != len(arch):
        print("    !! STAGES NOT DISTINCT -- the method is the experiment here")
    print(f"  relations: {dict(collections.Counter(e['relation'] for e in doc['relations']))}")
    act = sum(1 for r in doc["models"] if r["in_grid_spec"] and r["status"] == "ACTIVE")
    print(f"\n  COMPLETENESS IS A QUERY, not a sentence: "
          f"{act} of {sum(1 for r in doc['models'] if r['in_grid_spec'])} ACTIVE"
          f"   ({n_exc} EXCLUDED, all pending_repair)")
    if a.dry_run:
        print("\nDRY RUN -- nothing written")
        return 0
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(ap.parse_args()))
