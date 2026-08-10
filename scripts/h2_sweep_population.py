#!/usr/bin/env python
"""h2_sweep_population.py — the H2 depth sweep population, ENUMERATED.

    scripts/h2_sweep_population.py            report + gate counts
    scripts/h2_sweep_population.py --write    emit data/h2_sweep_population.json

Written because [5221].3 requires a plan to carry its population as STRINGS with
a hash, not as a description. "All the models on disk" is a description; this is
the list, with every exclusion named and counted.

## THE FOUR GATES, APPLIED IN THIS ORDER AND ALL RECORDED

    LOCAL       both arms have a snapshot. Never downloads.
    ARCH        the weight patch reaches `model.model.layers`. Architectures
                that do not expose it are EXCLUDED AND NAMED rather than
                silently skipped -- and the SSM/Mamba/RWKV cases are excluded
                on a stronger ground: they have no attention blocks in the
                sense the patch assumes, so a block swap there is not the same
                operation and must not be pooled with one that is.
    PREFLIGHT   data/model_load_environments.json, matched on CAUSE not on the
                environment tag (mpt's repo is gone everywhere; deepseek and
                croissant mangle the prompt in the TOKENIZER, which no card
                changes).
    CELLS       the word sets come from Step/Cell/movement, so a (pair, prompt)
                with no stored cell contributes NOTHING. Counted per pair, and
                pairs below a floor are reported rather than dropped quietly.

**A GATE IS A COLUMN, NOT A FILTER.** Every pair is written with its gate
verdicts, so an exclusion is a query and is reversible. The counts on the face
are what the plan cites; the file is what a recount reads.
"""
import argparse, hashlib, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

#: architectures whose blocks live at model.model.layers, verified by reading
#: config.json rather than by family name
BLOCKS_AT_MODEL_LAYERS = {
    "LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    "MistralForCausalLM", "Olmo2ForCausalLM", "OlmoForCausalLM",
    "Gemma2ForCausalLM", "Phi3ForCausalLM", "StableLmForCausalLM",
    "GraniteForCausalLM", "SmolLM3ForCausalLM", "MiniCPMForCausalLM",
    "CohereForCausalLM", "Starcoder2ForCausalLM",
}
#: excluded on a STRONGER ground than the path: no attention blocks in the
#: sense the weight patch assumes
NO_ATTENTION = ("mamba", "rwkv", "zamba", "recurrentgemma", "falconh1", "hybrid")
PORTABLE_FAIL = ("not a valid model identifier", "gated repo", "deletes",
                 "destroys the prompt", "normalises", "encode('a b')")


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-cells", type=int, default=10,
                    help="prompts with a stored cell needed to keep a pair")
    a = ap.parse_args()

    from malign_logits.registry import Registry
    from malign_logits.weightdelta import snapshot_dir
    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.movement import CANONICAL

    P = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    items = list(P.values()) if isinstance(P, dict) else P
    want = ("sexual_liminal", "sexual_explicit", "violence_liminal", "violence_explicit")
    prompts = []
    for v in items:
        pid = v.get("prompt_id") or ""
        if any(pid.startswith(w) for w in want) or pid.startswith("e7_"):
            prompts.append({"prompt": v["prompt"], "prompt_id": pid,
                            "domain": v.get("domain"), "subdomain": v.get("subdomain")})
    seen, uniq = set(), []
    for p in prompts:
        if p["prompt"] not in seen:
            seen.add(p["prompt"]); uniq.append(p)
    prompts = uniq

    rec = json.load(open(os.path.join(ROOT, "data", "model_load_environments.json")))
    per_obs = {}
    for o in rec.get("observations", []):
        per_obs.setdefault(o.get("model_id"), []).append(o)

    rows, tally = [], Counter()
    for i, pr in enumerate(Registry().base_aligned_pairs()):
        b, al = pr["base"], pr["aligned"]
        r = {"idx": i, "base": b, "aligned": al, "stage": pr.get("stage"),
             "family": pr.get("family"), "ambiguous": pr.get("ambiguous"),
             "gates": {}}
        sb, sa = snapshot_dir(b), snapshot_dir(al)
        r["gates"]["local"] = bool(sb and sa)
        arch = None
        if sb and os.path.exists(os.path.join(sb, "config.json")):
            cfg = json.load(open(os.path.join(sb, "config.json")))
            arch = (cfg.get("architectures") or [None])[0]
            r["n_blocks"] = cfg.get("num_hidden_layers")
        r["arch"] = arch
        low = (b + " " + al + " " + (arch or "")).lower()
        r["gates"]["has_attention"] = not any(k in low for k in NO_ATTENTION)
        r["gates"]["arch_patchable"] = bool(arch in BLOCKS_AT_MODEL_LAYERS)
        blocked = None
        for m in (b, al):
            for o in per_obs.get(m, []):
                if o.get("outcome") == "loads":
                    continue
                c = o.get("cause") or ""
                if any(k.lower() in c.lower() for k in PORTABLE_FAIL):
                    blocked = "%s: %s" % (m, c[:60]); break
            if blocked: break
        r["gates"]["preflight_clean"] = blocked is None
        r["preflight_reason"] = blocked
        n_cells = None
        if r["gates"]["local"] and r["gates"]["arch_patchable"] and not blocked:
            try:
                st = Step(Checkpoint(b), Checkpoint(al))
                n_cells = 0
                for p in prompts:
                    try:
                        c = st.cell(p["prompt"]); m = c.movement(CANONICAL)
                        if len(m.fallers) + len(m.risers) >= 4: n_cells += 1
                    except Exception:
                        pass
            except Exception as e:
                r["cells_error"] = type(e).__name__
        r["n_cells"] = n_cells
        r["gates"]["enough_cells"] = bool(n_cells is not None and n_cells >= a.min_cells)
        r["INCLUDED"] = all(r["gates"].values())
        rows.append(r)
        for g, v in r["gates"].items():
            if not v: tally[g] += 1; break

    inc = [r for r in rows if r["INCLUDED"]]
    print("H2 SWEEP POPULATION\n")
    print("  prompts enumerated              %d   sha256/16 %s"
          % (len(prompts), sha16("\n".join(p["prompt"] for p in prompts))))
    print("  pairs in Registry               %d" % len(rows))
    print("  first gate that excluded a pair:")
    for g, n in tally.most_common():
        print("     %-18s %d" % (g, n))
    print("  INCLUDED                        %d pairs" % len(inc))
    if inc:
        print("  cells per included pair: min %d  median %d  max %d"
              % (min(r["n_cells"] for r in inc),
                 sorted(r["n_cells"] for r in inc)[len(inc)//2],
                 max(r["n_cells"] for r in inc)))
        print("  total (pair, prompt) cells      %d" % sum(r["n_cells"] for r in inc))
        print("\n  included pairs:")
        for r in inc:
            print("    idx %-3d %-46s %2d blocks  %2d cells"
                  % (r["idx"], r["aligned"][:46], r.get("n_blocks") or -1, r["n_cells"]))
    lst = "\n".join(sorted("%s>%s" % (r["base"], r["aligned"]) for r in inc))
    print("\n  included-pair list sha256/16    %s" % sha16(lst))
    if a.write:
        out = {"_about": "H2 depth sweep population, enumerated with every gate "
                         "recorded as a column so exclusions are queries.",
               "_producer": "scripts/h2_sweep_population.py",
               "prompt_list_sha256_16": sha16("\n".join(p["prompt"] for p in prompts)),
               "included_list_sha256_16": sha16(lst),
               "min_cells": a.min_cells,
               "n_prompts": len(prompts), "n_pairs_registry": len(rows),
               "n_included": len(inc), "prompts": prompts, "pairs": rows}
        p = os.path.join(ROOT, "data", "h2_sweep_population.json")
        json.dump(out, open(p, "w"), indent=1)
        print("  wrote data/h2_sweep_population.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
