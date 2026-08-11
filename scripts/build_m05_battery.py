#!/usr/bin/env python
"""Enumerate the M05 SAMPLED battery -> data/m05_battery.json.

    cd ~/github/malign-logits && uv run python scripts/build_m05_battery.py

RH's word (2026-08-11): sample, don't sweep -- "target a few hundred prompts
and not a few 1000." Blocks, in precedence order (first block wins a text;
overlaps reported):

    PAIRS_105     the standing 105-pair sample of the 684 minimal
                  transgressive/neutral pairs, plus the two anger prompts
                  (data/beam_sample_105_plus_anger.csv; the sample a lot of
                  experiments already use -- seeded, manifested)
    INSTITUTIONAL the 24 paired F21 core (source=INSTITUTIONAL, en only --
                  the finding=F21 predicate leaked 13 zh rows and 14 SETE;
                  caught at RH's "which institutional?", 2026-08-11)
    M03_SLICE     36 = 18 scenarios x 2 conditions (indiv_I_final vs
                  inst_I_final): the individual-vs-institutional deference
                  contrast at matched clause position -- F24's deference
                  stage as a designed instrument; the rest of the factorial
                  stays off this battery
    DEFAULT_CORE  the full legacy DEFAULT battery incl. liminal/explicit
                  (the F01 originals)
    QUINT_EN      the English quintuplets, ACTIVE only (frame-exit
                  acquisition + the pole_sep arrow + the U discriminator)
    CAPACITY_*    the five M05 capacity families from pair_drafts/m05_*.yaml,
                  M05-ONLY blocks (not catalogue rows; cleared for this run
                  by RH's word; catalogue ingestion is a separate later step)

Hash on TEXTS ([5346]). Both units printed. zh, LITERARY, and
M03_SPEAKER_KERNEL are out by design ([registrar scan, RH-approved]); the
rest of the catalogue simply is not drawn -- a battery is an order, not
the menu.
"""
import csv
import hashlib
import json
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
OUT = "data/m05_battery.json"


def sha16(texts):
    return hashlib.sha256("\n".join(sorted(texts)).encode()).hexdigest()[:16]


def main():
    cat = json.load(open("data/prompt_categorisation.json"))["prompts"]
    act = [p for p in cat if p["status"] == "ACTIVE"]

    blocks = {}
    blocks["PAIRS_105"] = sorted({
        r["prompt"] for r in csv.DictReader(
            open("data/beam_sample_105_plus_anger.csv"))})
    blocks["INSTITUTIONAL"] = sorted({
        p["prompt"] for p in act
        if p.get("source") == "INSTITUTIONAL" and p.get("language") == "en"})
    blocks["M03_SLICE"] = sorted({
        p["prompt"] for p in act
        if p.get("source") == "M03_SPEAKER_KERNEL"
        and p.get("group_role") in ("indiv_I_final", "inst_I_final")})
    blocks["DEFAULT_CORE"] = sorted({
        p["prompt"] for p in act if p.get("source") == "DEFAULT"})
    blocks["QUINT_EN"] = sorted({
        p["prompt"] for p in act
        if p.get("source") == "QUINTUPLETS" and p.get("language") == "en"})

    fam_files = {
        "CAPACITY_REFERENCE": "m05_reference.yaml",
        "CAPACITY_REASONING": "m05_reasoning.yaml",
        "CAPACITY_DISCOURSE": "m05_discourse_reference.yaml",
        "CAPACITY_POETIC": "m05_poetic_texture.yaml",
        "CAPACITY_PACKAGES": "m05_semantic_packages.yaml",
    }
    fam_sha = {}
    for name, f in fam_files.items():
        path = f"pair_drafts/{f}"
        fam_sha[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        texts = set()
        for r in yaml.safe_load(open(path)):
            for k in ("prompt", "FORMULAIC", "PARAPHRASE"):
                if k in r:
                    texts.add(r[k])
        blocks[name] = sorted(texts)

    # precedence dedup: first block wins; overlaps reported, never silent
    seen, final, overlaps = set(), {}, []
    for name, texts in blocks.items():
        kept = [t for t in texts if t not in seen]
        for t in texts:
            if t in seen:
                overlaps.append((name, t))
        seen.update(kept)
        final[name] = kept

    all_texts = sorted(seen)
    out = {
        "_about": ("M05 sampled battery, RH's word 2026-08-11: a few hundred "
                   "prompts, not a few thousand. Blocks in precedence order; "
                   "first block wins a text; overlaps listed. Hash on TEXTS."),
        "_producer": "scripts/build_m05_battery.py",
        "_generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_excluded_by_design": ["language=zh", "source=LITERARY",
                                "M03_SPEAKER_KERNEL beyond the 2-condition slice",
                                "everything not drawn -- a battery is an "
                                "order, not the menu"],
        "_capacity_family_file_sha16": fam_sha,
        "n_texts": len(all_texts),
        "sha256_16_over_texts": sha16(all_texts),
        "blocks": {k: {"n": len(v), "sha256_16": sha16(v), "texts": v}
                   for k, v in final.items()},
        "overlaps_won_by_earlier_block": [
            {"block": b, "text": t} for b, t in overlaps],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}: {len(all_texts)} distinct texts, "
          f"sha {out['sha256_16_over_texts']}")
    for k, v in final.items():
        print(f"  {k:22} {len(v):4} texts  {sha16(v)}")
    if overlaps:
        print(f"  overlaps (won by earlier block): {len(overlaps)}")
        for b, t in overlaps:
            print(f"    {b:22} {t[:60]}")
    print(f"\ncells at 87 checkpoints: {len(all_texts) * 87:,}")


if __name__ == "__main__":
    sys.exit(main() or 0)
