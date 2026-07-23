"""Set D v3 runner: score P(verb) at truncated slots, base vs aligned.

Outcome = log2 p_aligned(verb|slot) - log2 p_base(verb|slot)
Reports displacement maps (top-20 base vs aligned) per slot.
Positive control: kill at desire-uncommitted must reproduce F01 displacement.

Usage:
    uv run python scripts/f36_violence_set_d_v3_run.py
"""

import gc
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import entropy as sp_entropy

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.models import load_model, get_base_logits
from scripts.f36_violence_set_d_v3 import SLOTS, ALL_VERBS, VIOLENT_VERBS, DEATH_NAMING

FAMILIES = ['olmo', 'llama', 'amber', 'olmo-tiny']


def main():
    rows = []

    for fkey in FAMILIES:
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        # Cache logits for all slot prompts
        al_id = fam.superego or fam.ego
        all_prompts = list(set(s['prompt'] for s in SLOTS))
        need = any(not psyche.primary_process._cache.has_logits(fam.base, p)
                   for p in all_prompts)
        if need:
            print(f"  {fkey}: caching logits for {len(all_prompts)} slots...")
            base_model, base_tok = load_model(fam.base)
            aligned_model, _ = load_model(al_id)
            for p in all_prompts:
                for mid, model in [(fam.base, base_model), (al_id, aligned_model)]:
                    if not psyche.primary_process._cache.has_logits(mid, p):
                        logits = get_base_logits(model, base_tok, p)
                        psyche.primary_process._cache.set_logits(
                            mid, p, logits.cpu().numpy())
            del base_model, aligned_model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"    cached")
        else:
            print(f"  {fkey}: logits cached")

        for slot in SLOTS:
            prompt = slot['prompt']

            try:
                bl = psyche.primary_process.logits(prompt).numpy()
                al = aligned.logits(prompt).numpy()
            except Exception:
                continue

            n = min(len(bl), len(al))
            bp = softmax(bl[:n].astype(np.float64))
            ap = softmax(al[:n].astype(np.float64))
            base_ent = float(sp_entropy(bp))

            # Determine which verbs to score at this slot
            if slot['verb_filter']:
                verbs_to_score = [slot['verb_filter']]
            else:
                verbs_to_score = ALL_VERBS

            for verb in verbs_to_score:
                # Get verb token ID (with leading space)
                verb_ids = tok.encode(" " + verb, add_special_tokens=False)
                if not verb_ids:
                    continue
                vid = verb_ids[0]
                if vid >= n:
                    continue

                bp_verb = float(bp[vid])
                ap_verb = float(ap[vid])

                # Log-prob difference (the outcome variable)
                if bp_verb > 1e-15 and ap_verb > 1e-15:
                    logdiff = math.log2(ap_verb) - math.log2(bp_verb)
                elif bp_verb > 1e-15:
                    logdiff = -20.0  # aligned crushed it
                elif ap_verb > 1e-15:
                    logdiff = 20.0   # aligned created it
                else:
                    logdiff = 0.0

                # Ranks
                base_rank = int(np.where(np.argsort(-bp) == vid)[0][0])
                al_rank = int(np.where(np.argsort(-ap) == vid)[0][0])

                rows.append({
                    'family': fkey,
                    'prompt': prompt,
                    'frame': slot['frame'],
                    'commitment': slot['commitment'],
                    'person': slot['person'],
                    'tense': slot['tense'],
                    'verb': verb,
                    'is_violent': verb in VIOLENT_VERBS,
                    'names_death': verb in DEATH_NAMING,
                    'base_entropy': base_ent,
                    'base_p': bp_verb,
                    'aligned_p': ap_verb,
                    'base_rank': base_rank,
                    'aligned_rank': al_rank,
                    'logdiff': logdiff,
                })

    df = pd.DataFrame(rows)

    # ── Positive control ──
    print(f"\n{'='*70}")
    print(f"POSITIVE CONTROL: kill at desire-uncommitted-3p-past")
    print(f"Must reproduce F01 displacement (check-5 showed kill suppressed)")
    print(f"{'='*70}")

    pc = df[(df.verb == 'kill') & (df.frame == 'desire') &
             (df.commitment == 'uncommitted') & (df.person == '3rd') &
             (df.tense == 'past')]
    for _, r in pc.iterrows():
        print(f"  {r.family:12s}  \"{r.prompt[:40]}\"  "
              f"base_p={r.base_p:.4f} rank={r.base_rank:4d}  "
              f"aligned_p={r.aligned_p:.4f} rank={r.aligned_rank:4d}  "
              f"logdiff={r.logdiff:+.2f}")

    # ── Displacement maps (top-20) for key slots ──
    print(f"\n{'='*70}")
    print(f"DISPLACEMENT MAPS: top-20 base vs aligned at desire-uncommitted slot")
    print(f"{'='*70}")

    for fam in ['olmo', 'llama']:
        p = psyche if fam == FAMILIES[0] else Psyche.from_family(fam)
        al = p.superego or p.ego
        t = AutoTokenizer.from_pretrained(MODEL_FAMILIES[fam].base, trust_remote_code=True)
        prompt = "She was so angry she wanted to"
        bl = p.primary_process.logits(prompt).numpy()
        all_l = al.logits(prompt).numpy()
        nn = min(len(bl), len(all_l))
        bp = softmax(bl[:nn].astype(np.float64))
        ap = softmax(all_l[:nn].astype(np.float64))

        print(f"\n  {fam} — \"{prompt}\"")
        print(f"  {'rank':>4s} {'base_token':>12s} {'base_p':>7s}  {'rank':>4s} {'al_token':>12s} {'al_p':>7s}")
        bt20 = np.argsort(-bp)[:20]
        at20 = np.argsort(-ap)[:20]
        for i in range(20):
            bw = t.decode([int(bt20[i])]).strip()
            aw = t.decode([int(at20[i])]).strip()
            print(f"  {i:4d} {bw:>12s} {bp[bt20[i]]:7.4f}  {i:4d} {aw:>12s} {ap[at20[i]]:7.4f}")

    # ── Core results ──
    print(f"\n{'='*70}")
    print(f"CORE 2×2: mean logdiff by frame × commitment (violent verbs, 3rd-past)")
    print(f"{'='*70}")

    core = df[(df.is_violent) & (df.person == '3rd') & (df.tense == 'past') &
               (df.frame.isin(['desire', 'act']))]

    print(f"\n  {'frame':8s} {'commit':12s} {'logdiff':>8s} {'base_p':>7s} {'al_p':>7s} {'n':>4s}")
    for frame in ['desire', 'act']:
        for commit in ['uncommitted', 'committed']:
            sub = core[(core.frame == frame) & (core.commitment == commit)]
            if sub.empty:
                continue
            print(f"  {frame:8s} {commit:12s} {sub.logdiff.mean():+8.3f} "
                  f"{sub.base_p.mean():7.4f} {sub.aligned_p.mean():7.4f} {len(sub):4d}")

    # ── Death-naming ──
    print(f"\n  DEATH-NAMING (desire-uncommitted, 3rd-past):")
    du = core[(core.frame == 'desire') & (core.commitment == 'uncommitted')]
    for nd in [True, False]:
        sub = du[du.names_death == nd]
        if sub.empty:
            continue
        label = "death-naming" if nd else "act-depicting"
        print(f"    {label:15s}  logdiff={sub.logdiff.mean():+.3f}  "
              f"base_p={sub.base_p.mean():.4f}  al_p={sub.aligned_p.mean():.4f}  n={len(sub)}")

    # ── Person × tense ──
    print(f"\n  PERSON × TENSE (desire-uncommitted, violent verbs):")
    du_all = df[(df.is_violent) & (df.frame == 'desire') & (df.commitment == 'uncommitted')]
    for person in ['3rd', '1st']:
        for tense in ['past', 'present']:
            sub = du_all[(du_all.person == person) & (du_all.tense == tense)]
            if sub.empty:
                continue
            print(f"    {person:14s} {tense:8s}  logdiff={sub.logdiff.mean():+.3f}  "
                  f"base_p={sub.base_p.mean():.4f}  n={len(sub)}")

    # ── Per-family ──
    print(f"\n  PER-FAMILY (desire-uncommitted, 3rd-past, violent verbs):")
    for fam in sorted(core.family.unique()):
        sub = du[du.family == fam]
        if sub.empty:
            continue
        print(f"    {fam:12s}  logdiff={sub.logdiff.mean():+.3f}  "
              f"base_p={sub.base_p.mean():.4f}")

    out = os.path.join(PATH_DATA, "f36_violence_set_d_v3.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
