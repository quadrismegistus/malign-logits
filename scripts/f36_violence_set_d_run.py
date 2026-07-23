"""Set D: desire × commitment frame-minimal pairs — token survival + base levels.

Reports per-cell: base p, base rank, base entropy at the slot, aligned p,
aligned rank, ratio, displaced. The regression and beam resistance run separately.

Usage:
    uv run python scripts/f36_violence_set_d_run.py
"""

import gc
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
from scripts.f36_violence_set_d import SET_D

FAMILIES = ['olmo', 'llama', 'amber', 'olmo-tiny']


def main():
    rows = []

    for fkey in FAMILIES:
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        # Cache logits
        need = any(not psyche.primary_process._cache.has_logits(fam.base, d['prompt'])
                   for d in SET_D)
        if need:
            print(f"  {fkey}: caching logits for {len(SET_D)} prompts...")
            base_model, base_tok = load_model(fam.base)
            al_id = (fam.superego or fam.ego)
            aligned_model, _ = load_model(al_id)
            for d in SET_D:
                for mid, model in [(fam.base, base_model), (al_id, aligned_model)]:
                    if not psyche.primary_process._cache.has_logits(mid, d['prompt']):
                        logits = get_base_logits(model, base_tok, d['prompt'])
                        psyche.primary_process._cache.set_logits(
                            mid, d['prompt'], logits.cpu().numpy())
            del base_model, aligned_model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"    cached")
        else:
            print(f"  {fkey}: logits cached")

        for d in SET_D:
            try:
                bl = psyche.primary_process.logits(d['prompt']).numpy()
                al = aligned.logits(d['prompt']).numpy()
            except Exception:
                continue

            n = min(len(bl), len(al))
            bp = softmax(bl[:n].astype(np.float64))
            ap = softmax(al[:n].astype(np.float64))

            f_id = int(np.argmax(bp))
            a_id = int(np.argmax(ap))

            ranking = np.argsort(-ap)
            f_rank_aligned = int(np.where(ranking == f_id)[0][0])

            # Base entropy at the slot
            base_ent = float(sp_entropy(bp))

            # Target verb token id and its probabilities
            verb = d['verb']
            verb_ids = tok.encode(" " + verb, add_special_tokens=False)
            verb_tid = verb_ids[0] if verb_ids else None

            verb_p_base = float(bp[verb_tid]) if verb_tid and verb_tid < n else np.nan
            verb_p_aligned = float(ap[verb_tid]) if verb_tid and verb_tid < n else np.nan
            verb_rank_base = int(np.where(np.argsort(-bp) == verb_tid)[0][0]) if verb_tid and verb_tid < n else np.nan
            verb_rank_aligned = int(np.where(ranking == verb_tid)[0][0]) if verb_tid and verb_tid < n else np.nan
            verb_ratio = verb_p_aligned / verb_p_base if verb_p_base > 1e-10 else np.nan

            rows.append({
                'family': fkey,
                'id': d['id'],
                'prompt': d['prompt'],
                'verb': verb,
                'frame': d['frame'],
                'commitment': d['commitment'],
                'person': d['person'],
                'tense': d['tense'],
                'category': d['category'],
                'severity': d['severity'],
                'names_death': d.get('names_death', False),
                'base_entropy': base_ent,
                'f_token': tok.decode([f_id]).strip(),
                'a_token': tok.decode([a_id]).strip(),
                'f_prob_base': float(bp[f_id]),
                'f_prob_aligned': float(ap[f_id]),
                'f_rank_aligned': f_rank_aligned,
                'displaced': f_id != a_id,
                'ratio': float(ap[f_id] / bp[f_id]) if bp[f_id] > 1e-10 else 0,
                'verb_token': tok.decode([verb_tid]).strip() if verb_tid else '',
                'verb_p_base': verb_p_base,
                'verb_p_aligned': verb_p_aligned,
                'verb_rank_base': verb_rank_base,
                'verb_rank_aligned': verb_rank_aligned,
                'verb_ratio': verb_ratio,
            })

    df = pd.DataFrame(rows)

    # ── Print results ──
    viol = df[df.category == 'violence']

    print(f"\n{'='*90}")
    print(f"SET D: CORE 2×2 (violence, 3rd-past) — VERB TOKEN survival")
    print(f"{'='*90}")

    core = viol[(viol.person == '3rd') & (viol.tense == 'past') &
                 (viol.frame.isin(['desire', 'act']))]

    print(f"\n  {'verb':10s} {'frame':8s} {'commit':12s} {'fam':8s} "
          f"{'v_p_base':>8s} {'v_p_al':>7s} {'v_rank_b':>8s} {'v_rank_a':>8s} "
          f"{'ratio':>6s} {'entropy':>7s}  argmax")

    for verb in sorted(core.verb.unique()):
        for frame in ['desire', 'act']:
            for commit in ['uncommitted', 'committed']:
                sub = core[(core.verb == verb) & (core.frame == frame) &
                           (core.commitment == commit)]
                for _, r in sub.iterrows():
                    vrb = f"{r.verb_p_base:.3f}" if not np.isnan(r.verb_p_base) else "  n/a"
                    vra = f"{r.verb_p_aligned:.3f}" if not np.isnan(r.verb_p_aligned) else " n/a"
                    vrankb = f"{r.verb_rank_base:.0f}" if not np.isnan(r.verb_rank_base) else "n/a"
                    vranka = f"{r.verb_rank_aligned:.0f}" if not np.isnan(r.verb_rank_aligned) else "n/a"
                    vrat = f"{r.verb_ratio:.3f}" if not np.isnan(r.verb_ratio) else " n/a"
                    print(f"  {r.verb:10s} {r.frame:8s} {r.commitment:12s} {r.family:8s} "
                          f"{vrb:>8s} {vra:>7s} {vrankb:>8s} {vranka:>8s} "
                          f"{vrat:>6s} {r.base_entropy:7.2f}  {r.f_token}→{r.a_token}")

    # Aggregate: mean verb_ratio by frame × commitment
    print(f"\n  AGGREGATE (mean verb_ratio across verbs and families):")
    print(f"  {'frame':8s} {'commitment':12s} {'verb_ratio':>10s} {'verb_rank_al':>12s} {'entropy':>8s} {'n':>4s}")
    for frame in ['desire', 'act']:
        for commit in ['uncommitted', 'committed']:
            sub = core[(core.frame == frame) & (core.commitment == commit)]
            vr = sub.verb_ratio.dropna()
            vrk = sub.verb_rank_aligned.dropna()
            ent = sub.base_entropy
            print(f"  {frame:8s} {commit:12s} {vr.mean():10.3f} {vrk.mean():12.1f} "
                  f"{ent.mean():8.2f} {len(sub):4d}")

    # Person × tense (desire-uncommitted)
    print(f"\n{'='*90}")
    print(f"PERSON × TENSE (desire-uncommitted, violence)")
    print(f"{'='*90}")
    du = viol[(viol.frame == 'desire') & (viol.commitment == 'uncommitted')]
    print(f"\n  {'person':14s} {'tense':8s} {'verb_ratio':>10s} {'verb_rank_al':>12s} {'entropy':>8s} {'n':>4s}")
    for person in ['3rd', '1st']:
        for tense in ['past', 'present']:
            sub = du[(du.person == person) & (du.tense == tense)]
            vr = sub.verb_ratio.dropna()
            vrk = sub.verb_rank_aligned.dropna()
            print(f"  {person:14s} {tense:8s} {vr.mean():10.3f} {vrk.mean():12.1f} "
                  f"{sub.base_entropy.mean():8.2f} {len(sub):4d}")

    # Death-naming
    print(f"\n  DEATH-NAMING (desire-uncommitted, 3rd-past):")
    du3p = du[(du.person == '3rd') & (du.tense == 'past')]
    for nd in [True, False]:
        sub = du3p[du3p.names_death == nd]
        vr = sub.verb_ratio.dropna()
        label = "death-naming" if nd else "act-depicting"
        print(f"    {label:15s}  verb_ratio={vr.mean():.3f}  n={len(sub)}")

    out = os.path.join(PATH_DATA, "f36_violence_set_d.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
