"""Set E runner: score P(verb) at realized action slots.

Scores BOTH infinitive and past-tense token for each verb at each slot.
Full audit trail: exact token string + id at every slot.

Usage:
    uv run python scripts/f36_violence_set_e_run.py
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
from scripts.f36_violence_set_e import SET_E, VIOLENT_VERBS, DEATH_NAMING, PAST_FORMS

FAMILIES = ['olmo', 'llama', 'amber', 'olmo-tiny']


def main():
    rows = []

    for fkey in FAMILIES:
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego
        al_id = fam.superego or fam.ego

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        # Cache logits
        all_prompts = list(set(d['prompt'] for d in SET_E))
        need = any(not psyche.primary_process._cache.has_logits(fam.base, p)
                   for p in all_prompts)
        if need:
            print(f"  {fkey}: caching logits...")
            base_model, _ = load_model(fam.base)
            aligned_model, _ = load_model(al_id)
            for p in all_prompts:
                for mid, model in [(fam.base, base_model), (al_id, aligned_model)]:
                    if not psyche.primary_process._cache.has_logits(mid, p):
                        logits = get_base_logits(model, tok, p)
                        psyche.primary_process._cache.set_logits(mid, p, logits.cpu().numpy())
            del base_model, aligned_model; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            print(f"    cached")
        else:
            print(f"  {fkey}: cached")

        for entry in SET_E:
            prompt = entry['prompt']
            verb = entry['verb']

            try:
                bl = psyche.primary_process.logits(prompt).numpy()
                al = aligned.logits(prompt).numpy()
            except Exception:
                continue

            n = min(len(bl), len(al))
            bp = softmax(bl[:n].astype(np.float64))
            ap = softmax(al[:n].astype(np.float64))
            base_ent = float(sp_entropy(bp))

            # Score BOTH infinitive and past-tense forms
            forms = [(" " + verb, "infinitive")]
            past = PAST_FORMS.get(verb)
            if past and past != verb:
                forms.append((" " + past, "past_tense"))

            for form_str, form_label in forms:
                verb_ids = tok.encode(form_str, add_special_tokens=False)
                if not verb_ids:
                    continue
                vid = verb_ids[0]
                if vid >= n:
                    continue

                # AUDIT TRAIL
                scored_token = tok.decode([vid])
                bp_verb = float(bp[vid])
                ap_verb = float(ap[vid])

                if bp_verb > 1e-15 and ap_verb > 1e-15:
                    logdiff = math.log2(ap_verb) - math.log2(bp_verb)
                elif bp_verb > 1e-15:
                    logdiff = -20.0
                elif ap_verb > 1e-15:
                    logdiff = 20.0
                else:
                    logdiff = 0.0

                base_rank = int(np.where(np.argsort(-bp) == vid)[0][0])
                al_rank = int(np.where(np.argsort(-ap) == vid)[0][0])

                rows.append({
                    'family': fkey,
                    'prompt': prompt,
                    'verb': verb,
                    'form': form_label,
                    'scored_token_str': scored_token,
                    'scored_token_id': int(vid),
                    'realization': entry['realization'],
                    'variant': entry['variant'],
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

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"SET E: REALIZED vs UNREALIZED (violent verbs, best form per slot)")
    print(f"{'='*70}")

    # Pick the best-scoring form per (family, prompt, verb)
    viol = df[df.is_violent].copy()
    best = viol.loc[viol.groupby(['family', 'prompt', 'verb'])['base_p'].idxmax()]

    print(f"\n  {'realization':25s} {'logdiff':>8s} {'base_p':>7s} {'al_p':>7s} {'entropy':>7s} {'n':>4s}")
    for real in ['realized_constrained', 'realized_boundary', 'realized_open', 'benign_realized']:
        sub = df[df.realization == real]
        if sub.empty:
            continue
        # Use best form
        bsub = best[best.realization == real] if real != 'benign_realized' else sub
        print(f"  {real:25s} {bsub.logdiff.mean():+8.3f} {bsub.base_p.mean():7.4f} "
              f"{bsub.aligned_p.mean():7.4f} {bsub.base_entropy.mean():7.2f} {len(bsub):4d}")

    # Compare against Set D desire-uncommitted (from the v3 data)
    setd = pd.read_csv(os.path.join(PATH_DATA, 'f36_violence_set_d_v3.csv'))
    du = setd[(setd.is_violent) & (setd.frame == 'desire') & (setd.commitment == 'uncommitted') &
               (setd.person == '3rd') & (setd.tense == 'past')]
    print(f"\n  Set D desire-uncommit:   logdiff={du.logdiff.mean():+8.3f} (for comparison)")

    # Audit trail: show scored tokens for a sample
    print(f"\n{'='*70}")
    print(f"AUDIT TRAIL (sample)")
    print(f"{'='*70}")
    for _, r in best[best.family == 'olmo'].head(10).iterrows():
        print(f"  {r.realization:25s} {r.verb:10s} form={r.form:12s} "
              f"token=\"{r.scored_token_str}\" id={r.scored_token_id}  "
              f"base_p={r.base_p:.4f} rank={r.base_rank}  "
              f"al_p={r.aligned_p:.4f} rank={r.aligned_rank}  "
              f"logdiff={r.logdiff:+.2f}")
        print(f"    prompt: \"{r.prompt}\"")

    # Per-family
    print(f"\n  PER-FAMILY (realized_constrained, violent, best form):")
    rc = best[best.realization == 'realized_constrained']
    for fam in sorted(rc.family.unique()):
        sub = rc[rc.family == fam]
        print(f"    {fam:12s}  logdiff={sub.logdiff.mean():+.3f}  "
              f"base_p={sub.base_p.mean():.4f}  n={len(sub)}")

    out = os.path.join(PATH_DATA, "f36_violence_set_e.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
