"""PILOT: rhyme_pull on the OLMo ladder — line_closure x rhyme_given_closure.

plan_rhyme.md's revised primary (RH's redesign + closure objection,
2026-08-13). Distributional, no generation: primer = a real poem's first 4
lines MINUS the last line's final word; the slot distribution comes from
malign_logits.twp (the verified local path, [5698]; never
scripts/true_word_probs.py; rule constants untouched).

Decomposition per RH's objection (a non-rhyming slot could mean "does not
know the line ends here", not "cannot rhyme"):
    line_closure         mass-weighted P(newline-ish next | primer + w)
    rhyme_given_closure  share of closure-weighted mass in the TARGET rime
                         class (scheme partner's end word, scheme read off
                         the real poem by the pinned prosodic)
    rhyme_raw            unconditional rime-class share (for comparison)
    nonpartner_*         same quantities on a non-partner line's rime class
    p_actual             mass on the poem's actual word (memorization split)

Rime keys: IPA of the syllables from the final stressed syllable onward,
onset consonants stripped from the first syllable in scope -- the SAME
normalization applied to target and candidates, so the comparison is
internally consistent even where IPA parsing is imperfect.

PILOT NUMBERS ARE NEVER QUOTED AS RESULTS.
Output: meta/M05_emergence/data/rhyme_pull_pilot.parquet
"""

import os
import re
import sys

import pandas as pd

REPO = os.path.expanduser("~/github/malign-logits")
CSV = os.path.expanduser(
    "~/github/generative-formalism1/data/data_as_in_paper/genai_rhyme_completions.csv.gz")
OUT = os.path.join(REPO, "meta/M05_emergence/data/rhyme_pull_pilot.parquet")

LADDER = [
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct-SFT",
    "allenai/Olmo-3-7B-Instruct-DPO",
    "allenai/Olmo-3-7B-Instruct",
]
N_PRIMERS = 12
N_UNRHYMED = 8
TOP_K = 40

_RIME_CACHE = {}
_VOWELS = "aeiouɑɒɔəɘɛɜɪʊʉæʌyøœɶɐɤʏɨ"


def rime_key(word):
    """Final-stressed-syllable-onward IPA, onset stripped from first syll."""
    w = word.lower().strip("'\"")
    if w in _RIME_CACHE:
        return _RIME_CACHE[w]
    key = None
    try:
        import prosodic
        pw = prosodic.Word(w)
        form = pw.children[0]
        sylls = form.children
        idx = 0
        for i in range(len(sylls) - 1, -1, -1):
            if sylls[i].is_stressed:
                idx = i
                break
        parts = []
        for j in range(idx, len(sylls)):
            t = sylls[j].txt
            if j == idx:  # strip onset consonants
                m = re.search(f"[{_VOWELS}]", t)
                t = t[m.start():] if m else t
            parts.append(t)
        key = "|".join(parts) if parts else None
    except Exception:
        key = None
    _RIME_CACHE[w] = key
    return key


def last_word(line):
    m = re.findall(r"[A-Za-z']+", line)
    return m[-1] if m else None


def _window_pairs(lines):
    import prosodic
    t = prosodic.Text("\n".join(lines))
    d = t.get_rhyming_lines(max_dist=1)
    out = set()
    for l1, (score, l2) in d.items():
        a, b = getattr(l1, "num", 0), getattr(l2, "num", 0)
        out.add((min(a, b), max(a, b)))
    return out


def pick_primers():
    """Template classes IN THE PROMPT WINDOW (RH, 2026-08-13): for a
    next-word measure the window IS the ground truth -- the model only
    knows what it is shown. ABAB (line 4 partners line 2), AABB (line 4
    partners line 3), UNRHYMED (no pair among lines 1-4). Templates must
    be exact: the defining pairs present, no cross-pairs."""
    df = pd.read_csv(CSV)
    df5 = df[df.first_n_lines == 5]
    out, counts = [], {"ABAB": 0, "AABB": 0, "unrhymed": 0}
    PER_CLASS = {"ABAB": 8, "AABB": 8, "unrhymed": 8}
    for id_human, g in df5.groupby("id_human"):
        if all(counts[k] >= PER_CLASS[k] for k in counts):
            break
        g0 = g[g.id == g.id.iloc[0]].sort_values("line_num")
        lines = g0[g0.line_num <= 4]["line_real"].tolist()
        if len(lines) < 4 or not all(isinstance(x, str) and x.strip() for x in lines):
            continue
        try:
            pairs = _window_pairs(lines)
        except Exception:
            continue
        scheme = None
        if pairs == {(1, 3), (2, 4)}:
            scheme, partner, nonpartner = "ABAB", 2, 3
        elif pairs == {(1, 2), (3, 4)}:
            scheme, partner, nonpartner = "AABB", 3, 1
        elif not pairs:
            scheme, partner, nonpartner = "unrhymed", 2, 3
        if not scheme or counts[scheme] >= PER_CLASS[scheme]:
            continue
        tw = last_word(lines[partner - 1])
        nw = last_word(lines[nonpartner - 1])
        aw = last_word(lines[3])
        tkey = rime_key(tw) if tw else None
        nkey = rime_key(nw) if nw else None
        if not tkey or not nkey or tkey == nkey:
            continue
        stub = re.sub(r"[A-Za-z']+\W*$", "", lines[3]).rstrip()
        if not stub:
            continue
        out.append({"id_human": id_human,
                    "prompt": "\n".join(lines[:3]) + "\n" + stub,
                    "scheme": scheme, "partner_line": partner,
                    "target_key": tkey, "target_word": tw,
                    "actual_word": aw, "nonpartner_key": nkey})
        counts[scheme] += 1
    print("selected:", counts, flush=True)
    return out


def newline_ids(tok, n):
    ids = set()
    toks = tok.convert_ids_to_tokens(list(range(min(n, len(tok)))))
    for i, t in enumerate(toks):
        if t and ("Ċ" in t or t.startswith("\n") or t in ("<0x0A>",)):
            ids.add(i)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(ids)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    primers = pick_primers()
    print(f"{len(primers)} primers with scheme + controls", flush=True)

    import torch
    from malign_logits.twp import (boundary_mask, expand, pick_device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = pick_device()

    rows = []
    for model_id in LADDER:
        print(f"== {model_id}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16).to(dev).eval()
        bmask = boundary_mask(tok, model.config.vocab_size)
        nl_ids = torch.tensor(newline_ids(tok, model.config.vocab_size),
                              device=dev)
        for p in primers:
            words, res = expand(model, tok, p["prompt"], dev, bmask)[:2]
            # expand keys by (surface, first_token): one word appears under
            # multiple keys when reachable via different first tokens (RH's
            # catch) -- sum mass per surface before ranking.
            wsum = {}
            for key, m in words.items():
                surf = key[0] if isinstance(key, tuple) else key
                wsum[surf] = wsum.get(surf, 0.0) + m
            cand = sorted(wsum.items(), key=lambda kv: -kv[1])[:TOP_K]
            # closure probes, one batch
            texts = [p["prompt"] + " " + w for w, _ in cand]
            encs = [tok(t, return_tensors="pt")["input_ids"][0] for t in texts]
            maxlen = max(len(e) for e in encs)
            pad = tok.pad_token_id or tok.eos_token_id or 0
            batch = torch.full((len(encs), maxlen), pad, dtype=torch.long)
            for i, e in enumerate(encs):
                batch[i, :len(e)] = e
            lens = torch.tensor([len(e) for e in encs])
            with torch.no_grad():
                lg = model(batch.to(dev)).logits
            pr_nl = []
            for i in range(len(encs)):
                probs = torch.softmax(lg[i, lens[i] - 1, :].float(), -1)
                pr_nl.append(float(probs[nl_ids].sum()))
            row = {"model": model_id, **{k: p[k] for k in
                   ("id_human", "scheme", "partner_line", "target_word",
                    "actual_word")}}
            tot = sum(pb for _, pb in cand) or 1e-12
            cw = sum(pb * c for (_, pb), c in zip(cand, pr_nl))
            def share(keyname):
                key = p[keyname]
                num = sum(pb for (w, pb) in cand if rime_key(w) == key)
                numc = sum(pb * c for (w, pb), c in zip(cand, pr_nl)
                           if rime_key(w) == key)
                return num / tot, (numc / cw if cw > 0 else None)
            row["n_candidates"] = len(cand)
            row["slot_mass_topk"] = tot
            row["line_closure"] = cw / tot
            row["rhyme_raw"], row["rhyme_given_closure"] = share("target_key")
            row["nonpartner_raw"], row["nonpartner_given_closure"] = share("nonpartner_key")
            aw = (p["actual_word"] or "").lower()
            row["p_actual"] = (sum(v for k, v in wsum.items()
                                   if k.lower() == aw) if aw else None)
            rows.append(row)
            print(f"   {p['id_human']} closure={row['line_closure']:.3f} "
                  f"rhyme|cl={row['rhyme_given_closure']}", flush=True)
        del model
        if dev == "mps":
            torch.mps.empty_cache()
        pd.DataFrame(rows).to_parquet(OUT)
        print(f"   checkpointed {len(rows)}", flush=True)
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"wrote {OUT}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
