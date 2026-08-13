"""Verse-fleet producer: candidate-set scoring at declared slots. SMOKE-FIRST.

plan_verse_fleet.md's instrument, built to iron wrinkles locally before any
box rents. Per poem: ONE cached prefix pass per slot (cross-slot KV reuse is
a later optimisation, noted below), then the slot's candidate class branched
as a single batched forward off the cached prefix. Per candidate we read:
    p_word      product of its token probs (teacher-forced path)
    p_close     newline-family mass at the branch's final position
Per slot we also read, FREE off the prefix pass's last softmax:
    norm composition over SINGLE-TOKEN K-rated words (declared limit;
    the battery-calibration slots price the single-token bias)

Slot layout per poem (1 called + 8 uncalled, the within-poem time course):
    called       line 4 minus final word
    near         line 4 minus final TWO words (locality control)
    end1..end3   lines 1..3, each context = poem prefix up to that line
                 minus ITS final word (end_partner = the class PRIOR slot:
                 the class's own defining word not yet seen)
    mid1..mid4   poem prefix up to the midpoint (by word count) of each line

Rime classes: built over a DECLARED vocabulary (data/rime_class_vocab.json,
frozen; smoke uses the k_ratings word list) via the paper-pinned prosodic
rime_key. The class at every slot of a poem is the poem's TARGET class (the
scheme partner's key); nonpartner class read beside it at the called slot.

Usage:
    uv run python verse_fleet_producer.py smoke   # 3 poems x 1 model, chatty
    uv run python verse_fleet_producer.py manifest  # write slot manifest only
Output: meta/M05_emergence/data/verse_fleet_smoke.parquet (+ manifest json)
"""

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhyme_pull_pilot import last_word  # pinned-prosodic helper

_RIME_CACHE_V2 = {}


def rime_key(word):
    """v2, PHONEMIC (the audit's fix: v1 fell back to syllable SPELLING and
    shattered /eI/ into 'ay'/'ey'/'eigh'). Key = rime phonemes of the final
    stressed syllable FROM ITS FIRST VOWEL (strips onset-glide leaks like
    weigh->weI) + full IPA of any following syllables, stress marks removed."""
    w = word.lower().strip("'\"")
    if w in _RIME_CACHE_V2:
        return _RIME_CACHE_V2[w]
    key = None
    try:
        import prosodic
        pw = prosodic.Word(w)
        sylls = pw.children[0].children
        idx = 0
        for i in range(len(sylls) - 1, -1, -1):
            if sylls[i].is_stressed:
                idx = i
                break
        parts = []
        for j in range(idx, len(sylls)):
            if j == idx:
                phs = list(sylls[j].rime.children) if sylls[j].rime is not None else []
                vi = next((k for k, ph in enumerate(phs)
                           if getattr(ph, "is_vowel", False)), 0)
                seg = "".join(ph.txt for ph in phs[vi:])
                # glide leak (way->weI) and espeak/lexicon vowel-symbol
                # divergence (blisses 'VZ' vs kisses '@z') split true classes
                seg = seg.lstrip("wj")
                parts.append(seg)
            else:
                ipa = str(getattr(sylls[j], "ipa", None) or sylls[j].txt)
                parts.append(ipa.replace("\u02c8", "").replace("\u02cc", ""))
        key = "|".join(x for x in parts if x) or None
        if key:
            key = key.replace("\u1d7b", "\u0259").replace("\u0268", "\u026a")
    except Exception:
        key = None
    _RIME_CACHE_V2[w] = key
    return key

REPO = os.path.expanduser("~/github/malign-logits")
ROSTER = os.path.join(REPO, "data/rhyme_fleet_roster.json")
CSV = os.path.expanduser(
    "~/github/generative-formalism1/data/data_as_in_paper/genai_rhyme_completions.csv.gz")
RIME_VOCAB = os.path.join(REPO, "data/rime_class_vocab_v2.json")
OUT_DIR = os.path.join(REPO, "meta/M05_emergence/data")

SMOKE_MODEL = os.environ.get("VERSE_SMOKE_MODEL", "HuggingFaceTB/SmolLM2-360M")


# ── rime-class vocabulary ────────────────────────────────────────

def build_rime_vocab(limit=None):
    """One-time: rime key for every word in the declared vocabulary
    (k_ratings en list). Cached to RIME_VOCAB with the rule recorded."""
    if os.path.exists(RIME_VOCAB):
        return json.load(open(RIME_VOCAB))["key_to_words"]
    from malign_logits.fields import _k
    ratings, _ = _k("en")
    words = sorted(ratings.keys())
    if limit:
        words = words[:limit]
    k2w = {}
    for i, w in enumerate(words):
        if not re.fullmatch(r"[a-z']+", w):
            continue
        k = rime_key(w)
        if k:
            k2w.setdefault(k, []).append(w)
        if i and i % 2000 == 0:
            print(f"  rime vocab {i}/{len(words)}", flush=True)
    json.dump({"_meta": {"vocabulary": "k_ratings en word list",
                         "rule": "v2 PHONEMIC: rime phonemes of final stressed syllable from first vowel (audit fix: v1 was orthographic and shattered /eI/), plus following sylls full IPA, stress marks stripped",
                         "n_words_in": len(words), "n_keys": len(k2w)},
               "key_to_words": k2w}, open(RIME_VOCAB, "w"))
    print(f"rime vocab: {len(k2w)} keys", flush=True)
    return k2w


# ── slot manifest ────────────────────────────────────────────────

def strip_last_word(line):
    # symmetric with last_word: the stripped token must CONTAIN a letter,
    # so a line ending in a bare quote mark loses its true final word too
    return re.sub(r"[A-Za-z']*[A-Za-z][A-Za-z']*[^A-Za-z]*$", "", line).rstrip()


# per-slot called/uncalled status for BOTH classes (the smoke's lesson:
# the "nonpartner" class is ITSELF called at its own scheme positions --
# line-1/3 ends in ABAB carry the A-rime -- so every class is pull-measure
# where called and control where not; the flag makes that explicit).
NONPARTNER_CALLED_AT = {  # slots where the nonpartner class is called
    "ABAB": {"end1", "end3"},      # A-rime at lines 1,3
    "AABB": {"end1", "end2"},      # A-couplet at lines 1,2 (end2=partner-prior of A)
    "unrhymed": set(),
}


def poem_slots(lines, partner_line):
    """The nine declared slots. Context is always a PREFIX of the poem text."""
    slots = []
    # line-end slots for lines 1..3
    for i in (1, 2, 3):
        ctx = "\n".join(lines[:i - 1] + [strip_last_word(lines[i - 1])]) if i > 1 \
            else strip_last_word(lines[0])
        kind = "end_partner_prior" if i == partner_line else f"end{i}"
        slots.append({"slot": kind, "phase":
                      ("prior" if i == partner_line else
                       ("pre" if i < partner_line else "post")),
                      "context": ctx})
    # mid-line slots
    for i in (1, 2, 3, 4):
        words = lines[i - 1].split()
        mid = max(1, len(words) // 2)
        ctx = "\n".join(lines[:i - 1] + [" ".join(words[:mid])])
        slots.append({"slot": f"mid{i}", "phase":
                      ("pre" if i <= partner_line else "post"),
                      "context": ctx})
    # near-called and called
    l4 = lines[3]
    stub1 = strip_last_word(l4)
    stub2 = strip_last_word(stub1)
    slots.append({"slot": "near", "phase": "approach",
                  "context": "\n".join(lines[:3] + [stub2])})
    slots.append({"slot": "called", "phase": "called",
                  "context": "\n".join(lines[:3] + [stub1])})
    return slots


def load_poems(n=3):
    roster = json.load(open(ROSTER))["poems"]
    df = pd.read_csv(CSV)
    df5 = df[df.first_n_lines == 5]
    out = []
    for r in roster:
        if len(out) >= n:
            break
        g = df5[df5.id_human == r["id_human"]]
        if not len(g):
            continue
        g0 = g[g.id == g.id.iloc[0]].sort_values("line_num")
        lines = g0[g0.line_num <= 4]["line_real"].tolist()
        if len(lines) < 4:
            continue
        partner = {"ABAB": 2, "AABB": 3, "unrhymed": 2}[r["scheme"]]
        tw = last_word(lines[partner - 1])
        nw = last_word(lines[{"ABAB": 3, "AABB": 1, "unrhymed": 3}[r["scheme"]] - 1])
        aw = last_word(lines[3])
        tk, ak = rime_key(tw), rime_key(aw)
        # [5739]/[5740] flag-not-filter: does the key put the poem's OWN
        # rhyme pair in one class? Measured 91.7% pre-1900 / 85.0% 1900+
        # (data/verse_fleet_key_resolution_check.csv); gates are columns.
        out.append({**r, "lines": lines, "partner_line": partner,
                    "target_word": tw, "target_key": tk,
                    "nonpartner_word": nw, "nonpartner_key": rime_key(nw),
                    "actual_word": aw, "actual_key": ak,
                    "key_resolves_own_rhyme":
                        bool(tk and ak and tk == ak
                             and r["scheme"] != "unrhymed")})
    return out


# ── scoring ──────────────────────────────────────────────────────

def newline_ids(tok, n):
    ids = set()
    for i, t in enumerate(tok.convert_ids_to_tokens(list(range(min(n, len(tok)))))):
        if t and ("Ċ" in t or t.startswith("\n") or t == "<0x0A>"):
            ids.add(i)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(ids)


def expand_slot(model, tok, dev, context, bmask):
    """Design B backbone ([5735]/[5736]/[5737]): the frozen twp expand() at the
    slot, full resolved word distribution at theta=0.001. Returns per-SURFACE
    mass (expand keys are (surface, first_token) tuples — RH's catch: sum per
    surface) plus the residual dict, which IS the declared censoring
    measurement (lacan [5736] §4: 'nearly harmless' is measured, not assumed).
    """
    from malign_logits import twp
    words, res, _calls = twp.expand(model, tok, context, dev, bmask)
    surface = {}
    for (surf, _t1), m in words.items():
        surface[surf] = surface.get(surf, 0.0) + float(m)
    return surface, res


def closure_rider(model, tok, dev, context, rider_words, nl_ids, torch):
    """The class-branch rider: p(newline-family | context + " w") for a small
    set of rider words (the poem's actual next word + top class members by
    expand mass). One padded batch of full sequences — no cache assumed;
    malign [5737]: twp never cached, so the rider shares the honest price."""
    ctx_ids = tok(context, return_tensors="pt")["input_ids"][0].tolist()
    rows = []
    for w in rider_words:
        wid = tok(" " + w, add_special_tokens=False)["input_ids"]
        if 0 < len(wid) <= 6:
            rows.append((w, ctx_ids + wid))
    if not rows:
        return {}
    maxlen = max(len(ids) for _, ids in rows)
    pad = tok.eos_token_id or 0
    batch = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    att = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, (_, ids) in enumerate(rows):
        batch[i, :len(ids)] = torch.tensor(ids)
        att[i, :len(ids)] = 1
    with torch.no_grad():
        out = model(input_ids=batch.to(dev), attention_mask=att.to(dev))
    lg = out.logits.float()
    nl = torch.tensor(nl_ids, device=lg.device)
    return {w: float(torch.softmax(lg[i, len(ids) - 1, :], -1)[nl].sum())
            for i, (w, ids) in enumerate(rows)}


N_RIDER_CLASS = 8  # top class members by expand mass joining the closure batch


def smoke():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    k2w = build_rime_vocab()
    poems = load_poems(3)
    print(f"{len(poems)} poems loaded: "
          f"{[(p['scheme'], p['target_word'], p['target_key']) for p in poems]}",
          flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(SMOKE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMOKE_MODEL, dtype=torch.bfloat16).to(dev).eval()
    nl_ids = newline_ids(tok, model.config.vocab_size)
    print(f"newline family: {len(nl_ids)} token ids", flush=True)
    from malign_logits import twp
    bmask = twp.boundary_mask(tok, model.config.vocab_size)

    rows = []
    wrows = []
    for p in poems:
        cls = list(dict.fromkeys(
            (k2w.get(p["target_key"], []) + [p["target_word"], p["actual_word"]])))
        ncls = k2w.get(p["nonpartner_key"], [])[:60]
        print(f"\n== {p['id_human']} [{p['scheme']}] target '{p['target_word']}' "
              f"class n={len(cls)} nonpartner n={len(ncls)}", flush=True)
        for s in poem_slots(p["lines"], p["partner_line"]):
            surface, resid = expand_slot(model, tok, dev, s["context"], bmask)
            in_cls = set(cls)
            in_ncls = set(ncls)
            mass = sum(m for w, m in surface.items() if w.lower() in in_cls)
            nmass = sum(m for w, m in surface.items() if w.lower() in in_ncls)
            pa = surface.get(p["actual_word"], 0.0) + \
                surface.get(p["actual_word"].capitalize(), 0.0)
            top_cls = sorted((w for w in surface if w.lower() in in_cls),
                             key=lambda w: -surface[w])[:N_RIDER_CLASS]
            closes = closure_rider(model, tok, dev, s["context"],
                                   list(dict.fromkeys([p["actual_word"]] + top_cls)),
                                   nl_ids, torch)
            cm = sum(surface[w] for w in top_cls)
            close_w = (sum(surface[w] * closes[w] for w in top_cls if w in closes)
                       / cm if cm > 0 else None)
            for w, m in surface.items():
                wrows.append({"id_human": p["id_human"], "slot": s["slot"],
                              "surface": w, "prob": m})
            np_called = (s["slot"].replace("_partner_prior", str(p["partner_line"]))
                         in NONPARTNER_CALLED_AT[p["scheme"]]) or \
                        (s["slot"] == "end_partner_prior" and False)
            rows.append({"id_human": p["id_human"], "scheme": p["scheme"],
                         "key_resolves_own_rhyme": p["key_resolves_own_rhyme"],
                         "slot": s["slot"], "phase": s["phase"],
                         "class_mass": mass, "close_given_class": close_w,
                         "p_actual": pa, "p_close_actual": closes.get(p["actual_word"]),
                         "nonpartner_mass": nmass,
                         "resid_total": resid["total"], "resid_tail": resid["tail"],
                         "nonpartner_called_here": bool(np_called),
                         "n_words_stored": len(surface), "ctx_chars": len(s["context"])})
            print(f"  {s['slot']:16s} [{s['phase']:8s}] class {mass:.5f} "
                  f"nonp {nmass:.5f} close|cls {close_w if close_w is None else round(close_w,3)} "
                  f"p_act {pa:.5f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(os.path.join(OUT_DIR, "verse_fleet_smoke.parquet"))
    pd.DataFrame(wrows).to_parquet(
        os.path.join(OUT_DIR, "verse_fleet_smoke_words.parquet"))
    print(f"\nwrote verse_fleet_smoke.parquet: {len(df)} rows; "
          f"words store: {len(wrows)} rows", flush=True)


MANIFEST = os.path.join(REPO, "data/verse_fleet_slot_manifest.json")


def build_manifest():
    """The frozen slot manifest — every (prompt, slot) cell the fleet runs,
    as an artifact malign can execute and audit. Contexts resolved HERE, once;
    the fleet never re-derives a prefix. Verse: 9 slots x roster poems.
    Prose: 102 LITERARY (licence baseline) + battery (calibration anchor),
    one slot each at snippet end."""
    cells = []
    poems = load_poems(n=10**9)
    for p in poems:
        slots = poem_slots(p["lines"], p["partner_line"])
        # [5749] defect 1: in short poems two slots can land on the SAME
        # prefix (mid4==near etc.) — one measurement under two labels.
        # Flag-not-nudge, malign's §5: analysis fences, design unchanged.
        first_at = {}
        for sl in slots:
            if sl["context"] in first_at:
                sl["context_collides_with"] = first_at[sl["context"]]
            else:
                sl["context_collides_with"] = None
                first_at[sl["context"]] = sl["slot"]
        for sl in slots:
            cells.append({
                "cell_type": "verse", "id_human": p["id_human"],
                "scheme": p["scheme"], "era": p["era"],
                "slot": sl["slot"], "phase": sl["phase"],
                "context": sl["context"],
                "target_word": p["target_word"], "target_key": p["target_key"],
                "nonpartner_word": p["nonpartner_word"],
                "nonpartner_key": p["nonpartner_key"],
                "actual_word": p["actual_word"], "actual_key": p["actual_key"],
                "key_resolves_own_rhyme": p["key_resolves_own_rhyme"],
                "context_collides_with": sl["context_collides_with"]})
    cat = json.load(open(os.path.join(REPO, "data/prompt_categorisation.json")))
    lit = [r for r in cat["prompts"] if r.get("source") == "LITERARY"]
    for r in lit:
        cells.append({"cell_type": "prose_literary",
                      "prompt_id": r["prompt_id"], "slot": "prose",
                      "phase": "prose", "context": r["prompt"]})
    b = json.load(open(os.path.join(REPO, "data/m05_battery.json")))
    # calibration anchor is ~100 prompts BY PLAN, not the full battery (584
    # unique): stratified by block, proportional, seeded — the anchor only
    # needs enough overlap with the expand-era store to alarm on drift
    import random as _random
    rng = _random.Random(20260813)
    seen = set()
    by_block = {}
    for blk_name, blk in sorted(b["blocks"].items()):
        for t in blk["texts"]:
            txt = t if isinstance(t, str) else t.get("text", t.get("prompt"))
            if txt in seen:
                continue
            seen.add(txt)
            by_block.setdefault(blk_name, []).append(txt)
    total = sum(len(v) for v in by_block.values())
    for blk_name, txts in sorted(by_block.items()):
        k = max(1, round(100 * len(txts) / total))
        for txt in rng.sample(sorted(txts), min(k, len(txts))):
            cells.append({"cell_type": "prose_battery", "block": blk_name,
                          "slot": "battery", "phase": "battery",
                          "context": txt})
    # [5755] §3: the store keys on (model, prompt) — cells sharing a context
    # ACROSS poems collapse on ingest too; flag them so no analysis joins one
    # measurement into two poems' worth of precision
    by_ctx = {}
    for i, c in enumerate(cells):
        by_ctx.setdefault(c["context"], []).append(i)
    for ctx, idxs in by_ctx.items():
        if len(idxs) > 1:
            for i in idxs:
                others = [cells[j].get("id_human") or cells[j].get("prompt_id")
                          or cells[j].get("block") for j in idxs if j != i]
                cells[i]["context_shared_with"] = sorted(set(
                    o for o in others
                    if o != (cells[i].get("id_human") or cells[i].get("prompt_id")
                             or cells[i].get("block")))) or None
    n = {"verse": 0, "prose_literary": 0, "prose_battery": 0}
    for c in cells:
        c["context_len"] = len(c["context"])
        n[c["cell_type"]] += 1
    out = {"_provenance": {
               "design": "B ([5735]/[5736]/[5737]); plan_verse_fleet.md frozen 2026-08-13",
               "roster": os.path.basename(ROSTER), "seed": 20260813,
               "rime_key": "v2.1 phonemic (final stressed syll from first vowel; glide-strip; schwa normalized)",
               "slots_per_poem": 9,
               "resolve_rate_denominator": "quote key_resolves_own_rhyme rates over RHYMED cells only — the 540 unrhymed cells are False by construction ([5749] §2)",
               "n_distinct_contexts_note": "1,786 distinct contexts over 1,820 cells — completeness gates use 250 x 1,786 = 446,500, NEVER 455,000 ([5755] §2)",
               "amendments": "[5749]+[5755]: context_collides_with (13 poems, duplicate prefixes flagged) + context_len (4-char primer tail fence)",
               "battery_rule": "100 of 584 unique, stratified by block proportional, seed 20260813",
               "counts": n, "n_cells": len(cells)},
           "cells": cells}
    with open(MANIFEST, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {MANIFEST}: {len(cells)} cells {n}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke()
    elif mode == "manifest":
        build_manifest()
