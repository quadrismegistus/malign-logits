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
                parts.append("".join(ph.txt for ph in phs[vi:]))
            else:
                ipa = getattr(sylls[j], "ipa", None) or sylls[j].txt
                parts.append(str(ipa).replace("\u02c8", "").replace("\u02cc", ""))
        key = "|".join(x for x in parts if x) or None
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

SMOKE_MODEL = "HuggingFaceTB/SmolLM2-360M"


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
    return re.sub(r"[A-Za-z']+\W*$", "", line).rstrip()


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
        out.append({**r, "lines": lines, "partner_line": partner,
                    "target_word": tw, "target_key": rime_key(tw),
                    "nonpartner_word": nw, "nonpartner_key": rime_key(nw),
                    "actual_word": last_word(lines[3])})
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


def score_slot(model, tok, dev, context, class_words, nl_ids, torch):
    """One prefix pass (cached), then all candidates as one padded batch
    branching off the cache. Returns per-candidate (p_word, p_close) and the
    prefix-softmax for free reads."""
    enc = tok(context, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    prefix_logits = out.logits[0, -1, :].float()
    prefix_probs = torch.softmax(prefix_logits, -1)
    past = out.past_key_values

    # candidates: " word" continuation tokens
    cand_ids = []
    for w in class_words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if 0 < len(ids) <= 6:
            cand_ids.append((w, ids))
    if not cand_ids:
        return {}, prefix_probs
    maxlen = max(len(ids) for _, ids in cand_ids)
    pad = tok.eos_token_id or 0
    batch = torch.full((len(cand_ids), maxlen), pad, dtype=torch.long)
    for i, (_, ids) in enumerate(cand_ids):
        batch[i, :len(ids)] = torch.tensor(ids)
    batch = batch.to(dev)

    # expand cache along batch dim
    def expand_past(past, n):
        if hasattr(past, "batch_repeat_interleave"):
            p2 = past.batch_repeat_interleave(n)
            return p2
        return tuple(tuple(t.expand(n, -1, -1, -1).contiguous() for t in layer)
                     for layer in past)
    past_n = expand_past(past, len(cand_ids))
    with torch.no_grad():
        out2 = model(input_ids=batch, past_key_values=past_n, use_cache=False)
    logits2 = out2.logits.float()  # (n, maxlen, vocab)

    results = {}
    logp_first = torch.log_softmax(prefix_logits, -1)
    for i, (w, ids) in enumerate(cand_ids):
        lp = float(logp_first[ids[0]])
        for j in range(1, len(ids)):
            lp += float(torch.log_softmax(logits2[i, j - 1, :], -1)[ids[j]])
        close = float(torch.softmax(logits2[i, len(ids) - 1, :], -1)[
            torch.tensor(nl_ids, device=logits2.device)].sum())
        results[w] = {"p_word": float(torch.exp(torch.tensor(lp))),
                      "p_close": close, "n_tokens": len(ids)}
    return results, prefix_probs


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

    rows = []
    for p in poems:
        cls = list(dict.fromkeys(
            (k2w.get(p["target_key"], []) + [p["target_word"], p["actual_word"]])))
        ncls = k2w.get(p["nonpartner_key"], [])[:60]
        print(f"\n== {p['id_human']} [{p['scheme']}] target '{p['target_word']}' "
              f"class n={len(cls)} nonpartner n={len(ncls)}", flush=True)
        for s in poem_slots(p["lines"], p["partner_line"]):
            res, _ = score_slot(model, tok, dev, s["context"], cls, nl_ids, torch)
            mass = sum(r["p_word"] for r in res.values())
            close_w = (sum(r["p_word"] * r["p_close"] for r in res.values()) / mass
                       if mass > 0 else None)
            pa = res.get(p["actual_word"], {}).get("p_word", 0.0)
            nres, _ = score_slot(model, tok, dev, s["context"], ncls, nl_ids, torch)
            nmass = sum(r["p_word"] for r in nres.values())
            np_called = (s["slot"].replace("_partner_prior", str(p["partner_line"]))
                         in NONPARTNER_CALLED_AT[p["scheme"]]) or \
                        (s["slot"] == "end_partner_prior" and False)
            rows.append({"id_human": p["id_human"], "scheme": p["scheme"],
                         "slot": s["slot"], "phase": s["phase"],
                         "class_mass": mass, "close_given_class": close_w,
                         "p_actual": pa, "nonpartner_mass": nmass,
                         "nonpartner_called_here": bool(np_called),
                         "n_class": len(res), "ctx_chars": len(s["context"])})
            print(f"  {s['slot']:16s} [{s['phase']:8s}] class {mass:.5f} "
                  f"nonp {nmass:.5f} close|cls {close_w if close_w is None else round(close_w,3)} "
                  f"p_act {pa:.5f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(os.path.join(OUT_DIR, "verse_fleet_smoke.parquet"))
    print(f"\nwrote verse_fleet_smoke.parquet: {len(df)} rows", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke()
