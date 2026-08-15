"""Does the riser RESEMBLE the faller, or is it ordered BELOW it? Same cells, both instruments.

    uv run --with lemminflect --with transformers --with torch python x_resemblance_test.py

RH's objection, which is what this exists to answer. Section 4 of the finding
claims the campaign's similarity instruments failed because the relation is
contiguity rather than resemblance. But at `took off her ___` **every candidate
is a wearable** -- `shirt`, `glasses`, `gloves`, `coat`. The resemblance is not
absent, it is undeniable. So the failure cannot be read as "there was nothing
there to find" until somebody measures whether the instrument can see it.

Two things had to be fixed before the comparison was fair.

**FIRST, THE TWO INSTRUMENTS WERE NEVER RUN ON THE SAME PAIRS.** The four
similarity instruments at ledger clause 6 and plan V's six geometric grains all
ran on the M01 population. X runs on the liminal/explicit battery, which is a
different population and explicitly not poolable with it. The contrast as
written was between a test on one set of words and a test on another. This
script runs both on the X cells.

**SECOND, `v_bare_vectors.npz` COVERS 30% OF THIS MATERIAL.** It was built for
the M01 vocabulary. Twelve of 33 cells at `took off her` had both words in it,
and the twelve are exactly the clothing words that also happen to appear in a
verb-heavy population -- a biased third. This encodes the X word set directly at
the same model and layer, into its own cache, and touches nothing of V's.

THE FAIR COMPARISON, and it is not the one section 5 first drew.

An intimacy scale scores a WORD, so a cell yields a difference: is the riser
less intimate than the faller? A cosine scores a PAIR, so there is no difference
to take -- it needs a reference pair. The two tests are not the same shape and
putting them side by side was comparing a paired test to nothing.

So both instruments are asked the question that HAS the same shape for each:

    at this cell the model had several risers to choose from.
    Does the instrument pick the one it actually chose?

    resemblance  is cos(faller, TRUE riser)     >  cos(faller, OTHER riser)?
    ordering     is intimacy(TRUE riser)        <  intimacy(OTHER riser)?

Same cells, same alternatives, same paired binomial. Whatever separates them
after that is a fact about the instruments and not about which test was run.

The weaker paired form is also reported -- riser against FALLER rather than
against the other risers -- because it is what section 5 currently rests on and
the reader should see both. It is the easier test: the faller is by construction
the word alignment moved off, so any scale aligned with the operation at all
will order it above the riser.

CONTROLS. Three, and they are three different questions, not three strengths of
one. Reporting only the first would have been the easy result.

    STATIC  the mean over words PRESENT IN BOTH ARMS THAT DID NOT MOVE. This is
            the campaign's decoy rule -- available in the slot, did not move --
            and it is the control that makes the head-to-head parallel to the
            weak form below: both then ask whether the word alignment moved
            ONTO differs from a word it left alone.
    ALL     the mean over every other riser at the cell
    ARGMAX  the runner-up riser alone -- the word the model came closest to
            choosing instead. The near-miss is the control that can fail, but
            NOTHING IN THE CLAIM REQUIRES EITHER INSTRUMENT TO PASS IT: "moves
            off intimate items onto peripheral ones" does not entail that the
            top riser is the least intimate riser. It is reported as a bound.

UNIT: the cell, one base>superego pair at one prompt. Never pooled across
prompts -- contiguity is within a scene and a pooled riser list has no scene
attached. Not the frozen population; descriptive throughout; not a rate.
"""
import collections
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
#: same model and layer as v_regions.py, so a cosine here means what one there
#: means. Its cache is NOT reused or written -- different population.
MODEL = "BAAI/bge-m3"
LAYER_FRAC = 0.25
CACHE = os.path.join(CAMP, "results", "x_bge_vectors.npz")
SCENES = [("sexual_liminal_6", "She slowly took off her ___"),
          ("sexual_liminal_7", "He slowly took off his ___"),
          ("sexual_explicit_3", "reached for his ___")]
#: the coded scale, averaged over its two model runs; they agree at +0.888.
#: Only the two `took off` frames were coded, so the ordering instrument is
#: absent at `reached for his` and the table says so rather than borrowing it.
SCALE = ["D_opus", "D_sonnet"]


def embed(words):
    """bge-m3 bare type vectors, own cache, exact word list or re-encode."""
    import numpy as np
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["words"]) == list(words):
            print("reusing %s" % os.path.basename(CACHE))
            return z["X"]
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL)
    mod.eval()
    L = int(round(LAYER_FRAC * mod.config.num_hidden_layers))
    out, B = [], 64
    for i in range(0, len(words), B):
        ch = words[i:i + B]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            hs = mod(**enc, output_hidden_states=True).hidden_states
        idx = enc["attention_mask"].sum(1) - 2
        out.append(hs[L][torch.arange(len(ch)), idx].float().numpy())
        if (i // B) % 10 == 0:
            print("  embed %d/%d" % (i, len(words)), flush=True)
    X = np.vstack(out)
    np.savez_compressed(CACHE, X=X, words=np.array(words, dtype=object))
    return X


def rows_for(st, model, prompt):
    k = dict(TWP); k["model"] = model; k["prompt"] = prompt
    try:
        v = st[k]
    except Exception:
        return None
    return v.get("rows") if isinstance(v, dict) else None


def cells():
    """One record per (scene, pair): faller, ranked risers. No embedding yet."""
    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from malign_logits import experiments as E
    from m05_sites import prepare

    st = get_cache()._stash("true_word_probs")
    src = inspect.getsource(E)
    #: `.isascii()` is load-bearing: the battery has Chinese twins under
    #: neighbouring keys and a plain dict lookup silently took one once.
    P = {k: v for k, v in re.findall(
        r'"((?:sexual|violence)_(?:liminal|explicit)_\d+)":\s*"([^"]+)"', src)
        if v.isascii()}

    #: THROUGH `Checkpoint`, NOT THE RAW FILE. This hand-rolled the
    #: `["models"]` shape of `model_registry.json` -- one of 16 consumers
    #: that did, each a place a schema change breaks silently.
    #: `.record` and not the attributes: the rows are read with `.get()`
    #: below, and `Checkpoint.__getattr__` RAISES on an unknown field where
    #: `.get()` returns None. Handing on plain dicts preserves that exactly.
    #: **This is not @lacan's reverted shim** -- that kept a routing `.get()`
    #: so the hand-rolled LOOKUP survived. Here the source is replaced and
    #: the shape assumption goes with it; the rows being dicts afterwards is
    #: not the defect.
    from malign_logits.checkpoint import Checkpoint as _CP
    reg = [cp.record for cp in _CP.all()]
    fam = collections.defaultdict(list)
    for m in reg:
        fam[m.get("family")].append(m)
    pairs = []
    for ms in fam.values():
        b = next((m for m in ms if m.get("position") == "base"), None)
        a = next((m for m in ms if m.get("position") == "superego"), None)
        if b and a:
            pairs.append((b["model_id"], a["model_id"]))
    pairs.sort()

    out = []
    for tag, label in SCENES:
        p = P[tag]
        for b, a in pairs:
            rb, ra = rows_for(st, b, p), rows_for(st, a, p)
            if not rb or not ra:
                continue
            ob, pb = prepare(rb)
            oa, pa = prepare(ra)
            mv = movement({w: pb[w] for w in ob}, {w: pa[w] for w in oa}, CANONICAL)
            F = [w for w in mv.fallers if w != RESIDUAL_KEY]
            R = [w for w in mv.risers if w != RESIDUAL_KEY]
            if not F or len(R) < 2:
                continue
            key = mv.excess if mv.rule.null_test else mv.delta
            #: the decoy population: in BOTH arms, classed as neither faller nor
            #: riser by the rule. Not a threshold of our own -- the rule already
            #: decided what counts as movement and inventing a second one here
            #: would let the control drift away from the thing it controls for.
            moved = set(mv.fallers) | set(mv.risers) | {RESIDUAL_KEY}
            static = [w for w in ob if w in pa and w not in moved]
            out.append(dict(scene=label, tag=tag, base=b.split("/")[-1],
                            faller=sorted(F, key=lambda w: mv.delta.get(w, 0.0))[0],
                            risers=sorted(R, key=lambda w: -key.get(w, 0.0)),
                            static=static))
    return out


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats

    C = cells()
    vocab = sorted({c["faller"] for c in C} | {w for c in C for w in c["risers"]}
                   | {w for c in C for w in c["static"]})
    print("%d cells over %d scenes, %d distinct types to embed" % (len(C), len(SCENES), len(vocab)))
    print("   static-control pool per cell: median %d words, min %d\n"
          % (int(np.median([len(c["static"]) for c in C])), min(len(c["static"]) for c in C)))
    X = embed(vocab)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    wi = {w: i for i, w in enumerate(vocab)}

    #: the floor. Two types drawn at random from THIS vocabulary, which is the
    #: only reference that makes a cosine of 0.70 readable as high or not.
    rng = np.random.default_rng(0)
    ii, jj = rng.integers(0, len(vocab), 60000), rng.integers(0, len(vocab), 60000)
    m = ii != jj
    floor = np.sum(X[ii[m]] * X[jj[m]], axis=1)

    W = pd.read_csv(os.path.join(CAMP, "results", "x_coder_words.csv"))
    W["intimacy"] = W[SCALE].mean(axis=1)
    IN = W.dropna(subset=["intimacy"]).set_index("word")["intimacy"].to_dict()

    rows = []
    for c in C:
        f, R = c["faller"], c["risers"]
        r, others = R[0], R[1:]
        d = dict(scene=c["scene"], base=c["base"], faller=f, riser=r,
                 runner_up=others[0], n_risers=len(R))
        d["cos_true"] = float(X[wi[f]] @ X[wi[r]])
        d["cos_all"] = float(np.mean([X[wi[f]] @ X[wi[o]] for o in others]))
        d["cos_argmax"] = float(X[wi[f]] @ X[wi[others[0]]])
        d["cos_static"] = (float(np.mean([X[wi[f]] @ X[wi[s]] for s in c["static"]]))
                           if c["static"] else None)
        if f in IN and r in IN:
            d["int_faller"], d["int_true"] = IN[f], IN[r]
            oi = [IN[o] for o in others if o in IN]
            d["int_all"] = float(np.mean(oi)) if oi else None
            d["int_argmax"] = IN[others[0]] if others[0] in IN else None
            si = [IN[s] for s in c["static"] if s in IN]
            #: 4 coded words is the floor for a cell mean; below that one
            #: unusual word carries the control.
            d["int_static"] = float(np.mean(si)) if len(si) >= 4 else None
            d["n_static_coded"] = len(si)
        rows.append(d)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(CAMP, "results", "x_resemblance_cells.csv"), index=False)

    def paired(sub, hi, lo, want_hi_bigger):
        """Sign test on hi - lo. `want_hi_bigger` is what the instrument claims."""
        s = sub.dropna(subset=[hi, lo])
        if len(s) < 6:
            return None
        d = (s[hi] - s[lo]).values
        k = int((d > 0).sum()) if want_hi_bigger else int((d < 0).sum())
        return dict(n=len(d), k=k, pct=100.0 * k / len(d), mean=float(d.mean()),
                    p=float(stats.binomtest(k, len(d), 0.5).pvalue))

    print("\nANISOTROPY FLOOR   cos(random type, random type) in this vocabulary")
    print("   mean %.3f   sd %.3f   5th-95th %.3f-%.3f\n"
          % (floor.mean(), floor.std(), *np.percentile(floor, [5, 95])))

    print("=" * 92)
    print("THE HEAD-TO-HEAD: at a cell with several risers, does the instrument pick the true one?")
    print("=" * 92)
    hdr = "%-30s %-22s %5s %14s %10s %10s" % ("scene", "instrument / control", "n", "picks true", "mean diff", "p")
    for tag, label in SCENES:
        sub = D[D.scene == label]
        print("\n%s   (%d cells)" % (label, len(sub)))
        print(hdr)
        tests = [("resemblance vs STATIC", "cos_true", "cos_static", True),
                 ("ordering    vs STATIC", "int_true", "int_static", False),
                 ("resemblance vs ALL", "cos_true", "cos_all", True),
                 ("ordering    vs ALL", "int_true", "int_all", False),
                 ("resemblance vs ARGMAX", "cos_true", "cos_argmax", True),
                 ("ordering    vs ARGMAX", "int_true", "int_argmax", False)]
        for name, hi, lo, big in tests:
            if hi not in sub or lo not in sub or sub[lo].isna().all():
                print("   %-22s %s" % (name, "not coded at this scene"))
                continue
            t = paired(sub, hi, lo, big)
            if t is None:
                print("   %-22s %s" % (name, "under 6 cells"))
                continue
            print("   %-22s %5d %8d (%3.0f%%) %10.3f %10.4f"
                  % (name, t["n"], t["k"], t["pct"], t["mean"], t["p"]))

    print("\n" + "=" * 92)
    print("THE WEAKER PAIRED FORM: riser against the FALLER -- what was posted to RH")
    print("PER FRAME. The 68%/p=1e-3 quoted earlier POOLED the two frames, doubling n on the")
    print("very axis section 3b's gender claim turns on. Two prompts agreeing is the result here,")
    print("not a pooled p-value.")
    print("=" * 92)
    for tag, label in SCENES:
        sub = D[D.scene == label]
        t = paired(sub, "int_true", "int_faller", False) if "int_faller" in sub else None
        if t is None:
            print("   %-30s not coded at this scene (the scale is the took-off word set)" % label)
            continue
        print("   %-30s n=%d   riser less intimate in %d (%.0f%%)   mean %+.1f   p=%.3f"
              % (label, t["n"], t["k"], t["pct"], t["mean"], t["p"]))

    print("\nwrote results/x_resemblance_cells.csv")


if __name__ == "__main__":
    main()
