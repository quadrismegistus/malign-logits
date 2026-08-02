"""C1-C2 producer: institutional vs neutral JS on the isolated sft->preference step.

THE PRODUCER THAT DID NOT EXIST. [1037]/[1058]/[1062] were computed in-session and the
numbers died with the session at the MCP version change. Every analytic choice that moved
a count in that thread is a named constant here, so the next disagreement is a diff of two
files rather than a diff of two memories.

    uv run python c1_institutional_neutral.py

DECLARED CHOICES, each one the thing that moved a number on 2026-07-31:

  POPULATION   English only, status ACTIVE, DISTINCT TEXTS, one row per text chosen by
               the ranked pick applied WITHIN the stratum. Row-wise iteration counts one
               institutional text twice (two ACTIVE rows over one string) and inflates
               the rank-sum -- the [1058].2 defect, 12 -> 11. contrast.by_field() is
               row-wise, so it is NOT used here. Prompt.find() is NOT used either: it
               ranks across strata ([1069].4's dual-active text), so it named the
               contradiction row as a member of the neutral 135.
  RESIDUAL     Kept as a bin. cell.js() default; never renormalised over survivors.
               Dropping it moved olmo z from 4.10 to 0.48 ([1029].2).
  SIDEDNESS    Two-sided. rank_sum() already returns two-sided; the one-sided count was
               malign's [1060] defect, 14 -> 13.
  STEP         ego->superego, the ISOLATED preference step, selected by STAGE not by
               position -- last-step selection takes dpo->rlvr for olmo/tulu, which is a
               verification step and not the same object ([1015].1).
  STORE        pinned by counting the `true_word_probs` STASH at read time -- the object
               Step.cell() actually measures -- and printed with the table. NOT
               data/twp_grid_v3, which is the rsync transport for the repair pass, runs
               ahead of the stash by whatever is un-ingested, and is gitignored.

Run it from the repo:  uv run python scripts/c1_institutional_neutral.py
Re-freeze the population:  ... --freeze
"""
import sys, os, json, hashlib, collections

# Repo root from THIS FILE's location, never a hardcoded checkout path. The earlier
# `~/github/malign-logits` worked only from the seat it was written at and would have
# pinned one machine's layout into a file whose purpose is to re-run elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits import MODEL_FAMILIES
from malign_logits.step import Step
from malign_logits.checkpoint import Checkpoint
from malign_logits.contrast import rank_sum
from malign_logits.prompts import Prompts, _rank

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_population.json")


def spec(prompts):
    """(prompt_id, text) pairs sorted by id, and the hash over them.

    THE ID ALONE IS NOT ENOUGH AND THE TEXT ALONE IS NOT ENOUGH. The id is the stable
    slug; the text is the join key against the store. An edit to a text under a kept id
    would leave an id-only spec agreeing while the measured population changed, which is
    the [1058] failure one layer down. Hashing the pairs catches both.
    """
    pairs = sorted((p.id, p.text) for p in prompts)
    blob = json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode()
    return pairs, hashlib.sha256(blob).hexdigest()


def freeze(inst, neut):
    """Write the enumerable artifact [1071].2 requires. Derivation is still done twice;
    this is step (iii), the thing both implementations then measure FROM."""
    (ip, ih), (np_, nh) = spec(inst), spec(neut)
    doc = {
        "_rule": "[1071].2 DERIVE TWICE, DIFF, FREEZE, MEASURE FROM THE FROZEN SPEC",
        "_derivation": "language=en, status=ACTIVE, domain in {institutional,neutral}; "
                       "DISTINCT TEXTS, one row per text chosen by the package's _rank "
                       "ordering applied WITHIN the requested domain (not Prompt.find(), "
                       "which ranks across strata and would name the other stratum's row)",
        "institutional": {"n": len(ip), "sha256": ih, "prompts": ip},
        "neutral": {"n": len(np_), "sha256": nh, "prompts": np_},
        "combined_sha256": hashlib.sha256((ih + nh).encode()).hexdigest(),
    }
    with open(FROZEN, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    return doc


def check_frozen(inst, neut):
    """Refuse to measure against a population that has drifted from the frozen one."""
    if not os.path.exists(FROZEN):
        return None
    doc = json.load(open(FROZEN))
    out = []
    for key, prompts in (("institutional", inst), ("neutral", neut)):
        pairs, h = spec(prompts)
        # SELF-CONSISTENCY FIRST. Comparing the live hash against the artifact's stored
        # sha256 field trusts that field; an artifact whose `prompts` list was edited
        # without recomputing its hash would then pass while naming a different
        # population. Caught by mutation test, not by review. Recompute from the list.
        stored = [(i, t) for i, t in doc[key]["prompts"]]
        blob = json.dumps(sorted(stored), ensure_ascii=False, sort_keys=True).encode()
        if hashlib.sha256(blob).hexdigest() != doc[key]["sha256"]:
            out.append(f"{key}: ARTIFACT IS INTERNALLY INCONSISTENT -- its sha256 does "
                       f"not match its own prompt list. Re-freeze; do not measure.")
            continue
        if len(stored) != doc[key]["n"]:
            out.append(f"{key}: ARTIFACT n={doc[key]['n']} but lists {len(stored)}.")
            continue
        if h != doc[key]["sha256"]:
            live, frozen_ = dict(pairs), dict(stored)
            have, want = set(live), set(frozen_)
            # A text edited under a KEPT id leaves the id sets identical, so an id-only
            # diff would report a mismatch it cannot explain. Name the texts too.
            edited = [i for i in have & want if live[i] != frozen_[i]]
            msg = (f"{key}: HASH MISMATCH  frozen n={doc[key]['n']} now n={len(pairs)}"
                   f"\n      only in live catalogue: {sorted(have - want) or '-'}"
                   f"\n      only in frozen spec:    {sorted(want - have) or '-'}")
            for i in sorted(edited):
                msg += (f"\n      TEXT CHANGED under kept id {i}:"
                        f"\n          frozen: {frozen_[i]!r}"
                        f"\n          live:   {live[i]!r}")
            out.append(msg)
    return out


def pin_store():
    """Payloads and models in the `true_word_probs` STASH at read time.

    THE STASH, NOT `data/twp_grid_v3`. That directory is the rsync TRANSPORT for the
    repair pass; its line count runs ahead of the store by whatever is not yet ingested
    -- 10,014 lines ahead on 2026-07-31 (90,350 against 80,336). `Step.cell()` measures
    the stash, so pinning the transport stamps the provenance of an object the producer
    never reads, and the pin is one of the qualifiers the clause carries.

    It is also the portability bug: `data/twp_grid_v3/` is gitignored with zero tracked
    files, so a transport-based pin raises FileNotFoundError on a fresh clone -- for
    precisely the independent implementer this file exists to serve.
    """
    from malign_logits.cache import get_cache
    n, models = 0, set()
    cm = get_cache()
    return (cm.count("true_word_probs"),
            len(cm.distinct("true_word_probs", "model")))


def distinct_texts(domain):
    """DISTINCT TEXTS, not rows, and the row is ranked WITHIN the requested stratum.

    Prompts.where() is row-wise and ACTIVE-only; 55 institutional rows sit over 54
    strings. Deduping on the string is the [1058].2 correction.

    THE RANKED PICK RUNS AMONG THE REQUESTED DOMAIN'S ROWS, NOT ACROSS ALL OF THEM.
    `Prompt.find()` ranks over every row carrying the text, including rows in the OTHER
    stratum, and `_rank` breaks ties on group membership before anything else. On
    'He was captive and chose to' -- ACTIVE `neutral` on setd_to_U_6, ACTIVE
    `contradiction` on store_g013_A -- find() returns the CONTRADICTION row, so the
    frozen neutral stratum named a contradiction row as one of its 135 members.

    [1069].4 rules that a dual-active text enters BOTH strata. Implemented here as a
    decision rather than as a side effect of find()'s tie-break: filter to the domain
    first, then rank. The measured population is unchanged either way -- both rows carry
    the same text and only the text is measured -- but the DECLARED population is what
    the frozen artifact is for, and it was naming the wrong row.
    """
    by_text = collections.defaultdict(list)
    for p in Prompts.where(language="en", domain=domain):
        by_text[p.text].append(p)
    # _rank is the package's own ordering (ACTIVE > DISPUTED > RETIRED, then grouped,
    # then role-bearing). Imported rather than reimplemented: a second hand-rolled
    # ranking is the defect this whole thread has been about.
    return [sorted(ps, key=lambda p: _rank(p.row))[0] for _, ps in sorted(by_text.items())]


def isolated_steps():
    """Families with BOTH an ego and a superego arm: the isolated preference step."""
    out = {}
    for key, fam in MODEL_FAMILIES.items():
        if fam.ego and fam.superego:
            out[key] = Step(Checkpoint(fam.ego), Checkpoint(fam.superego))
    return out


def main():
    n_payloads, n_models = pin_store()
    inst, neut = distinct_texts("institutional"), distinct_texts("neutral")

    if "--freeze" in sys.argv:
        doc = freeze(inst, neut)
        print(f"FROZEN -> {FROZEN}")
        print(f"  institutional n={doc['institutional']['n']} "
              f"sha256={doc['institutional']['sha256'][:16]}")
        print(f"  neutral       n={doc['neutral']['n']} "
              f"sha256={doc['neutral']['sha256'][:16]}")
        print(f"  combined      sha256={doc['combined_sha256']}")
        return

    drift = check_frozen(inst, neut)
    if drift:
        print("REFUSING TO MEASURE -- the live catalogue has drifted from the frozen "
              "population.\nRe-derive at both seats, diff, and re-freeze; do not "
              "silently re-measure.\n")
        for d in drift:
            print("   ", d)
        return

    print(f"STORE PINNED   {n_payloads} payloads / {n_models} models")
    print(f"POPULATION     nI={len(inst)}  nN={len(neut)}   "
          f"(English, ACTIVE, distinct texts)"
          + ("  [MATCHES FROZEN SPEC]" if drift is not None else "  [NO FROZEN SPEC]"))
    print(f"CHOICES        residual KEPT as bin | TWO-SIDED | step ego->superego\n")

    rows, partial = [], {}
    for key, step in sorted(isolated_steps().items()):
        A, B, miss = [], [], 0
        for bucket, prompts in ((A, inst), (B, neut)):
            for p in prompts:
                c = step.cell(p.text)
                v = c.js() if c is not None else None
                if v is None:
                    miss += 1
                else:
                    bucket.append(v)
        # COMPLETE COVERAGE OR NOTHING. A family whose arms are partly scored would
        # otherwise enter the count on a subset and the summary line would read
        # "x of 17" with one family standing on a different population. This is not
        # hypothetical: the store is growing under the repair pass and the next
        # family due to arrive, olmo-32b, currently has one arm at 51 of 189.
        if len(A) != len(inst) or len(B) != len(neut):
            if A or B:
                partial[key] = (len(A), len(B))
            continue
        # Any family reaching here has miss == 0 by construction: the guard above
        # admits only complete coverage. There is deliberately no "incomplete but
        # counted" branch -- that state is what the guard exists to make unreachable.
        U, z, p2 = rank_sum(A, B)
        rows.append((key, len(A), len(B), z, p2))

    rows.sort(key=lambda r: (r[4] if r[4] is not None else 1.0))
    print(f"{'family':<16}{'nI':>4}{'nN':>5}{'z':>8}{'p two-sided':>14}  sig")
    sig = 0
    for key, nA, nB, z, p2 in rows:
        mark = ""
        if p2 is not None and p2 < 0.05:
            mark = "*"
            sig += 1
        print(f"{key:<16}{nA:>4}{nB:>5}{z:>8.2f}{p2:>14.2e}  {mark}")
    print(f"\n{sig} of {len(rows)} significant at p<0.05, two-sided")
    print(f"{sum(1 for r in rows if r[3] > 0)} of {len(rows)} positive in DIRECTION")
    if partial:
        print("\nEXCLUDED, PARTIAL COVERAGE -- these are NOT in the count above;"
              "\nthey enter it only when both arms reach the full population:")
        for k, (a, b) in sorted(partial.items()):
            print(f"    {k:<16} nI={a:<4} nN={b}")


if __name__ == "__main__":
    main()
