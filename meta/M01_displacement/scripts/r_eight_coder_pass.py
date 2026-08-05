"""One random sample of 100 items from the minimal-pair population, coded by
EIGHT models, to measure inter-coder agreement on a DIRECTLY ASKED axis.

WHY THIS RUN EXISTS. P derived the paradigmatic/syntagmatic axis from a label
list by a mapping we wrote, and the derivation failed: among items where all
three coders agreed a relation EXISTS, they disagreed about the axis 49.7% of
the time, and the disagreement ran perfectly monotone in how many labels each
coder ticked (deepseek 1.30/annotation -> 6.7% both-axes; sonnet 1.59 -> 13.4%;
gpt 1.85 -> 17.4%). `code_relation_axis` asks the axis directly and forces ONE
label. This run measures whether that changes the agreement rate.

    THE COMPARISON IS TWO DISAGREEMENT RATES, NOT TWO ANSWER SETS.
    Nothing here is correlated against P's derived axis -- that variable is
    the artifact under investigation, and using it as ground truth would
    validate an instrument against the thing it was built to replace.
    `intensity` is carried unchanged from P's instrument precisely so it can
    serve as a CONTINUITY CONTROL: if intensity tracks P on the same items,
    any movement in axis and relation is attributable to the schema change.

SAMPLE. 100 rows drawn with a fixed seed from `data/r_population_k2.parquet`
(malign's build: 5,976 faller/riser pairs over 684 stems, both members, >=2
edges). The drawn frame is written to disk BEFORE any call, so the same 100
items reach all eight coders and a re-run is checkable rather than re-drawn.

METADATA IS PART OF THE STASH KEY. Confirmed intentional upstream and not
invertible: a re-run passing different metadata silently re-keys and re-pays at
full price while every receipt stays honest. So the metadata is built from the
frozen frame, byte-identical per item, and never from anything run-scoped.

TRANSPORT. Six of the eight batch; both DeepSeek models raise, because DeepSeek
has no batch API and the library refuses to invent the discount. `can_batch`
below CALLS the library's own validator rather than carrying a model list --
a list would drift the first time a provider ships batching.
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

#: same derivation the campaign's other producers use (q_primary.py):
#: HERE -> scripts, CAMPAIGN -> M01_displacement, ROOT -> the repo.
HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, ROOT)

from largeliterarymodels.batch import _validate_batchable
from malign_logits.tasks.code_relation_axis import RelationAxisTask, prepare

POP = os.path.join(ROOT, "data", "r_population_k2.parquet")
OUT = os.path.join(CAMPAIGN, "results")
FRAME = os.path.join(OUT, "r_eight_coder_sample_100.parquet")
FRAME_PAIRED = os.path.join(OUT, "r_eight_coder_paired_50x2.parquet")
FRAME_VERB = os.path.join(OUT, "r_eight_coder_verbpaired_50x2.parquet")
FRAME_DECOY = os.path.join(OUT, "r_decoys_100.parquet")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"

SEED = 20260805
N = 100          #: unpaired draw (pilot 1) — retained so it stays reproducible
N_STEMS = 50     #: PAIRED draw: 50 stems x 2 members = 100 items

MODELS = [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "google/gemini-3.6-flash",
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-5",
    "openai/gpt-4o-mini",
    "openai/gpt-5.4-mini",
]


def can_batch(model):
    """-> True if this model has a batch API, by asking the library.

    Wraps `batch._validate_batchable`, which RAISES for DeepSeek and local
    endpoints. Deliberately not a model-name list: the set of batch-capable
    providers is the library's fact to know, and a copy of it here would be
    wrong the first day a provider ships batching.
    """
    try:
        _validate_batchable(model)
        return True
    except Exception:
        return False


def draw(write=True):
    """The 100 items of PILOT 1, unpaired. Retained for reproducibility."""
    if os.path.exists(FRAME):
        df = pd.read_parquet(FRAME)
        print("sample: reusing frozen frame, %d rows" % len(df))
        return df
    pop = pd.read_parquet(POP)
    df = pop.sample(n=N, random_state=SEED).reset_index(drop=True)
    df = df[["stem", "member", "prompt", "faller", "riser", "domain", "n_edges"]]
    if write:
        os.makedirs(OUT, exist_ok=True)
        df.to_parquet(FRAME, index=False)
        print("sample: drew %d rows at seed %d -> %s" % (len(df), SEED, FRAME))
    else:
        print("sample: drew %d rows at seed %d (NOT written -- dry run)" % (len(df), SEED))
    return df


def draw_paired(write=True):
    """50 STEMS, BOTH MEMBERS, ONE ROW PER MEMBER — the design where more
    coders BUY power instead of costing it.

    Pilot 1 drew 100 rows at random from 5,976 over 684 stems and landed
    SIX usable pairs: 800 annotations for six within-stem comparisons. The
    unpaired test came back p=0.22 because it was fighting between-sentence
    variation — some sentences simply have more relatable continuations
    than others, and that noise swamps a one-coder-per-item effect.

    Pairing removes it by construction, exactly as the minimal-pair design
    removes it for `tail_excess`. Each coder rates BOTH members of a stem,
    so every coder contributes one within-stem difference and averaging
    over coders sharpens the per-stem estimate rather than gating it. The
    unit of analysis is the STEM (n=50); the coders are replicates.

    ONE ROW PER MEMBER. A stem can carry many faller/riser pairs; taking
    several from one member would make the members' rows non-comparable
    and reintroduce the item-level noise pairing exists to remove. The row
    kept per member is the one with the most edges, deterministically —
    the best-evidenced movement at that prompt.
    """
    if os.path.exists(FRAME_PAIRED):
        df = pd.read_parquet(FRAME_PAIRED)
        print("paired sample: reusing frozen frame, %d rows (%d stems)"
              % (len(df), df.stem.nunique()))
        return df
    pop = pd.read_parquet(POP)
    both = pop.groupby("stem").member.nunique()
    eligible = sorted(both[both == 2].index)
    rng = np.random.RandomState(SEED)
    stems = sorted(rng.choice(eligible, size=N_STEMS, replace=False).tolist())
    #: deterministic pick per (stem, member): most edges, then the
    #: alphabetically first faller/riser to break ties reproducibly.
    sub = pop[pop.stem.isin(stems)].sort_values(
        ["stem", "member", "n_edges", "faller", "riser"],
        ascending=[True, True, False, True, True])
    df = sub.groupby(["stem", "member"], as_index=False).first()
    df = df[["stem", "member", "prompt", "faller", "riser", "domain", "n_edges"]]
    df = df.sort_values(["stem", "member"]).reset_index(drop=True)
    assert len(df) == 2 * N_STEMS, "expected %d rows, got %d" % (2 * N_STEMS, len(df))
    assert df.groupby("stem").member.nunique().eq(2).all(), "a stem lost a member"
    if write:
        os.makedirs(OUT, exist_ok=True)
        df.to_parquet(FRAME_PAIRED, index=False)
        print("paired sample: %d stems x 2 = %d rows at seed %d -> %s"
              % (N_STEMS, len(df), SEED, FRAME_PAIRED))
    else:
        print("paired sample: %d stems x 2 = %d rows at seed %d (NOT written)"
              % (N_STEMS, len(df), SEED))
    return df


def _byu_pos():
    """word -> CLAWS tag from the BYU frequency list.

    CLAWS separates LEXICAL verbs (`vv*`) from the auxiliaries and modals:
    be/is/was -> vb*, have/has/had -> vh*, do/does/did -> vd*, and every
    modal -> vm. So `vv*` excludes them all by construction, without a
    hand-written list of function verbs.
    """
    pos = {}
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            w, t = f[-1].strip().lower(), f[-3].strip()
            if w and w not in pos:
                pos[w] = t
    return pos


def draw_verb_paired(write=True):
    """50 stems, both members, and BOTH WORDS LEXICAL VERBS on both sides.

    WHY THIS FILTER. The first paired pilot returned p=0.53 and the smell
    test showed why: relatedness is almost perfectly determined by whether
    the words carry content (P=0.96 vs P=0.02), and the two members of a
    stem routinely differ on it -- 16 of 50 stems compared a content pair
    against a function pair. The test was measuring which member happened
    to have content words move.

    `vv*` on both sides is validated against the coders' own judgements on
    800 annotations: P(related | both vv*) = 0.948, P(related | not) =
    0.069, with 8 false keeps and 33 false drops. It is chosen over
    nn/vv/jj because with verbs on both sides the axis question is
    well-posed -- two acts either substitute or sequence -- where
    `verb -> adjective` makes IN_PLACE_OF vs BESIDE harder to answer.

    WHAT IT DOES NOT CATCH, deliberately. Light verbs (`began`, `put`,
    `get`) are `vv*` by tag and absent from NLTK's stopwords, and all 8
    false keeps were this class. They are left to the coders'
    `a_is_content_word` flags rather than a hand list, because which verbs
    are "too light" is a construct judgement and hard-coding it here would
    make it untestable.
    """
    if os.path.exists(FRAME_VERB):
        df = pd.read_parquet(FRAME_VERB)
        print("verb-paired: reusing frozen frame, %d rows (%d stems)"
              % (len(df), df.stem.nunique()))
        return df
    pos = _byu_pos()
    vv = lambda w: str(pos.get(str(w).strip().lower(), "")).startswith("vv")
    pop = pd.read_parquet(POP)
    n0 = len(pop)
    pop = pop[[vv(a) and vv(b) for a, b in zip(pop.faller, pop.riser)]]
    both = pop.groupby("stem").member.nunique()
    eligible = sorted(both[both == 2].index)
    print("verb filter: %d of %d rows survive; %d stems have BOTH members"
          % (len(pop), n0, len(eligible)))
    if len(eligible) < N_STEMS:
        raise SystemExit("REFUSING: only %d eligible stems, need %d"
                         % (len(eligible), N_STEMS))
    rng = np.random.RandomState(SEED)
    stems = sorted(rng.choice(eligible, size=N_STEMS, replace=False).tolist())
    sub = pop[pop.stem.isin(stems)].sort_values(
        ["stem", "member", "n_edges", "faller", "riser"],
        ascending=[True, True, False, True, True])
    df = sub.groupby(["stem", "member"], as_index=False).first()
    df = df[["stem", "member", "prompt", "faller", "riser", "domain", "n_edges"]]
    df = df.sort_values(["stem", "member"]).reset_index(drop=True)
    assert len(df) == 2 * N_STEMS, "expected %d rows, got %d" % (2 * N_STEMS, len(df))
    assert df.groupby("stem").member.nunique().eq(2).all(), "a stem lost a member"
    assert all(vv(a) and vv(b) for a, b in zip(df.faller, df.riser)), "a non-verb survived"
    if write:
        os.makedirs(OUT, exist_ok=True)
        df.to_parquet(FRAME_VERB, index=False)
        print("verb-paired: %d stems x 2 = %d rows -> %s" % (N_STEMS, len(df), FRAME_VERB))
    else:
        print("verb-paired: %d stems x 2 = %d rows (NOT written)" % (N_STEMS, len(df)))
    return df


def draw_decoy(write=True):
    """The NEAR-MISS arm: the same 100 prompts and the same 100 fallers as the
    verb-paired pilot, with the riser replaced by a word that was AVAILABLE in
    that slot and DID NOT MOVE.

    WHY THIS IS THE CONTROL AND THE UNMARKED TWIN WAS NOT. Three pilots
    contrasted marked against unmarked sentences and all three came back null.
    That contrast cannot answer "are these pairs related or random", because
    BOTH arms are real faller/riser pairs -- alignment displaces at 82-94% of
    cells in every partition of the corpus, so the neutral twin is not a
    no-displacement condition. Holding the prompt and the faller fixed and
    swapping only the riser is the comparison that isolates having-moved.

    Built by `build_r_decoys.py`, which carries P's declared stationary rule
    verbatim (p_base >= CANONICAL.min_prob and |delta| <= 0.0005, highest
    co-occurrence, alphabetical tie-break) with one deliberate change: the
    decoy must be `vv*`, matching the verb filter this pilot's real arm uses.

    METADATA CARRIES `arm` HERE AND ONLY HERE. It is a true property of the
    item and it guarantees these keys cannot collide with the real arm's,
    which share stem/member/faller. The existing frames' metadata is left
    untouched -- adding a field there would silently re-key work already paid
    for. The coder never sees it: `prepare` takes prompt, A and B and nothing
    else.
    """
    df = pd.read_parquet(FRAME_DECOY)
    real = pd.read_parquet(FRAME_VERB)
    #: the decoy arm must be the SAME prompts and the SAME fallers, or the
    #: comparison is between two different item sets wearing one name.
    lhs = set(zip(real.stem, real.member, real.faller))
    rhs = set(zip(df.stem, df.member, df.faller))
    assert lhs == rhs, "decoy arm does not match the real arm on (stem, member, faller)"
    assert (df.riser.values != real.sort_values(["stem", "member"]).riser.values).all(), \
        "a decoy equals its own riser"
    print("decoy arm: %d rows, %d stems; prompts and fallers identical to %s"
          % (len(df), df.stem.nunique(), os.path.basename(FRAME_VERB)))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="everything except the calls: draw, build items, "
                         "resolve transport, show what would be spent")
    ap.add_argument("--freeze", action="store_true",
                    help="with --dry-run, also WRITE the frozen sample frame")
    ap.add_argument("--paired", action="store_true",
                    help="draw 50 STEMS x both members instead of 100 random "
                         "rows. The unit becomes the stem and coders become "
                         "replicates.")
    ap.add_argument("--verb", action="store_true",
                    help="paired draw restricted to LEXICAL VERBS (vv*) on "
                         "both sides of every faller/riser pair")
    ap.add_argument("--decoy", action="store_true",
                    help="the NEAR-MISS arm: same prompts and fallers as "
                         "--verb, riser replaced by an available word that "
                         "did not move. Overrides --verb/--paired.")
    ap.add_argument("--parallel", action="store_true",
                    help="run the four PROVIDERS concurrently, models within "
                         "a provider in series. Caps concurrency per provider "
                         "at --workers rather than twice it, so two of our own "
                         "jobs never contend for one rate limit.")
    ap.add_argument("--workers", type=int, default=4,
                    help="threads per model inside task.map (library default "
                         "4). With --parallel the total in flight is "
                         "workers x providers.")
    ap.add_argument("--sync", action="store_true",
                    help="force batch=False on EVERY model. Costs list price "
                         "and finishes in minutes; batching costs half and "
                         "queues for an unbounded time. Use when the data is "
                         "wanted now.")
    args = ap.parse_args(argv)

    #: EVERY CALL-FREE STEP ABOVE THE EARLY RETURN. A --dry-run that returns
    #: before a check leaves that check untested forever; that defect cost
    #: P four firings and it is cheaper to obey than to discover.
    if args.decoy:
        df = draw_decoy(write=False)
    else:
        _draw = draw_verb_paired if args.verb else (draw_paired if args.paired else draw)
        df = _draw(write=not args.dry_run or args.freeze)
    texts = [prepare(r.prompt, r.faller, r.riser) for r in df.itertuples()]
    #: metadata is KEY material -- built only from the frozen frame, never
    #: from anything that varies between runs. `arm` is added for the decoy
    #: draw ONLY: it is a true property of those items and it keeps their keys
    #: disjoint from the real arm's. Adding it to the existing frames would
    #: re-key annotations already bought.
    if args.decoy:
        metas = [dict(stem=r.stem, member=r.member, faller=r.faller,
                      riser=r.riser, arm="DECOY") for r in df.itertuples()]
    else:
        metas = [dict(stem=r.stem, member=r.member, faller=r.faller, riser=r.riser)
                 for r in df.itertuples()]

    def transport(m):
        return False if args.sync else can_batch(m)

    print("items %d  models %d  mode=%s"
          % (len(texts), len(MODELS), "SYNC (forced)" if args.sync else "batch where able"),
          flush=True)
    for m in MODELS:
        print("  %-40s batch=%s" % (m, transport(m)), flush=True)

    #: RESOLVED ABOVE THE DRY-RUN RETURN, not below it. Grouping computed after
    #: the early return would be a branch --dry-run can never exercise, which
    #: is the defect this file's own comment warns about four lines down.
    import collections as _c
    groups = _c.OrderedDict()
    for m in MODELS:
        groups.setdefault(m.split("/")[0], []).append(m)
    if args.parallel:
        print()
        print("PARALLEL: %d providers concurrently, %d workers each; models "
              "within a provider run in SERIES, so two of our own jobs never "
              "contend for one provider's rate limit."
              % (len(groups), args.workers))
        for prov, ms in groups.items():
            print("  %-12s %d model(s) in series: %s"
                  % (prov, len(ms), ", ".join(m.split("/")[1] for m in ms)))
        print("  max concurrent requests: %d providers x %d workers = %d"
              % (len(groups), args.workers, len(groups) * args.workers))

    if args.dry_run:
        print()
        print("  first item:")
        for line in texts[0].split(chr(10)):
            print("    %s" % line)
        print("  first metadata: %r" % (metas[0],))
        print("  members: %s" % dict(df.member.value_counts()))
        print("  domains: %s" % dict(df.domain.value_counts()))
        print("  distinct stems: %d of %d rows" % (df.stem.nunique(), len(df)))
        print("  would issue %d calls (%d items x %d models)"
              % (len(texts) * len(MODELS), len(texts), len(MODELS)))
        print()
        print("--dry-run: sample drawn, items built, transport resolved. "
              "NO CALLS MADE.")
        return 0

    def run_one(m):
        """One model, start to finish. Returns (model, summary dict, log lines).

        Output is BUFFERED and returned rather than printed, because with
        providers running concurrently a shared stdout interleaves partial
        lines from four jobs and the log stops being readable at exactly the
        moment something goes wrong in one of them.
        """
        buf = []
        b = transport(m)
        buf.append("=== %s  (batch=%s)" % (m, b))
        task = RelationAxisTask()
        errors, per_item = {}, {}
        try:
            res = task.map(texts, model=m, metadata_list=metas, batch=b,
                           num_workers=args.workers,
                           errors=errors, per_item_usage=per_item)
        except Exception as e:
            #: PRINT THE WHOLE MESSAGE, same reason as the serial path.
            buf.append("  FAILED: %s" % type(e).__name__)
            buf.extend("    | %s" % ln for ln in str(e).split("\n"))
            buf.append(traceback.format_exc())
            return m, {"ok": 0, "batch": b, "error_type": type(e).__name__,
                       "error": str(e)}, buf
        ok = sum(1 for r in res if r is not None)
        buf.append("  parsed %d/%d  errors %d" % (ok, len(res), len(errors)))
        buf.append("  usage: %s" % task.usage.summary_line())
        return m, {"ok": ok, "n": len(res), "errors": len(errors),
                   "batch": b, "certify": task.certify_raw()}, buf

    if args.parallel:
        #: ONE THREAD PER PROVIDER, models within a provider run SEQUENTIALLY.
        #: `num_workers` is a ThreadPoolExecutor inside the library (llm.py),
        #: not multiprocessing, so nesting is safe; the usage tracker and the
        #: fail-fast breaker are per-Task-instance, so eight instances share no
        #: mutable state. The grouping exists because two models on ONE
        #: provider would contend for that provider's rate limit and pay it
        #: back in 429 retries. `groups` is built above the dry-run return.
        from concurrent.futures import ThreadPoolExecutor

        def run_provider(ms):
            out = []
            for m in ms:
                out.append(run_one(m))
            return out

        summary = {}
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            for chunk in ex.map(run_provider, groups.values()):
                for m, s, buf in chunk:
                    print("\n" + "\n".join(buf), flush=True)
                    summary[m] = s
        tag = "_decoy" if args.decoy else ("_verb" if args.verb else ("_paired" if args.paired else ""))
        with open(os.path.join(OUT, "r_eight_coder_runlog%s.json" % tag), "w") as fh:
            json.dump(summary, fh, indent=1, default=str)
        print("\nwrote runlog")
        return 0

    summary = {}
    for m in MODELS:
        b = transport(m)
        print("\n=== %s  (batch=%s)" % (m, b), flush=True)
        task = RelationAxisTask()
        errors, per_item = {}, {}
        try:
            res = task.map(texts, model=m, metadata_list=metas, batch=b,
                           num_workers=args.workers,
                           errors=errors, per_item_usage=per_item)
        except Exception as e:
            #: PRINT THE WHOLE MESSAGE. Library exceptions on this path carry
            #: operator resolutions in their text; truncating them destroys the
            #: diagnosis to preserve the run, which is what happened on the
            #: first pass and cost the two gemini failures.
            print("  FAILED: %s" % type(e).__name__, flush=True)
            for line in str(e).split("\n"):
                print("    | %s" % line, flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            summary[m] = {"ok": 0, "batch": b, "error_type": type(e).__name__,
                          "error": str(e)}
            continue
        ok = sum(1 for r in res if r is not None)
        print("  parsed %d/%d  errors %d" % (ok, len(res), len(errors)), flush=True)
        print("  usage: %s" % task.usage.summary_line(), flush=True)
        summary[m] = {"ok": ok, "n": len(res), "errors": len(errors),
                      "batch": b, "certify": task.certify_raw()}

    tag = "_decoy" if args.decoy else ("_verb" if args.verb else ("_paired" if args.paired else ""))
    with open(os.path.join(OUT, "r_eight_coder_runlog%s.json" % tag), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("\nwrote runlog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
