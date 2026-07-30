"""Risers and fallers: ONE implementation, importable, for logits AND true_word_probs.

    from malign_logits.movement import movement, CANONICAL, DRAW

    m = movement(pre_word_probs, post_word_probs)       # true_word_probs dicts
    m = movement_from_logits(pre_logits, post_logits)   # full-vocabulary arrays
    m.risers, m.fallers, m.excess["scream"], m.diagnostics

WHY THIS EXISTS. Until now nothing in the package defined a riser or a faller. The
definitions lived in fourteen scripts, and `scripts/f13_movement_table.py` was written to
end that -- its own docstring records the cost: "every seat derived fallers and risers on
the fly from the logits, and the derivations disagreed -- 1,650 cells against 3,366 on the
same question, because the thresholds and the cell filters lived in three scripts instead
of one file." **It fixed the logits path and the true_word_probs path never adopted it.**

So TWO INCOMPATIBLE RULES ARE CURRENTLY IN USE and both are shipped here, named, because
silently unifying them would invalidate work already done under each:

    CANONICAL   f13_movement_table.py. Tests risers against the RENORMALISATION NULL.
    DRAW        f13_draw_relation_items.py, and f13_code_amber_stages.py which imports
                its constants. NO NULL TEST AT ALL -- a riser is anything gaining >= DT.
                This is what feeds the annotation item draw, so it is what M01's
                clauses 5-6 rest on. Kept because it is load-bearing, NOT because it is
                right; new work should take CANONICAL and say so.

THE CANONICAL RULE, and the null is the whole point of it:

    faller  iff  P >= min_prob  AND  Q < fall_ratio * P
    R = 1 - sum_fallers Q       mass left over once the fallers have fallen
    S = sum_non-fallers P       pre-mass of everything that did not fall
    null = P * (R / S)          what each survivor gets from PURE RENORMALISATION
    riser   iff  not faller  AND  max(P,Q) > min_prob
                 AND  (Q - P) > delta        moved enough to matter
                 AND  Q > null               MORE than renormalisation explains

Without the last line a "riser" is any word that went up, and every word goes up a little
when a faller's mass is removed. The null is what separates redistribution from bookkeeping.

ASYMMETRY, DECLARED AND PRESERVED FROM THE ORIGINAL: risers are tested against the null;
FALLERS ARE NOT. A faller is a bare ratio rule. **Nothing downstream may describe fallers
as "beyond renormalisation"** -- they are not tested for it, and a word can halve purely
because mass left the system elsewhere.

THE true_word_probs CASE, AND ITS ONE HONEST COMPROMISE. The null needs total mass, and
`true_word_probs` is truncated at theta: the scored words sum to 1 - residual, with the
rest in one undifferentiated bucket. So R and S cannot be computed over the full
vocabulary. THE RESIDUAL IS CARRIED AS AN EXPLICIT NON-FALLER MASS rather than dropped or
renormalised away -- dropping it inflates every survivor's null, renormalising deletes the
mass that left the scored set entirely, which on this instrument is a quarter of the
distribution. `diagnostics["residual_share"]` reports how much of the distribution the
approximation rests on, and `diagnostics["exact_null"]` is False whenever it is used.

**A null computed over a truncated support is APPROXIMATE and says so.** Read
`residual_share` before quoting an excess: at 0.26 the bucket is larger than most words.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RESIDUAL_KEY = "__TAIL__"


@dataclass(frozen=True)
class Rule:
    """A named movement rule. Cite the name; do not re-type the numbers."""
    name: str
    min_prob: float          # a word must reach this in one arm to be eligible
    fall_ratio: float        # faller iff Q < fall_ratio * P
    delta: float             # a riser must gain at least this
    null_test: bool          # test risers against the renormalisation null?
    floor: float = 0.0       # DRAW only: a faller must start above this
    theta: float = 0.001     # the true_word_probs scoring threshold these assume


CANONICAL = Rule(name="canonical", min_prob=0.003, fall_ratio=0.5, delta=0.003,
                 null_test=True)

# DRAW's faller rule is delta-based, not ratio-based, so fall_ratio is unused and set to
# 1.0 (never binding). Recorded exactly as f13_draw_relation_items.py has it.
DRAW = Rule(name="draw", min_prob=0.0, fall_ratio=1.0, delta=0.003,
            null_test=False, floor=0.005)


@dataclass
class Movement:
    fallers: list = field(default_factory=list)
    risers: list = field(default_factory=list)
    null: dict = field(default_factory=dict)       # per-key renormalisation expectation
    excess: dict = field(default_factory=dict)     # Q - null, risers only
    delta: dict = field(default_factory=dict)      # Q - P, every key
    inflation: float = float("nan")                # R/S, the renormalisation factor
    rule: Rule = CANONICAL
    diagnostics: dict = field(default_factory=dict)

    def top_faller(self):
        return max(self.fallers, key=lambda w: self.delta.get(w, 0.0) * -1, default=None)

    def top_riser(self):
        """By EXCESS where the null was computed, else by delta. The distinction matters:
        ranking risers by delta re-introduces exactly what the null removes."""
        if not self.risers:
            return None
        key = self.excess if self.rule.null_test else self.delta
        return max(self.risers, key=lambda w: key.get(w, 0.0))


def _movement(P, Q, rule, residual_share, exact_null):
    keys = set(P) | set(Q)
    d = {k: Q.get(k, 0.0) - P.get(k, 0.0) for k in keys}

    if rule.null_test:
        fall = [k for k in keys if P.get(k, 0.0) >= rule.min_prob
                and Q.get(k, 0.0) < rule.fall_ratio * P.get(k, 0.0)]
    else:
        fall = [k for k in keys if P.get(k, 0.0) >= rule.floor and d[k] <= -rule.delta]
    fallset = set(fall)

    m = Movement(fallers=sorted(fall), delta=d, rule=rule)

    if not rule.null_test:
        m.risers = sorted(k for k in keys if d[k] >= rule.delta)
        m.diagnostics = {"rule": rule.name, "null_tested": False,
                         "residual_share": residual_share, "exact_null": None,
                         "n_fallers": len(fall), "n_risers": len(m.risers)}
        return m

    R = 1.0 - sum(Q.get(k, 0.0) for k in fallset)
    S = sum(P.get(k, 0.0) for k in keys if k not in fallset)
    if S <= 0:
        m.diagnostics = {"rule": rule.name, "refused": "no non-faller pre-mass",
                         "residual_share": residual_share, "exact_null": exact_null}
        return m
    infl = R / S
    m.inflation = infl
    m.null = {k: P.get(k, 0.0) * infl for k in keys if k not in fallset}
    rise = [k for k in keys if k not in fallset
            and max(P.get(k, 0.0), Q.get(k, 0.0)) > rule.min_prob
            and d[k] > rule.delta
            and Q.get(k, 0.0) > m.null.get(k, 0.0)]
    m.risers = sorted(rise)
    m.excess = {k: Q.get(k, 0.0) - m.null[k] for k in rise}
    m.diagnostics = {"rule": rule.name, "null_tested": True, "inflation": infl,
                     "residual_share": residual_share, "exact_null": exact_null,
                     "n_fallers": len(fall), "n_risers": len(rise)}
    return m


def movement(pre, post, rule=CANONICAL, residual_pre=None, residual_post=None):
    """Risers and fallers from two `true_word_probs` word->prob mappings.

    `residual_pre`/`residual_post` are the arms' untruncated remainders. Supply them:
    the null needs total mass and the scored words do not carry it. Omitted, they are
    read from a RESIDUAL_KEY entry if present, and if neither exists the null is computed
    over the scored set alone and `diagnostics["exact_null"]` is False with
    `residual_share` 0.0 -- which is a claim about the input, not a property of the data.
    """
    P = {k: v for k, v in pre.items() if k != RESIDUAL_KEY}
    Q = {k: v for k, v in post.items() if k != RESIDUAL_KEY}
    rp = residual_pre if residual_pre is not None else pre.get(RESIDUAL_KEY, 0.0)
    rq = residual_post if residual_post is not None else post.get(RESIDUAL_KEY, 0.0)
    if rp or rq:
        # The residual participates as one non-faller mass. It cannot be a faller: an
        # undifferentiated bucket has no word to fall, and treating it as one would let
        # tail movement masquerade as a lexical event.
        P, Q = {**P, RESIDUAL_KEY: rp}, {**Q, RESIDUAL_KEY: rq}
    share = max(rp, rq)
    m = _movement(P, Q, rule, share, exact_null=False)
    for coll in (m.null, m.excess, m.delta):
        coll.pop(RESIDUAL_KEY, None)
    m.fallers = [k for k in m.fallers if k != RESIDUAL_KEY]
    m.risers = [k for k in m.risers if k != RESIDUAL_KEY]
    return m


def movement_from_logits(pre_logits, post_logits, rule=CANONICAL, labels=None):
    """Risers and fallers from two full-vocabulary logit vectors.

    This is the EXACT case -- the support is the whole vocabulary, so R and S are exact
    and `diagnostics["exact_null"]` is True. Vocab-size mismatches truncate to the shared
    prefix and renormalise, as f13_movement_table.py does for tulu (128,256 vs 128,264).
    """
    a = [float(x) for x in pre_logits]
    b = [float(x) for x in post_logits]
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

    def sm(v):
        mx = max(v)
        e = [math.exp(x - mx) for x in v]
        s = sum(e)
        return [x / s for x in e]

    p, q = sm(a), sm(b)
    idx = labels if labels is not None else list(range(len(p)))
    P = {idx[i]: p[i] for i in range(len(p))}
    Q = {idx[i]: q[i] for i in range(len(q))}
    return _movement(P, Q, rule, residual_share=0.0, exact_null=True)


# ---------------------------------------------------------------------------
# Cache accessors. THE ONE-LINER EVERYONE WRITES IS WRONG.
# ---------------------------------------------------------------------------

@dataclass
class WordProbs:
    """A prompt's word distribution for one model, plus what it took to build it."""
    probs: dict                  # word -> probability, SUMMED over token paths
    residual: float              # untruncated remainder; probs + residual == 1
    rule_version: int = None     # the boundary rule that produced the cells
    n_rows: int = 0              # payload rows
    n_surfaces: int = 0          # distinct words
    collapsed: int = 0           # rows folded into an existing surface

    @property
    def total(self):
        return sum(self.probs.values()) + self.residual


def word_probs(model, prompt, theta=0.001, mode="raw", cache=None):
    """`{word: prob}` for one model and prompt, or None if the cell is not cached.

    **DO NOT WRITE `{r["word"]: r["p"] for r in payload["rows"]}`.** The payload is one
    row per (word, FIRST TOKEN) -- a surface reachable by several token paths gets
    several rows -- and those rows are a PARTITION: summed over every row, plus the
    residual, they come to 1.000000. A dict comprehension keeps the last path and DROPS
    THE REST.

    Measured on this cache: **20% of payloads contain a duplicated surface**, up to three
    rows for one word, and on a Chinese payload the naive comprehension lost 2.7% of the
    distribution (0.973 instead of 1.000). The error is silent, it is larger where a
    language has more token paths per surface, and it therefore falls hardest on exactly
    the cross-language comparison it would be used for.

    `collapsed` reports how many rows were folded, so a caller can see when it happened.
    """
    from .cache import get_cache
    cm = cache or get_cache()
    payload = cm.get_true_word_probs(model, prompt, theta=theta, mode=mode)
    if payload is None:
        return None
    rows = payload.get("rows") or []
    probs = {}
    for r in rows:
        w = r["word"]
        probs[w] = probs.get(w, 0.0) + r["p"]      # SUM, never overwrite
    return WordProbs(
        probs=probs,
        residual=(payload.get("residual") or {}).get("total", 0.0),
        rule_version=payload.get("rule_version"),
        n_rows=len(rows), n_surfaces=len(probs), collapsed=len(rows) - len(probs))


def movers(pre_model, post_model, prompt, rule=CANONICAL, theta=0.001, mode="raw",
           cache=None, allow_mixed_rule_version=False):
    """Risers and fallers between two models on one prompt, straight from the cache.

    Returns None if either cell is missing.

    **REFUSES A MIXED rule_version BY DEFAULT.** A v1 pre-arm against a v3 post-arm books
    an INSTRUMENT CHANGE as alignment movement: v3 changed what a word is, so words
    appear, merge and vanish between the arms for reasons that have nothing to do with
    the model. Pass `allow_mixed_rule_version=True` only with a reason, and never for a
    number that will be quoted.
    """
    a = word_probs(pre_model, prompt, theta, mode, cache)
    b = word_probs(post_model, prompt, theta, mode, cache)
    if a is None or b is None:
        return None
    if (a.rule_version != b.rule_version) and not allow_mixed_rule_version:
        raise ValueError(
            f"rule_version mismatch: {pre_model} is v{a.rule_version}, {post_model} is "
            f"v{b.rule_version}. The arms were produced by different instruments, so a "
            f"difference between them is not attributable to alignment. Re-run the "
            f"lagging arm, or pass allow_mixed_rule_version=True with a stated reason.")
    m = movement(a.probs, b.probs, rule,
                 residual_pre=a.residual, residual_post=b.residual)
    m.diagnostics.update(rule_version=a.rule_version, collapsed_pre=a.collapsed,
                         collapsed_post=b.collapsed, n_surfaces_pre=a.n_surfaces,
                         n_surfaces_post=b.n_surfaces)
    return m


# ---------------------------------------------------------------------------
# Decomposition. JS is a SUM over words, so it partitions by role exactly.
# ---------------------------------------------------------------------------

def js_terms(p, q):
    """Per-key JS contributions in bits. `sum(js_terms(p,q).values()) == JS(p,q)`.

    The whole point: a divergence that is a sum can be ATTRIBUTED. Plain JS answers
    "how much did this distribution move", which is not the question -- it conflates
    mass moving between identifiable words with mass draining into an unresolved tail,
    and those have opposite meanings for a displacement claim.
    """
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1.0, sum(q.values()) or 1.0
    out = {}
    for k in keys:
        a, b = p.get(k, 0.0) / sp, q.get(k, 0.0) / sq
        m = 0.5 * (a + b)
        if m <= 0:
            out[k] = 0.0
            continue
        t = 0.0
        if a > 0:
            t += 0.5 * a * math.log2(a / m)
        if b > 0:
            t += 0.5 * b * math.log2(b / m)
        out[k] = t
    return out


def decompose(pre, post, rule=CANONICAL, residual_pre=0.0, residual_post=0.0):
    """Split a step's divergence into the parts a displacement claim cares about.

    THE DIVERGENCE, BY ROLE. These four are EXACT and sum to `js_total`:

        js_fallers    contributed by words that FELL
        js_risers     contributed by words that ROSE BEYOND THE NULL
        js_tail       contributed by the residual bin -- movement into or out of the
                      UNRESOLVED mass, which is not a lexical event at all
        js_other      words that moved but too little to be either

    THE MASS. Note what `excess` is before reading these:

        **Excess is ZERO-SUM across the survivors.** sum_non-fallers null == R by
        construction, and sum_non-fallers Q == R too, so the excesses cancel to zero
        over the union support (verified: 1e-07 on a live cell). Excess is therefore a
        REDISTRIBUTION AMONG SURVIVORS laid on top of proportional renormalisation --
        not a share of anything the fallers gave up.

        departed      mass that left the fallers. The magnitude of the repression.
        arrived       positive excess on the flagged risers. The magnitude of the
                      SELECTIVE uptake.
        tail_excess   the residual bin's OWN excess, and the substitution-vs-deflection
                      quantity: POSITIVE means mass went into the unresolved tail beyond
                      what renormalisation hands it -- the step dispersed. NEGATIVE means
                      the tail gave mass up to nameable words -- the step substituted.
        selectivity   arrived / departed. **NOT a share of `departed`** -- the zero-sum
                      identity above means the two have no ordering relation and this
                      routinely exceeds 1. Read it as selective uptake per unit repressed:
                      near 0 is pure renormalisation, large is a step that promotes
                      particular words while it demotes others.
        captured      arrived / all positive excess. THIS one is a 0-1 share: how much of
                      the selective uptake landed on words the rule flags at all, rather
                      than dribbling across words below its thresholds.
        concentration top riser's share of `arrived`. Scale-free, so unlike JS it does not
                      shrink when a tokenizer resolves a language coarsely -- it asks how
                      the resolvable mass DISTRIBUTED itself, not how much there was.
        tail_share    js_tail / js_total. **The diagnostic that decides whether a
                      cross-language JS comparison means anything**: high here says the
                      divergence is dominated by mass the instrument cannot see inside,
                      and two languages with different tail_shares are not comparable on
                      plain JS however significant the difference looks.
    """
    m = movement(pre, post, rule, residual_pre=residual_pre, residual_post=residual_post)
    P = {**pre, RESIDUAL_KEY: residual_pre}
    Q = {**post, RESIDUAL_KEY: residual_post}
    terms = js_terms(P, Q)

    fall, rise = set(m.fallers), set(m.risers)
    js_f = sum(v for k, v in terms.items() if k in fall)
    js_r = sum(v for k, v in terms.items() if k in rise)
    js_t = terms.get(RESIDUAL_KEY, 0.0)
    total = sum(terms.values())

    # Excess over EVERY survivor, on the UNION support. Iterating P's keys alone would
    # skip post-only words, which carry excess = Q against a null of zero -- and dropping
    # them broke the zero-sum identity by 0.006 on the first cell tested.
    R = 1.0 - sum(Q.get(w, 0.0) for w in fall)
    S = sum(P.get(k, 0.0) for k in set(P) | set(Q) if k not in fall)
    ratio = (R / S) if S > 0 else 1.0
    exc_all = {k: Q.get(k, 0.0) - P.get(k, 0.0) * ratio
               for k in set(P) | set(Q) if k not in fall}
    pos_excess = sum(v for v in exc_all.values() if v > 0)

    departed = sum(-m.delta[w] for w in m.fallers) if m.fallers else 0.0
    arrived = sum(m.excess.values()) if m.excess else 0.0
    top = max(m.excess.values(), default=0.0)

    return {
        "js_total": total, "js_fallers": js_f, "js_risers": js_r, "js_tail": js_t,
        "js_other": total - js_f - js_r - js_t,
        "departed": departed, "arrived": arrived,
        "tail_excess": exc_all.get(RESIDUAL_KEY, 0.0),
        "selectivity": (arrived / departed) if departed > 0 else None,
        "captured": (arrived / pos_excess) if pos_excess > 0 else None,
        "concentration": (top / arrived) if arrived > 0 else None,
        "tail_share": (js_t / total) if total > 0 else None,
        "n_fallers": len(m.fallers), "n_risers": len(m.risers),
    }
