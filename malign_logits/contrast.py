"""Two labelled samples, and the right test. The comparison frames, in one place.

    from malign_logits.contrast import by_field, by_role, by_translation, by_step, sweep

    by_role(step, "MARKED", "UNMARKED")               # minimal pairs, PAIRED
    by_field(step, "worker", "mgmt", field="subdomain")   # labor vs management, STRATIFIED
    by_translation(step, metric="departed")           # en vs zh, PAIRED
    by_step(s1, s2, metric="js")                      # socialization vs legislation, PAIRED
    sweep(step, "domain", baseline="neutral")         # every domain against neutral

Every frame returns a `Contrast`: two labelled samples plus a declared `paired` flag.
The class does one job -- hold the samples, run the test that matches, and refuse to
report a rate without its population.

WHY A CLASS AND NOT MORE SCRIPTS. Four scripts in this repo had each grown their own
copy of "measure both arms, drop the incomplete ones, sign-test the difference", and the
copies had drifted: one dropped incomplete units silently, one used the mean where the
distribution is skewed 3:1, one pooled strata the analysis was supposed to separate.
The frame is the part worth naming; the plumbing is not.

PAIRED IS DECLARED, NEVER INFERRED. `worker` and `mgmt` carry NO group_id -- the F21
position prompts are written independently, not as minimal pairs -- while MARKED and
UNMARKED share one. **The same conceptual comparison is paired in one place and
stratified in another**, so a frame that sniffed the data and picked a test would be
making the analytic choice this layer exists to keep visible. Paired uses a sign test on
within-unit differences; stratified uses Mann-Whitney on two independent samples. They
answer different questions and `Contrast.test()` names which one it ran.

THE POPULATION IS PART OF THE RESULT. `.n` is what was actually compared and `.dropped`
says why the rest went, always as a Counter and never silently. This matters more than
it sounds: `selectivity` divides by `departed`, which vanishes on late-chain steps, so
on archangel it produced a p-value resting on SEVEN of 601 cells sitting beside columns
computed over all 601. `.table()` prints n on every line.

A SMALL SAMPLE IS A REAL ANSWER, NOT AN ERROR. `worker` vs `mgmt` is 7 against 7 in
English. The frame runs, reports n=7, and lets the test say what 7 can support. Refusing
would hide a designed comparison; pretending it is well powered would be worse.
"""
from __future__ import annotations

import collections
import math
import statistics as st

#: Metric names a frame will accept. `js` and `l1` come off the Cell directly; everything
#: else is a key of `movement.decompose`. Naming them here means a typo is an error at the
#: call site rather than a column of silent Nones.
METRICS = ("js", "l1", "js_total", "js_fallers", "js_risers", "js_tail", "tail_share",
           "departed", "arrived", "tail_excess", "selectivity", "captured",
           "concentration", "n_fallers", "n_risers")


def _measure(cell, metric, rule):
    """One number from one cell, or None where the cell cannot supply it.

    Returns None rather than raising on a mixed rule_version: the CALLER counts it as a
    drop with a reason, which keeps one bad arm from killing a whole frame while still
    refusing to mix instruments inside a number.
    """
    if not cell.is_present:
        return None
    try:
        if metric == "js":
            return cell.js()
        if metric == "l1":
            return cell.l1()
        d = cell.decompose(rule)
        return None if d is None else d.get(metric)
    except ValueError:
        return None          # mixed rule_version, counted by the frame


def _population(language, where):
    """The prompts a frame runs over. `where` is any catalogue field, e.g. finding="F36".

    WITHOUT THIS A FRAME SILENTLY POOLS DESIGNS. `MARKED`/`UNMARKED` is a role used by
    THREE findings -- 133 and 129 rows across the catalogue, of which F36 owns 69 and 68.
    Asking for "the marked pairs" and getting all of them mixes a transgressive swap with
    a gender swap with a register swap, which are different manipulations answering
    different questions. The pooled number would look fine and mean nothing.
    """
    from .prompts import Prompts
    kw = dict(where or {})
    if language:
        kw["language"] = language
    return Prompts.where(**kw) if kw else Prompts.all()


def _check(metric):
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}. Known: {', '.join(METRICS)}")


# ---------------------------------------------------------------------------
# The tests. Both exact enough to need no dependency, both returning n.
# ---------------------------------------------------------------------------

def sign_test(diffs):
    """(k positive, n non-zero, two-sided p). Exact binomial.

    Zero differences are DROPPED, not counted as half. That is the standard treatment and
    it is why n here is smaller than the number of pairs: on a late-chain step most cells
    have no fallers in either arm, so most differences are exactly zero and the honest n
    collapses. Reporting the pair count instead would inflate the population tenfold.
    """
    nz = [d for d in diffs if d != 0]
    n, k = len(nz), sum(1 for d in nz if d > 0)
    if n == 0:
        return 0, 0, None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return k, n, min(1.0, 2 * tail / (2 ** n))


def rank_sum(a, b):
    """Mann-Whitney U with tie correction, normal approximation. (U, z, two-sided p).

    Ties are corrected because these metrics tie constantly -- `departed` is exactly 0.0
    on most late-chain cells -- and the uncorrected variance would understate p on
    precisely the comparisons where the data is thinnest.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, None
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(allv)
    i = ties = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        t = j - i + 1
        if t > 1:
            ties += t ** 3 - t
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    U = r1 - n1 * (n1 + 1) / 2
    N = n1 + n2
    var = n1 * n2 * (N ** 3 - N - ties) / (12 * N * (N - 1)) if N > 1 else 0.0
    if var <= 0:
        return U, 0.0, None
    z = (U - n1 * n2 / 2) / math.sqrt(var)
    return U, z, math.erfc(abs(z) / math.sqrt(2))


class Contrast:
    """Two labelled samples of one metric, and whether they are paired.

    Built by the frame functions below rather than directly. `a` and `b` are aligned
    element-for-element when `paired` is True, and independent otherwise.
    """

    def __init__(self, label_a, a, label_b, b, *, paired, metric, step=None,
                 frame=None, dropped=None, units=None, tag=None):
        self.label_a, self.label_b = label_a, label_b
        self.a, self.b = list(a), list(b)
        self.paired = paired
        self.metric = metric
        self.step = step
        self.frame = frame
        self.dropped = collections.Counter(dropped or {})
        self.units = units or []      # identifiers, so a result can be traced back
        #: What the row is CALLED in a table. Needed because several frames produce rows
        #: whose a/b labels are identical -- a `by_step` sweep over domains gives three
        #: rows all reading "base->sft vs sft->dpo", and without a tag the stratum that
        #: distinguishes them is invisible in the output.
        self.tag = tag

    def __repr__(self):
        return (f"Contrast({self.frame} {self.label_a} vs {self.label_b}, "
                f"metric={self.metric}, {'paired' if self.paired else 'stratified'}, "
                f"n={self.n})")

    @property
    def n(self):
        """Units compared. For a paired frame that is PAIRS, not prompts; for a
        stratified frame it is the two samples summed, and `.n_display` shows them
        SEPARATELY -- a single figure would hide that `institutional vs neutral` is 55
        against 135, which is the asymmetry a reader needs to judge the test."""
        return len(self.a) if self.paired else len(self.a) + len(self.b)

    @property
    def n_display(self):
        return f"{len(self.a)}" if self.paired else f"{len(self.a)}/{len(self.b)}"

    @property
    def diffs(self):
        if not self.paired:
            raise ValueError("within-unit differences require a paired frame; this is "
                             f"{self.frame!r}, which compares independent samples")
        return [x - y for x, y in zip(self.a, self.b)]

    def test(self):
        """The test the frame's own structure calls for, named in the result.

        Paired -> sign test on within-unit differences. Stratified -> Mann-Whitney on two
        independent samples. `n` here is the TEST's population and can be far below the
        unit count, because the sign test drops exact ties.
        """
        if not self.a or not self.b:
            return {"test": None, "n": 0, "p": None,
                    "note": "a sample is empty; nothing was compared"}
        if self.paired:
            k, n, p = sign_test(self.diffs)
            return {"test": "sign", "n": n, "k": k, "p": p,
                    "direction": None if not n else
                                 (f"{self.label_a} > {self.label_b}" if k > n / 2
                                  else f"{self.label_b} > {self.label_a}")}
        U, z, p = rank_sum(self.a, self.b)
        return {"test": "rank_sum", "n": self.n, "n_a": len(self.a), "n_b": len(self.b),
                "U": U, "z": z, "p": p,
                "direction": (f"{self.label_a} > {self.label_b}" if z > 0
                              else f"{self.label_b} > {self.label_a}")}

    def summary(self):
        """Medians AND means, because these distributions are skewed roughly 3:1 and a
        single central figure hides the shape."""
        out = {"frame": self.frame, "metric": self.metric,
               "paired": self.paired, "n": self.n,
               "a": self.label_a, "b": self.label_b,
               "median_a": st.median(self.a) if self.a else None,
               "median_b": st.median(self.b) if self.b else None,
               "mean_a": st.mean(self.a) if self.a else None,
               "mean_b": st.mean(self.b) if self.b else None,
               "n_a": len(self.a), "n_b": len(self.b),
               "dropped": dict(self.dropped)}
        if self.paired and self.a:
            out["median_diff"] = st.median(self.diffs)
        out.update({f"test_{k}": v for k, v in self.test().items()})
        return out

    def line(self):
        """One row. `med diff` is a DASH on a stratified frame rather than a number:
        there are no within-unit differences to take a median of, and printing nan there
        invites a reader to treat a missing quantity as a failed computation."""
        s = self.summary()
        p = s.get("test_p")
        md = s.get("median_diff")
        tag = self.tag or f"{str(self.label_a)[:14]} vs {str(self.label_b)[:14]}"
        return (f"  {tag[:31]:<32}{self.n_display:>9}"
                f"{(s['median_a'] if s['median_a'] is not None else 0):>10.4f}"
                f"{(s['median_b'] if s['median_b'] is not None else 0):>10.4f}"
                f"{('-' if md is None else f'{md:.4f}'):>10}"
                f"{s.get('test_n', 0):>7}{('-' if p is None else f'{p:.2g}'):>11}"
                f"  {s.get('test_direction') or ''}")

    @staticmethod
    def header(metric, frame):
        return (f"  frame={frame}  metric={metric}\n"
                f"  {'contrast':<32}{'units':>9}{'med a':>10}{'med b':>10}"
                f"{'med diff':>10}{'test n':>7}{'p':>11}")

    def table(self):
        print(Contrast.header(self.metric, self.frame))
        print(self.line())
        if self.dropped:
            print(f"  dropped: {dict(self.dropped)}")


# ---------------------------------------------------------------------------
# The frames. Each one answers "which cells are comparable, and how".
# ---------------------------------------------------------------------------

def by_role(step, a, b, key="group_id", role="group_role", metric="js", rule=None,
            language="en", tag=None, where=None):
    """PAIRED within a catalogue group: MARKED vs UNMARKED, POLE_A vs POLE_B.

    A unit is a group holding EXACTLY ONE of each role. Groups holding two of a role, or
    only one role, are dropped and counted -- a frame that quietly kept the first match
    would compare a different population than it names.
    """
    _check(metric)
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in _population(language, where):
        k, r = p.row.get(key), p.row.get(role)
        if k and r in (a, b):
            buckets[k][r].append(p)

    A, B, units, dropped = [], [], [], collections.Counter()
    for k, roles in buckets.items():
        if len(roles.get(a, [])) != 1 or len(roles.get(b, [])) != 1:
            dropped["group is not a clean one-of-each pair"] += 1
            continue
        va = _measure(step.cell(roles[a][0].text), metric, rule)
        vb = _measure(step.cell(roles[b][0].text), metric, rule)
        if va is None or vb is None:
            dropped["a cell is absent, or the metric is undefined there"] += 1
            continue
        A.append(va); B.append(vb); units.append(k)
    return Contrast(a, A, b, B, paired=True, metric=metric, step=step,
                    frame=f"role:{role}", dropped=dropped, units=units,
                    tag=tag or f"{a} vs {b}")


def by_field(step, a, b, field="domain", metric="js", rule=None, language="en", tag=None,
             where=None):
    """STRATIFIED on any catalogue field: worker vs mgmt, institutional vs neutral.

    Two INDEPENDENT samples. Use this where no grouping key pairs the prompts -- which is
    the case for the whole F21 position axis (worker, mgmt, tenant, landlord, patient,
    doctor all carry group_id None), and would be the wrong choice for MARKED/UNMARKED,
    which are paired and lose power if compared this way.
    """
    _check(metric)
    A, B, units, dropped = [], [], [], collections.Counter()
    for p in _population(language, where):
        v = p.row.get(field)
        if v not in (a, b):
            continue
        m = _measure(step.cell(p.text), metric, rule)
        if m is None:
            dropped["cell absent, or the metric is undefined there"] += 1
            continue
        (A if v == a else B).append(m)
        units.append(p.id)
    return Contrast(a, A, b, B, paired=False, metric=metric, step=step,
                    frame=f"field:{field}", dropped=dropped, units=units,
                    tag=tag or f"{a} vs {b}")


def by_translation(step, metric="js", rule=None, tag=None):
    """PAIRED across languages: the same design in English and Chinese.

    **This frame does not control what it looks like it controls.** The manipulation is
    held fixed and the language varies, but so does the tokenizer's grip on that language,
    and the two cannot be separated by any statistic computed here -- `js_total` gives
    opposite significant answers on amber and yi. Treat a result as a diagnostic and read
    `tail_share` beside it. See scripts/decompose_steps.py for the control that killed the
    one metric that looked like it survived.
    """
    _check(metric)
    from .prompts import Prompts
    A, B, units, dropped = [], [], [], collections.Counter()
    for p in Prompts.where(language="en"):
        z = p.translation
        if z is None:
            continue
        va = _measure(step.cell(p.text), metric, rule)
        vb = _measure(step.cell(z.text), metric, rule)
        if va is None or vb is None:
            dropped["an arm is absent, or the metric is undefined there"] += 1
            continue
        A.append(va); B.append(vb); units.append(p.id)
    return Contrast("en", A, "zh", B, paired=True, metric=metric, step=step,
                    frame="translation", dropped=dropped, units=units,
                    tag=tag or "en vs zh")


def by_step(step_a, step_b, metric="js", rule=None, language="en", texts=None, tag=None):
    """PAIRED on the prompt, across two STEPS: socialization vs legislation.

    The one frame that varies the step rather than the prompt, so the tokenizer and the
    prompt set are both held constant. Restrict `texts` to a stratum to ask whether a
    step's extra movement is specific to that stratum -- which is how amber's safety-DPO
    was separated from ordinary step-size decay.
    """
    _check(metric)
    from .prompts import Prompts
    if texts is None:
        texts = [p.text for p in Prompts.where(language=language)]
    A, B, units, dropped = [], [], [], collections.Counter()
    for t in texts:
        va = _measure(step_a.cell(t), metric, rule)
        vb = _measure(step_b.cell(t), metric, rule)
        if va is None or vb is None:
            dropped["a step is missing this prompt, or the metric is undefined"] += 1
            continue
        A.append(va); B.append(vb); units.append(t)
    return Contrast(step_a.label, A, step_b.label, B, paired=True, metric=metric,
                    step=step_a, frame="step", dropped=dropped, units=units,
                    tag=tag or f"{step_a.label} vs {step_b.label}")


def sweep(step, field, baseline, metric="js", rule=None, language="en", min_n=25,
          where=None):
    """Every level of `field` against one baseline level. Returns a list of Contrasts.

    `min_n` drops levels too small to say anything, and the count of dropped levels is
    on each returned Contrast rather than swallowed -- `domain` has 14 levels and four of
    them hold fewer than 15 prompts.
    """
    _check(metric)
    levels = collections.Counter(p.row.get(field) for p in _population(language, where))
    out = []
    for lvl, n in levels.most_common():
        if lvl is None or lvl == baseline or n < min_n:
            continue
        out.append(by_field(step, lvl, baseline, field=field, metric=metric,
                            rule=rule, language=language, where=where,
                            tag=f"{lvl} vs {baseline}"))
    return out


def table(contrasts, title=""):
    """Print several Contrasts under one header. They must share a metric and frame."""
    if not contrasts:
        print("  no contrasts")
        return
    c0 = contrasts[0]
    if title:
        print(f"\n{title}")
    print(Contrast.header(c0.metric, c0.frame))
    for c in sorted(contrasts, key=lambda c: (c.test().get("p") is None,
                                              c.test().get("p") or 1.0)):
        print(c.line())
