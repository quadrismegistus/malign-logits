"""One measurement: a step's two checkpoints, at one prompt.

    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint

    c = Step(Checkpoint("LLM360/Amber"), Checkpoint("LLM360/AmberChat")).cell(text)
    c.movement(CANONICAL).risers
    c.js(), c.l1()
    c.prompt.domain            # stratification, for free

WHY THE CELL IS THE UNIT. Risers and fallers are per (step, prompt) -- a step alone has
none, a prompt alone has none. Everything measured in this project lives here, which is
also why a graph is the wrong container for it: 100,837 of these is a table, not a
topology.

THREE THINGS IT REFUSES TO DO SILENTLY, all of them defects from 2026-07-30:

  MIXED `rule_version` RAISES. A v1 pre-arm against a v3 post-arm books an INSTRUMENT
  CHANGE as alignment movement -- v3 changed what a word is, so words appear, merge and
  vanish between arms for reasons that have nothing to do with the model. Pass
  `allow_mixed=True` with a reason; never for a number that will be quoted.

  THE PARTITION IS SUMMED, NOT OVERWRITTEN. `word_probs()` does the folding; a cell never
  rebuilds it. `{r["word"]: r["p"] for r in rows}` drops mass on 20% of payloads and up to
  99.9% on the smallest ones, which is the defect this project shipped in three separate
  consumers.

  `prompt` IS THE CATALOGUE ROW WHERE ONE EXISTS, None WHERE IT DOES NOT. The grid scores
  strings from the census as well as the catalogue, so not every cell has metadata. A
  None here is a real gap and analyses that stratify must decide what to do with it --
  which is better than a default domain quietly pooling unclassified prompts into
  "neutral".

STRATIFY BEFORE THE STATISTIC, NOT AFTER. `cell.prompt.domain` is one attribute away, and
that is deliberate: the amendment that made stratification a declared step cost a round
on the docket, and this is what makes it the path of least resistance instead of a rule
someone remembers.
"""
from __future__ import annotations

import math
from functools import cached_property


class Cell:
    """A (step, prompt) pair. Constructed by `Step.cell()`; rarely built directly."""

    def __init__(self, step, prompt, theta=0.001, mode="raw"):
        self.step = step
        self.prompt_text = prompt.text if hasattr(prompt, "text") else prompt
        self.theta = theta
        self.mode = mode

    #: REFUSE TO COMPUTE. NEVER REFUSE TO DESCRIBE.
    #:
    #: Once `rule_version` is a KEY field, a read that names no rule RAISES in
    #: a store holding two -- correctly, because a statistic pooled across
    #: boundary rules is the defect the key exists to prevent. But that raise
    #: propagated into `is_present` and `__repr__`, so in exactly the state
    #: worth inspecting the object became UNPRINTABLE, and `rule_version` --
    #: documented as returning "a tuple when they disagree" -- raised instead
    #: of reporting the disagreement. The mixed-rule reporter was unreachable
    #: in the mixed-rule case.
    #:
    #: The line is drawn by WHAT THE CALLER DOES WITH THE ANSWER, which is why
    #: it differs from `has_true_word_probs` (which keeps its raise): that one
    #: is asked in order to DECIDE -- the ingest writes or skips on it, so a
    #: wrong False changes what is stored. These are asked in order to LOOK.
    #:
    #:     A PREDICATE THAT GATES AN ACTION MUST REFUSE WHEN IT CANNOT ANSWER.
    #:     A PREDICATE THAT GATES A PRINT MUST ANSWER.

    AMBIGUOUS = "ambiguous"          #: sentinel: present, but at >1 rule

    def __repr__(self):
        t = (self.prompt_text or "")[:34]
        rv = self.rule_version
        extra = f", rules={rv}" if rv == self.AMBIGUOUS else ""
        return (f"Cell({self.step.label}, {t!r}, "
                f"present={self.is_present}{extra})")

    # -- the two distributions ----------------------------------------------
    def _arm(self, checkpoint_id):
        """One arm's word probabilities, or the AMBIGUOUS sentinel.

        Only an ambiguity is converted; every other failure propagates, because
        swallowing them is how a missing arm becomes a zero.
        """
        from .movement import word_probs
        try:
            return word_probs(checkpoint_id, self.prompt_text,
                              self.theta, self.mode)
        except KeyError as exc:
            if "AMBIGUOUS" in str(exc).upper() or "ambiguous" in str(exc):
                return self.AMBIGUOUS
            raise

    @cached_property
    def pre(self):
        return self._arm(self.step.pre.id)

    @cached_property
    def post(self):
        return self._arm(self.step.post.id)

    @property
    def is_present(self):
        """Both arms have this prompt scored. A cell can be asked for and not exist.

        ANSWERS, never raises. An ambiguous arm IS present -- the data is there,
        the read was underspecified -- so this is True and `rule_version`
        carries the reason. A caller that only handles None would otherwise be
        unable to tell "absent" from "ambiguous".
        """
        return (self.pre is not None and self.post is not None)

    @property
    def rule_version(self):
        """The instrument that produced both arms, a tuple when they disagree,
        or AMBIGUOUS when either arm is present at more than one rule."""
        if not self.is_present:
            return None
        if self.pre is self.AMBIGUOUS or self.post is self.AMBIGUOUS:
            return self.AMBIGUOUS
        a, b = self.pre.rule_version, self.post.rule_version
        return a if a == b else (a, b)

    # -- the prompt's own metadata ------------------------------------------
    @cached_property
    def prompt(self):
        """The catalogue row for this text, or None if the text is not catalogued.

        Uses the ranked pick, because 61 texts carry more than one row and choosing
        arbitrarily among them is what reported 48 group disagreements where there was 1.
        """
        from .prompts import Prompt
        try:
            return Prompt.find(self.prompt_text)
        except Exception:
            return None

    @property
    def domain(self):
        p = self.prompt
        return p.domain if p is not None else None

    @property
    def language(self):
        p = self.prompt
        return p.language if p is not None else None

    # -- measurements --------------------------------------------------------
    def movement(self, rule=None, allow_mixed=False):
        """Risers and fallers under a NAMED rule.

        The rule is an argument and has no default that hides it: `movement(CANONICAL)`
        says at the call site what `displacement_map()` never did.
        """
        from .movement import CANONICAL, movement as _movement
        rule = rule or CANONICAL
        if not self.is_present:
            return None
        self._check_versions(allow_mixed)
        m = _movement(self.pre.probs, self.post.probs, rule,
                      residual_pre=self.pre.residual, residual_post=self.post.residual)
        m.diagnostics.update(step=self.step.label, direction=self.step.direction,
                             prompt=self.prompt_text, rule_version=self.rule_version)
        return m

    def matched_control(self, rule=None, tau=0.005, tol=1.0, basis="post"):
        """(faller, matched non-mover) for this cell, or (faller, None), or None.

        The pairing Finding A's spec asked for and no corpus had: a word the
        aligned arm finds as improbable as the faller but did NOT demote. See
        `Movement.matched_nonmover` for why the match is on the POST arm.
        """
        m = self.movement(rule)
        if m is None:
            return None
        f = m.top_faller()
        if f is None:
            return None
        return f, m.matched_nonmover(f, tau=tau, tol=tol, basis=basis)

    def js(self, allow_mixed=False):
        """Jensen-Shannon divergence in bits, over the scored words plus the residual.

        THE RESIDUAL IS A BIN, NOT A RENORMALISATION. Dropping it would report a
        redistribution among survivors and hide the mass that left the scored set --
        about a quarter of the distribution on this instrument.
        """
        return self._divergence(_js, allow_mixed)

    def l1(self, allow_mixed=False):
        """Summed |dp|. Clause 8's booked metric, and NOT the same quantity as `js` --
        the two disagree by roughly 50% of a ratio, which is why a movement statistic
        that does not name its metric is not a number."""
        return self._divergence(_l1, allow_mixed)

    def decompose(self, rule=None, allow_mixed=False):
        """JS split by ROLE -- fallers, risers, tail, other -- plus the mass ratios.

        `js()` answers "how much did this move", which conflates two opposite events:
        mass passing between identifiable words, and mass draining into an unresolved
        tail. Because JS is a sum over words it partitions exactly, so the parts can be
        told apart. `tail_share` is the diagnostic that decides whether a cross-language
        JS comparison means anything at all. See `movement.decompose`.
        """
        from .movement import CANONICAL, decompose as _decompose
        if not self.is_present:
            return None
        self._check_versions(allow_mixed)
        return _decompose(self.pre.probs, self.post.probs, rule or CANONICAL,
                          residual_pre=self.pre.residual,
                          residual_post=self.post.residual)

    def _divergence(self, fn, allow_mixed):
        if not self.is_present:
            return None
        self._check_versions(allow_mixed)
        p = dict(self.pre.probs)
        q = dict(self.post.probs)
        p["__TAIL__"] = self.pre.residual
        q["__TAIL__"] = self.post.residual
        return fn(p, q)

    def _check_versions(self, allow_mixed):
        rv = self.rule_version
        #: AMBIGUOUS IS NOT "MIXED", AND `allow_mixed` MUST NOT COVER IT.
        #: Mixed means two rules I can name and have chosen to accept.
        #: Ambiguous means the store holds more than one and the read did not
        #: say which -- so there is no pair of distributions to accept. This
        #: is the single choke point every compute path passes through
        #: (`movement`, `js`, `l1`, `decompose`, `_divergence`), and without
        #: it the sentinel would reach `self.pre.probs` and die as an
        #: AttributeError on a string: a confusing failure in place of a
        #: refusal that names its cause.
        if rv == self.AMBIGUOUS:
            raise ValueError(
                f"refusing to compute on {self!r}: an arm is present at MORE "
                f"THAN ONE boundary rule and this read named none. Pass the "
                f"rule explicitly; `allow_mixed` does not cover ambiguity, "
                f"because there is no chosen pair to accept.")
        if isinstance(rv, tuple) and not allow_mixed:
            raise ValueError(
                f"rule_version mismatch on {self.prompt_text[:40]!r}: "
                f"{self.step.pre.id} is v{rv[0]}, {self.step.post.id} is v{rv[1]}. "
                f"The arms were produced by different instruments, so a difference "
                f"between them is not attributable to training. Re-run the lagging arm, "
                f"or pass allow_mixed=True with a stated reason.")

    def record(self, rule=None):
        """A flat dict for a dataframe row: the measurement WITH its stratification.

        Carries domain, language and rule_version alongside the numbers, so a table
        built from these can be stratified without a second join -- and so a pooled
        statistic that ignored the strata is visibly a choice.
        """
        m = self.movement(rule)
        return {
            "step": self.step.label, "direction": self.step.direction,
            "pre": self.step.pre.id, "post": self.step.post.id,
            "prompt": self.prompt_text, "domain": self.domain,
            "language": self.language, "rule_version": self.rule_version,
            "js": self.js(), "l1": self.l1(),
            "n_fallers": len(m.fallers) if m else None,
            "n_risers": len(m.risers) if m else None,
            "top_riser": m.top_riser() if m else None,
            "residual_pre": self.pre.residual if self.pre else None,
            "residual_post": self.post.residual if self.post else None,
        }


def _js(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1.0, sum(q.values()) or 1.0
    d = 0.0
    for k in keys:
        a, b = p.get(k, 0.0) / sp, q.get(k, 0.0) / sq
        m = 0.5 * (a + b)
        if m <= 0:
            continue
        if a > 0:
            d += 0.5 * a * math.log2(a / m)
        if b > 0:
            d += 0.5 * b * math.log2(b / m)
    return max(0.0, d)


def _l1(p, q):
    return sum(abs(q.get(k, 0.0) - p.get(k, 0.0)) for k in set(p) | set(q))
