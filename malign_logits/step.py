"""A step between two checkpoints: the thing a Cell is measured across.

    from malign_logits.step import Step
    from malign_logits.checkpoint import Checkpoint

    s = Step(Checkpoint("LLM360/Amber"), Checkpoint("LLM360/AmberChat"))
    s.label           # "base->sft"
    s.direction       # "forward"
    s.cell(text)      # one Cell
    s.cells           # every prompt both arms have scored

A STEP IS A RELATION, SO ITS IDENTITY IS THE PAIR. Not a family plus a label: a label
version would need a lookup table naming each edge, and such a table was dead on 62 of
103 models earlier today when the registry cache was stale. `Step(a, b)` needs nothing
but the two checkpoints and works for pairs nobody thought to name -- the net edge, a
cross-family comparison, a size contrast.

IT DOES NOT REFUSE A REVERSE PAIR. Teacher-forcing base->sft and then sft->base is real
work in this project, so a step that raised on reverse order would block the experiment.
**It stamps direction instead**: every Movement and every record carries `direction`, so
a pooled analysis cannot silently mix forward and reverse cells. Detectable rather than
forbidden, because forbidden is wrong here.

DIRECTION IS READ FROM STAGE ORDER, NOT FROM THE EDGE LIST. The registry's relations are
STAR-SHAPED from the base -- an aligned arm hangs off the base, not off its own SFT arm --
so `path()` cannot tell you that sft->dpo runs forwards. Stage order can: base precedes
sft precedes any preference method precedes rlvr. Where either stage is unknown the
direction is `"unknown"` and says so rather than assuming.
"""
from __future__ import annotations

from functools import cached_property

from .cell import Cell
from .checkpoint import Checkpoint

# Training order. Everything at one index is mutually incomparable -- the four preference
# methods are alternatives, not a sequence, so kto->dpo has no direction and claiming one
# would invent an ordering the training never had.
_ORDER = (
    ("base",),
    ("sft",),
    ("dpo", "kto", "ppo", "slic", "orpo", "simpo", "instruct", "aligned"),
    ("rlvr",),
)
_RANK = {s: i for i, group in enumerate(_ORDER) for s in group}


class Step:
    """An ordered pair of checkpoints. The unit a Cell is measured across."""

    def __init__(self, pre, post, theta=0.001, mode="raw"):
        self.pre = pre if isinstance(pre, Checkpoint) else Checkpoint(pre)
        self.post = post if isinstance(post, Checkpoint) else Checkpoint(post)
        self.theta = theta
        self.mode = mode

    def __repr__(self):
        return f"Step({self.label!r}, {self.direction}, {self.pre.id} -> {self.post.id})"

    def __eq__(self, other):
        return (isinstance(other, Step) and other.pre == self.pre
                and other.post == self.post)

    def __hash__(self):
        return hash((self.pre.id, self.post.id))

    # -- what this step is ---------------------------------------------------
    @property
    def label(self):
        """Derived from the endpoints' stages, never declared: 'base->sft', 'sft->kto'."""
        a = getattr(self.pre, "stage", None) if self.pre.is_known else None
        b = getattr(self.post, "stage", None) if self.post.is_known else None
        return f"{a or '?'}->{b or '?'}"

    @cached_property
    def direction(self):
        """'forward', 'reverse', 'lateral' or 'unknown'.

        `lateral` is two checkpoints at the same point in training -- kto against dpo,
        or one family's base against another's. Those are real comparisons and they are
        not steps through training, which is a distinction a bare forward/reverse flag
        would erase.
        """
        a = getattr(self.pre, "stage", None) if self.pre.is_known else None
        b = getattr(self.post, "stage", None) if self.post.is_known else None
        ra, rb = _RANK.get(a), _RANK.get(b)
        if ra is None or rb is None:
            return "unknown"
        if ra < rb:
            return "forward"
        if ra > rb:
            return "reverse"
        return "lateral"

    @property
    def is_forward(self):
        return self.direction == "forward"

    @property
    def family(self):
        """The family both arms belong to, or None if they differ or are unknown."""
        a, b = self.pre.family, self.post.family
        return a if a and a == b else None

    @property
    def is_runnable(self):
        """Both arms are ACTIVE in the registry — no load-failure hole."""
        return (self.pre.is_known and self.post.is_known
                and not self.pre.is_excluded and not self.post.is_excluded)

    # -- cells ---------------------------------------------------------------
    def cell(self, prompt):
        """One Cell. `prompt` may be text or a `Prompt`."""
        return Cell(self, prompt, theta=self.theta, mode=self.mode)

    @cached_property
    def prompts(self):
        """Texts BOTH arms have scored, sorted. The intersection, never one arm's list.

        Taking either arm alone silently measures a different population — which is how a
        producer iterating a registry instead of the store dropped 65% of amber's cells.
        """
        return sorted(_scored(self.pre.id) & _scored(self.post.id))

    @property
    def cells(self):
        """A Cell per shared prompt. Lazy: nothing is read until a cell is measured."""
        return [self.cell(p) for p in self.prompts]

    def records(self, rule=None, domain=None):
        """Flat dicts for a dataframe, one per cell, with stratification attached.

            pd.DataFrame(step.records())          # every shared prompt
            pd.DataFrame(step.records(domain="violence"))
        """
        out = []
        for c in self.cells:
            if domain is not None and c.domain != domain:
                continue
            if not c.is_present:
                continue
            out.append(c.record(rule))
        return out

    # -- constructors --------------------------------------------------------
    @staticmethod
    def of(family, pre="base", post="ego"):
        """A step between two POSITIONS of one family.

            Step.of("olmo")                       # base -> ego
            Step.of("amber", "ego", "superego")   # the preference step
        """
        from .family import Family
        f = family if isinstance(family, Family) else Family(family)
        a, b = f[pre], f[post]
        if a is None or b is None:
            return None
        return Step(a, b)

    @staticmethod
    def chain(family):
        """The declared training sequence as consecutive steps, skipping absent arms.

        base -> ego -> superego -> reinforced_superego. **This is the sequence the
        RELATIONS cannot give you** — they are star-shaped from the base, so the SFT
        arm is invisible to a traversal even though it is a declared node.
        """
        from .family import Family
        f = family if isinstance(family, Family) else Family(family)
        arms = [f[p] for p in ("base", "ego", "superego", "reinforced_superego")]
        arms = [a for a in arms if a is not None]
        return [Step(arms[i], arms[i + 1]) for i in range(len(arms) - 1)]


def _scored(model_id):
    """Prompts this model has in the true_word_probs store."""
    from .cache import get_cache
    cm = get_cache()
    return cm.distinct("true_word_probs", "prompt", model=model_id)
