"""A model family: its checkpoints, reachable by stage or by position.

    from malign_logits.family import Family

    f = Family("olmo")
    f.base, f.sft, f.dpo, f.rlvr        # by STAGE  -- what training produced it
    f["superego"]                        # by POSITION -- the taxonomy's slot
    f.checkpoints, f.is_three_arm
    Family.all(is_three_arm=True)

TWO VOCABULARIES, BOTH KEPT, BECAUSE THEY ANSWER DIFFERENT QUESTIONS.

  POSITION is the taxonomy's slot: base / ego / superego / reinforced_superego. It is
  complete for every family and it is what `MODEL_FAMILIES` declares. **It cannot
  distinguish preference methods** -- all four archangel arms are `superego`.

  STAGE is what training produced the checkpoint: base / sft / dpo / kto / ppo / slic /
  rlvr / instruct. It separates the archangel four, which is the whole reason that cell
  can hold SFT constant and vary method. It is also the field that carries `instruct`
  for one-step families, where naming the artifact is honest and naming the procedure
  would be a guess.

So `f.dpo` and `f["superego"]` are usually the same checkpoint and NOT ALWAYS: for
`archangel-kto` the superego is a KTO checkpoint and `f.dpo` is None. **Asking by stage
when you mean method, and by position when you mean slot, is the point of having both.**

WHAT THIS DELIBERATELY DOES NOT DO. It does not construct comparisons. `Step` takes two
checkpoints and is built separately, because a step is a relation between checkpoints
and not a property of a family -- and because the artifact's edges are star-shaped from
the base, so a family cannot hand you the training chain even if it wanted to.
"""
from __future__ import annotations

from functools import cached_property, lru_cache

from .checkpoint import Checkpoint

POSITIONS = ("base", "ego", "superego", "reinforced_superego", "reasoning")


@lru_cache(maxsize=1)
def _families():
    from . import MODEL_FAMILIES
    return MODEL_FAMILIES


def refresh():
    _families.cache_clear()


class Family:
    """One entry of `MODEL_FAMILIES`, with its checkpoints resolved."""

    def __init__(self, key):
        if key not in _families():
            raise KeyError(
                f"no family {key!r}. Known: {', '.join(sorted(_families()))}")
        self.key = key
        self._fam = _families()[key]

    def __repr__(self):
        return (f"Family({self.key!r}, {self.name!r}, "
                f"{len(self.checkpoints)} checkpoints)")

    def __eq__(self, other):
        return isinstance(other, Family) and other.key == self.key

    def __hash__(self):
        return hash(self.key)

    def __iter__(self):
        return iter(self.checkpoints)

    def __len__(self):
        return len(self.checkpoints)

    @property
    def name(self):
        """The human name, e.g. 'OLMo 3 7B'."""
        return getattr(self._fam, "name", self.key)

    # -- by position ---------------------------------------------------------
    def __getitem__(self, position):
        """f["superego"] -> Checkpoint, or None if the family has no such arm."""
        if position not in POSITIONS:
            raise KeyError(f"{position!r} is not a position. One of: {POSITIONS}")
        m = getattr(self._fam, position, None)
        return Checkpoint(m) if m else None

    @property
    def base(self):
        return self["base"]

    @property
    def ego(self):
        return self["ego"]

    @property
    def superego(self):
        return self["superego"]

    @property
    def reinforced_superego(self):
        return self["reinforced_superego"]

    @property
    def reasoning(self):
        return self["reasoning"]

    @cached_property
    def positions(self):
        """{position: Checkpoint} for the arms this family actually has."""
        return {p: self[p] for p in POSITIONS if self[p] is not None}

    @cached_property
    def checkpoints(self):
        return list(self.positions.values())

    # -- by stage -------------------------------------------------------------
    @cached_property
    def stages(self):
        """{stage: Checkpoint}. A stage the family lacks is simply absent.

        Read from the registry, so `archangel-kto` reports `kto` where `position` can
        only say `superego`.
        """
        out = {}
        for cp in self.checkpoints:
            s = getattr(cp, "stage", None) if cp.is_known else None
            if s:
                out.setdefault(s, cp)
        return out

    def stage(self, name):
        """The checkpoint produced by a named training stage, or None."""
        return self.stages.get(name)

    @property
    def sft(self):
        return self.stages.get("sft")

    @property
    def dpo(self):
        return self.stages.get("dpo")

    @property
    def rlvr(self):
        return self.stages.get("rlvr")

    @property
    def instruct(self):
        return self.stages.get("instruct")

    @property
    def preference(self):
        """The preference-optimised arm whatever its method — dpo, kto, ppo or slic.

        The four archangel families differ ONLY here, so a caller comparing across them
        wants this rather than `.dpo`, which is None for three of the four.
        """
        for m in ("dpo", "kto", "ppo", "slic", "orpo", "simpo"):
            if m in self.stages:
                return self.stages[m]
        return None

    # -- shape ----------------------------------------------------------------
    @property
    def is_three_arm(self):
        """Has base, an SFT arm and a preference arm — the shape a staged
        decomposition needs. 22 families qualify; 5 lose it to load failures."""
        return bool(self.base and self.ego and self.superego)

    @property
    def is_complete(self):
        """Every declared arm is ACTIVE in the registry — no load-failure holes."""
        return all(cp.is_known and not cp.is_excluded for cp in self.checkpoints)

    @property
    def excluded(self):
        """Arms the registry marks EXCLUDED, with their reasons."""
        return [(cp, cp.exclusion) for cp in self.checkpoints if cp.is_known
                and cp.is_excluded]

    @property
    def landed(self):
        return [cp for cp in self.checkpoints if cp.landed]

    # -- collection ------------------------------------------------------------
    @staticmethod
    def all(**fields):
        """Every family, filtered on any Family property.

            Family.all(is_three_arm=True)
            Family.all(is_complete=True, is_three_arm=True)
        """
        out = []
        for key in sorted(_families()):
            f = Family(key)
            if any(getattr(f, k, None) != v for k, v in fields.items()):
                continue
            out.append(f)
        return out

    @staticmethod
    def of(model_id):
        """The family a checkpoint belongs to, or None.

        A base shared by several families (pythia-2.8b bases all four archangels)
        returns None rather than picking one — use `Checkpoint.siblings` there.
        """
        cp = model_id if isinstance(model_id, Checkpoint) else Checkpoint(model_id)
        hits = [k for k in _families() if cp.id in
                {getattr(_families()[k], p, None) for p in POSITIONS}]
        return Family(hits[0]) if len(hits) == 1 else None
