"""A model checkpoint: everything you need to know before touching it.

    from malign_logits.checkpoint import Checkpoint

    cp = Checkpoint("ContextualAI/archangel_sft-kto_pythia2-8b")
    cp.stage           # "kto"      -- the preference METHOD, not just "aligned"
    cp.position        # "superego" -- the taxonomy's slot
    cp.architecture, cp.cjk_tier, cp.vocab_size, cp.weights_format
    cp.in_spec, cp.landed, cp.is_excluded
    cp.base, cp.children, cp.siblings

A THIN FACADE OVER `Registry`, WHICH IS THE STORE. `data/model_registry.json` is the
canonical artifact -- 112 models, 155 typed edges, `_schema` and `_provenance` -- and
`Registry` already loads it and traverses it (`parent_of`, `children_of`, `path`,
`variants_of`, `base_of`, `family_key`). **This class adds no facts and re-implements no
traversal.** It exists so that answering "what is this checkpoint" is one object instead
of four lookups, which is what it was earlier today when doing it by hand produced a
wrong architectural claim about the roster.

WRITTEN TWICE, AND THE FIRST VERSION IS INSTRUCTIVE. Before the canonical file existed
this class carried two derivations to paper over gaps: it bypassed a stale cache by
forcing a rebuild from `MODEL_FAMILIES`, and it read the preference method off the model
id because `stage` collapsed all four archangel arms to "dpo". **Both are gone.** The
artifact does both properly now, and a derivation kept "just in case" is a second opinion
that will disagree with the file the moment either changes.

ONE THING IT STILL WILL NOT DO: guess. `architecture`, `stage`, `country` and the rest
return whatever the file says, including None. A field the artifact does not carry is a
gap to be filled at the source, never a default supplied here -- the majority-class
fallback ("everything unmatched is a transformer") is exactly the move that mis-described
a roster holding two pure-SSM families.
"""
from __future__ import annotations

import glob
import json
import os
from functools import cached_property, lru_cache


@lru_cache(maxsize=1)
def _registry():
    from .registry import Registry
    return Registry()


@lru_cache(maxsize=1)
def _artifact(name):
    """A committed data artifact as parsed JSON, or {} if absent.

    Absent is NOT an error here: a machine without the file should get an empty
    map and a caller that can tell empty from zero, rather than an exception on
    import. **The distinction lives in the accessors**, which return None only
    for a missing row.
    """
    from . import PATH_DATA
    fp = os.path.join(PATH_DATA, name)
    if not os.path.exists(fp):
        return {}
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _tokenizer_props():
    """{model_id: row} from data/tokenizer_properties.json. 150 models."""
    d = _artifact("tokenizer_properties.json")
    rows = d.get("models") if isinstance(d, dict) else d
    if isinstance(rows, dict):
        return {k: v for k, v in rows.items() if isinstance(v, dict)}
    return {r["model"]: r for r in (rows or []) if isinstance(r, dict) and "model" in r}



@lru_cache(maxsize=1)
def _edge_overlap():
    """{(base, aligned): row} from data/edge_token_overlap.json. 198 edges."""
    d = _artifact("edge_token_overlap.json")
    rows = d.get("edges") if isinstance(d, dict) else d
    out = {}
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        #: THE FIELDS ARE `parent` / `child`, read from the artifact rather than
        #: guessed. The first version tried base/aligned/a/b/from/to -- five
        #: names, none of them the two that exist.
        a = r.get("parent")
        b = r.get("child")
        if a and b:
            out[(a, b)] = r
    return out


def _landed():
    """Model ids with cells on disk. Not the same as being scheduled."""
    from . import PATH_DATA
    d = os.path.join(PATH_DATA, "twp_grid_v3")
    return frozenset(os.path.basename(f)[:-6].replace("__", "/")
                     for f in glob.glob(os.path.join(d, "*.jsonl")))


def refresh():
    """Drop cached reads. Call after the grid writes or the registry is rebuilt."""
    _registry.cache_clear()
    _landed.cache_clear()


class Checkpoint:
    """One model. Identity is its HuggingFace id."""

    def __init__(self, model_id):
        self.id = model_id

    def __repr__(self):
        return (f"Checkpoint({self.id!r}, family={self.family!r}, "
                f"position={self.position!r}, stage={self.stage!r})")

    def __eq__(self, other):
        return isinstance(other, Checkpoint) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

    # -- the record ---------------------------------------------------------
    @cached_property
    def _info(self):
        return _registry().info(self.id)

    @property
    def is_known(self):
        return self._info is not None

    def __getattr__(self, name):
        """Any registry field as an attribute: cp.architecture, cp.vocab_size, cp.country.

        `Registry` attaches the artifact's extra columns to `ModelInfo` rather than
        widening the dataclass, so the field set here follows the FILE and not a schema
        frozen in Python. An unknown name raises rather than returning None, so a typo is
        a mistake instead of a silent null.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        info = self.__dict__.get("_info") or _registry().info(self.__dict__.get("id"))
        if info is not None and hasattr(info, name):
            return getattr(info, name)
        raise AttributeError(
            f"{name!r} is not a field of the model registry for {self.id!r}. "
            f"Fields: {', '.join(sorted(vars(info))) if info else '(model not in registry)'}")

    @property
    def record(self):
        """The registry row as a plain dict. Read-only by convention."""
        return dict(vars(self._info)) if self._info else {}

    # -- status -------------------------------------------------------------
    @property
    def is_excluded(self):
        return getattr(self._info, "status", None) == "EXCLUDED"

    @property
    def exclusion(self):
        """(reason, pending_repair) when excluded, else None.

        `pending_repair` is the difference between "excluded" and "excluded until the
        repair pass" -- tonight every exclusion looked permanent and every one was a
        version floor.
        """
        if not self.is_excluded:
            return None
        return (getattr(self._info, "exclusion_reason", None),
                getattr(self._info, "pending_repair", None))

    @property
    def landed(self):
        """Has cells in `data/twp_grid_v3`. Distinct from `in_grid_spec`, which is
        whether it was SCHEDULED -- the gap between them is the run's progress."""
        return self.id in _landed()

    @property
    def in_spec(self):
        return bool(getattr(self._info, "in_grid_spec", False))

    # -- relations, delegated to Registry -----------------------------------
    @cached_property
    def family(self):
        return _registry().family_key(self.id)

    @cached_property
    def base(self):
        b = _registry().base_of(self.id)
        return Checkpoint(b) if b and b != self.id else None

    @cached_property
    def parent(self):
        """(Checkpoint, relation) this one descends from, or None.

        NOTE the edges are STAR-SHAPED from the base, not chained: an aligned arm's
        parent is the family's base, not its SFT checkpoint. So this is ancestry in the
        artifact's sense, not the training sequence. To compare two checkpoints, build a
        Step from the pair -- a Step does not depend on how the edges are shaped.
        """
        p = _registry().parent_of(self.id)
        return (Checkpoint(p[0]), p[1]) if p else None

    @cached_property
    def children(self):
        return [(Checkpoint(c), rel) for c, rel in _registry().children_of(self.id)]

    @cached_property
    def siblings(self):
        """Checkpoints sharing this one's base, excluding itself."""
        b = _registry().base_of(self.id) or self.id
        return [Checkpoint(v) for v in _registry().variants_of(b) if v != self.id]

    def path_to(self, other):
        """The declared relation path to another checkpoint, or []."""
        oid = other.id if isinstance(other, Checkpoint) else other
        return _registry().path(self.id, oid)

    # -- MEASURED ARTIFACTS ------------------------------------------------
    #: WHAT IS DELIBERATELY *NOT* HERE: the surface fingerprint.
    #: `data/surface_fingerprint.json` measures how a model's cells were WRITTEN
    #: into `twp_words`. RH, 2026-08-15: *is that just surveying the current
    #: situation of ClickHouse, and not a true property of the models?* **It is.**
    #: The same checkpoint re-expanded fingerprints clean, so it describes the
    #: STORE at a moment, not the model -- and a committed file describing a
    #: mutable state, keyed per model, is `NOT_IN_GRID` again: a snapshot
    #: recorded as though it were a property, false the moment the state moves.
    #: **It is also a 5.6-second query over all 401 models.** Recompute it; do
    #: not carry it. Hanging it off `Checkpoint` would have put store diagnostics
    #: on the model object and given the staleness somewhere to hide.
    #: THREE COMMITTED MEASUREMENTS THIS CLASS COULD NOT SEE, until 2026-08-15.
    #: `grep tokenizer_properties|edge_token_overlap malign_logits/*.py` returned
    #: ZERO, and `Checkpoint(...).vocab_size` returned None for a value measured
    #: and sitting in another file. **Under the plan to make this class the only
    #: public read path, that made tokenizer comparability formally unreachable**
    #: -- every migrating consumer would have lost the one artifact answering
    #: *can these two distributions be compared at all* ([6304], malign).
    #:
    #: EACH RETURNS None ONLY FOR "NO ROW", NEVER FOR A MEASURED ZERO. Tonight
    #: booked that collision three times: `is_excluded` testing a literal nothing
    #: writes, a schema field absent-reading-as-false, and null-versus-unknown.
    #: A caller that needs to tell "unmeasured" from "measured clean" uses the
    #: `_measured` companions rather than truthiness.

    @cached_property
    def tokenizer(self):
        """The `data/tokenizer_properties.json` row, or None if unmeasured.

        Carries `vocab_len` and `n_added_tokens`, which the registry row does
        NOT: the runbook's §2.22 lesson is that a vocab guard on the model does
        not protect the tokenizer and the stricter bound is
        `min(model_vocab, len(tokenizer))`. Somebody measured both and nothing
        read either.
        """
        return _tokenizer_props().get(self.id)

    @property
    def tokenizer_measured(self):
        """Whether a tokenizer row exists. Distinct from its values being zero."""
        return self.id in _tokenizer_props()


    def comparable_with(self, other, store=None):
        """The `data/edge_token_overlap.json` row for this pair, or None.

        **NECESSARY AND NOT SUFFICIENT** (lacan, [6309]). Token-ID overlap and
        stored-surface conformance are INDEPENDENT axes. dolphin/Mistral differ
        by one token in 32,000, so every field here reads comparable -- and the
        word-level join still lost 83% of prompts, because one endpoint's cells
        were written with a boundary marker. **Check `surface()` on both ends
        too; this answers the id question only.**
        """
        oid = other.id if isinstance(other, Checkpoint) else other
        e = _edge_overlap()
        return e.get((self.id, oid)) or e.get((oid, self.id))

    # -- collection ---------------------------------------------------------
    @staticmethod
    def all(landed=None, **fields):
        """Every registered checkpoint, filtered on any registry field.

            Checkpoint.all(architecture="ssm", in_grid_spec=True)
            Checkpoint.all(status="EXCLUDED")
            Checkpoint.all(cjk_tier="FLUENT", landed=True)
        """
        out = []
        for mid in _registry().models():
            cp = Checkpoint(mid)
            if landed is not None and cp.landed != landed:
                continue
            if any(getattr(cp._info, k, None) != v for k, v in fields.items()):
                continue
            out.append(cp)
        return sorted(out, key=lambda c: c.id)

    @staticmethod
    def counts(field, **fields):
        """Distribution of a field's values. Reports what it counted over."""
        import collections
        c = collections.Counter(getattr(cp._info, field, None)
                                for cp in Checkpoint.all(**fields))
        return dict(c.most_common())
