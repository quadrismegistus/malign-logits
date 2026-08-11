"""Lineages and their representatives: the unit of analysis, from the declared map.

    from malign_logits.lineage import lineage_of, collapse, representative

    lineage_of("tiiuae/Falcon3-3B-Base")     -> "tiiuae/Falcon3-7B-Base"
    collapse(models)                          -> one representative per lineage
    representative("tiiuae/Falcon3-7B-Base", members)

THE SOURCE IS `data/lineage_map_models.json`, NOT A NAME HEURISTIC. The first
version of this module grouped models with regexes on their ids, and its own
docstring said why that was wrong -- "name heuristics break silently, which is
the failure mode this whole module exists to prevent" -- and then shipped the
heuristic anyway. On 2026-08-10 it was used to report "52 pairs -> 45 lineages"
as if that were a measurement. It was not: it merged Llama-3.1 8B/70B and
Falcon-H1 1.5B/7B by luck of pattern, and would have split any model whose name
did not match a case someone had thought of.

The map has existed since 1 Aug with a written rule, six consumers, and a
`_unit_warning` recording that the roster was once counted as **37, 42, 21 and
32 in one evening** because four calculations used four units. This module was
the fifth. It now reads the map.

    THE RULE (the map's, quoted): "A model's lineage is its BASE CHECKPOINT's.
    Lineages are connected components over the registry's base relation. Two
    alignment recipes applied to one base are two recipes, not two
    implementations."

    THE CAVEAT (the map's, and it binds every count here): the lineage count is
    an UPPER BOUND on independence. Relations are populated unevenly, so two
    genuinely related models with no recorded edge stay separate. That is not
    hypothetical -- sibling edges existed for Falcon3, Olmo-3 and Qwen2.5 and
    for nobody else, so Llama-3.1 and Falcon-H1 were separate lineages BY
    OMISSION until 2026-08-10.

WHY THE REPRESENTATIVE IS STORED AND NOT COMPUTED. **This docstring said the
opposite until 2026-08-10, and the argument was wrong for tonight's own
reason.** "It depends on the question, so compute it" sounds principled and
produces N implementations of one decision -- and this module's copy sorted on
a name-parsed size, read `archangel_sft-dpo_pythia2-8b` as 8.0B (it is a 2.8B
pythia), and picked that ALIGNED arm to stand for the pythia lineage. The map
holds `lineage_to_representative` and `_representative_rule`; this reads them.
A caller wanting a different target size passes `target_b` and gets a LOCAL
answer it is then obliged to report as local.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, "data", "lineage_map_models.json")

#: VENDOR-DECLARED DERIVATIONS, each a quote from the model's own card. These
#: are a STRONGER claim than the map makes: the map says these models share a
#: lineage, the cards say which came from which. A derivative is never a
#: representative -- a compression of a model is not a second observation of it.
#:
#: `Falcon3-10B-Base` is DELIBERATELY ABSENT. Its card says "depth up-scaled
#: from Falcon3-7B-Base with continual pretraining on 2 Teratokens" -- 2T of new
#: tokens is more than most models in this roster see in total, so it is its own
#: lineage, not a derivative of 7B (RH, 2026-08-10). 1B and 3B are pruned and
#: distilled at 80-100 GT, which is a compression; the distinction is the
#: token budget, and it is the vendor's own.
DERIVATIVES = {
    "tiiuae/Falcon3-1B-Base": (
        "tiiuae/Falcon3-3B-Base",
        "pruned in terms of depth, width, number of heads, and embedding "
        "channels from a larger 3B Falcon model, and efficiently trained on "
        "only 80 GT using a knowledge distillation objective"),
    "tiiuae/Falcon3-3B-Base": (
        "tiiuae/Falcon3-7B-Base",
        "pruned in terms of depth and width from Falcon3-7B-Base and "
        "efficiently trained on only 100 GT using a knowledge distillation "
        "objective"),
    "meta-llama/Llama-3.2-3B": (
        "meta-llama/Llama-3.1-8B",
        "logits from the Llama 3.1 8B and 70B models used as token-level "
        "targets; knowledge distillation after pruning"),
}

_MAP = None


class UnmappedModel(Exception):
    """A model absent from the lineage map.

    **NEVER SILENTLY A SINGLETON.** A model with no lineage and a model that is
    genuinely independent are indistinguishable once you default, and the
    default inflates n in the direction that flatters every finding. Rebuild
    the map (`scripts/build_lineage_map.py --write`) rather than catching this.
    """


def _map():
    global _MAP
    if _MAP is None:
        with open(MAP) as fh:
            _MAP = json.load(fh)
    return _MAP


def lineage_of(model, strict=True):
    """The lineage id (its base checkpoint) for a model. Raises if unmapped."""
    m = _map()["model_to_lineage"]
    if model in m:
        return m[model]
    if strict:
        raise UnmappedModel(
            "%r is not in %s. The map is regenerated from the registry; a "
            "model missing from it usually means the registry gained a model "
            "and the map was not rebuilt. Run "
            "scripts/build_lineage_map.py --write." % (model, os.path.basename(MAP)))
    return None


def base_of(model):
    """The model's own base checkpoint -- the RUN level, one grain finer than
    the lineage. Two scale siblings share a lineage and have different bases."""
    return _map()["model_to_base"].get(model)


def representative(lineage, members=None, cells=None, target_b=None):
    """The lineage's representative. **READ FROM THE MAP, not recomputed.**

    The rule and its result are both in `lineage_map_models.json`
    (`_representative_rule`, `lineage_to_representative`) because six scripts
    consume that file, and two implementations of one decision is the defect
    this campaign spent 2026-08-10 removing. This function had its own copy --
    with its own name-parsed size, which read
    `archangel_sft-dpo_pythia2-8b` as 8.0B (it is a 2.8B pythia) and picked
    that ALIGNED arm to stand for the pythia lineage. The map's version sorts
    on the registry's `params_b` and prefers base-stage members.

    `target_b` is the escape hatch: pass one and you get a LOCAL computation
    with a declared target, which is a different quantity from the stored
    answer and should be reported as such.
    """
    if target_b is None:
        rep = _map()["lineage_to_representative"].get(lineage)
        if rep:
            return rep
        if members is None:
            raise UnmappedModel(
                "%r has no stored representative and no members were passed. "
                "Rebuild with scripts/build_lineage_map.py --write." % lineage)
    #: LOCAL, DECLARED-TARGET PATH. Sizes come from the registry, never from
    #: the model id -- the parse is what produced the archangel error.
    import json as _json
    reg = os.path.join(ROOT, "data", "model_registry.json")
    with open(reg) as fh:
        rows = _json.load(fh)["models"]
    pb = {m["model_id"]: m.get("params_b") for m in rows}
    stage = {m["model_id"]: str(m.get("stage", "")).lower() for m in rows}
    members = list(members or [])
    pool = [m for m in members if stage.get(m) in ("base", "id")] or members
    cand = [m for m in pool if m not in DERIVATIVES] or pool
    t = 7.0 if target_b is None else target_b
    return sorted(cand, key=lambda m: (
        0 if pb.get(m) is not None else 1,
        abs((pb.get(m) if pb.get(m) is not None else 1e3) - t),
        -(cells or {}).get(m, 0), str(m)))[0]


def groups(models, strict=True):
    """{lineage_id: [models]} for the given models."""
    from collections import defaultdict
    g = defaultdict(list)
    for m in models:
        g[lineage_of(m, strict=strict)].append(m)
    return dict(g)


def collapse(models, cells=None, target_b=None, strict=True):
    """One representative per lineage. **Use this for any cross-lineage n.**"""
    return sorted(representative(k, v, cells, target_b)
                  for k, v in groups(models, strict=strict).items())


def report(models, cells=None, strict=True):
    """[(lineage, members, representative, derivatives)] -- a checkable table."""
    out = []
    for k, v in groups(models, strict=strict).items():
        v = sorted(v)
        out.append((k, v, representative(k, v, cells),
                    [x for x in v if x in DERIVATIVES]))
    return sorted(out, key=lambda r: (-len(r[1]), r[0]))
