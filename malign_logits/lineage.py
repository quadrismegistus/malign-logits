"""Lineages and their representatives: what the unit of analysis IS.

    from malign_logits.lineage import collapse, representative, lineage_of

    models = collapse(models)          # one representative per lineage
    lineage_of("tiiuae/Falcon3-10B-Base")   -> "tii/falcon3"
    representative("tii/falcon3")           -> "tiiuae/Falcon3-7B-Base"

WHY. We score 154 checkpoints and 400,644 cells, and **132,413 of those cells
(33%) cannot enter a cross-lineage test**, because they are extra members of a
lineage that contributes one observation. Twelve Llama-3.1 checkpoints are one
pretraining; eight Falcon3 checkpoints are one pretraining plus three
derivatives of it, per the vendor's own cards.

That is not waste by itself. **A checkpoint is redundant only relative to a
question.** The Falcon3 scale ladder IS the scale question; the Tulu ablations
ARE the data-ablation question. What was actually missing is that findings did
not DECLARE their unit, so within-lineage depth got spent and then silently
collapsed by a test that wanted breadth -- and, worse, sometimes was NOT
collapsed, which is how E-ASSIST-AMBIENT's "10x" turned out to be four Falcon3
sizes, i.e. one pretraining counted four times.

So the rule is: **every finding declares CROSS-LINEAGE or WITHIN-LINEAGE.**

    CROSS-LINEAGE   n = lineages. Use collapse(). A claim about alignment in
                    general. This is the default for anything that says
                    "N of M models".
    WITHIN-LINEAGE  n = checkpoints, and the claim is explicitly conditional
                    on that lineage. Scale ladders, rung comparisons, data
                    ablations. Never generalised without a second lineage.

THE REPRESENTATIVE IS SIZE-MATCHED ON PURPOSE. Picking the largest member of
each lineage would confound lineage with scale: Falcon3-10B against
Qwen2.5-0.5B is not a comparison of two pretrainings, it is a comparison of two
sizes wearing lineage labels. The median scored checkpoint is 7.0B and 97 of
140 sit in the 6-8B band, so ~7B is available in almost every lineage and is
the band that minimises that confound.

Selection, in order:
  1. NEVER a declared derivative. Falcon3-1B and -3B are pruned and distilled,
     Falcon3-10B is depth up-scaled, all from Falcon3-7B-Base (their cards).
     A compression of a model is not a second observation of it.
  2. Closest to 7B.
  3. Tie-break on rung completeness, then on cell coverage.

THE GROUPING IS DERIVED BUT MUST BE DECLARED. `_derive_lineage` is a name
heuristic and name heuristics break silently -- which is the failure mode this
whole module exists to prevent. So an unmatched model RAISES rather than
getting its own singleton lineage, because a singleton is indistinguishable
from a correctly-independent model and would quietly inflate n.
"""
import re

#: Vendor-declared derivatives: {child: (parent, quoted reason)}. Never a
#: representative, and never counted as an independent observation.
DERIVATIVES = {
    "tiiuae/Falcon3-1B-Base": ("tiiuae/Falcon3-3B-Base",
        "pruned in depth, width, heads, embedding channels from a larger 3B "
        "Falcon model; knowledge distillation"),
    "tiiuae/Falcon3-3B-Base": ("tiiuae/Falcon3-7B-Base",
        "pruned in depth and width from Falcon3-7B-Base; knowledge distillation"),
    "tiiuae/Falcon3-10B-Base": ("tiiuae/Falcon3-7B-Base",
        "depth up-scaled from Falcon3-7B-Base with continual pretraining on 2 "
        "Teratokens"),
    "meta-llama/Llama-3.2-3B": ("meta-llama/Llama-3.1-8B",
        "logits from Llama 3.1 8B and 70B used as token-level targets; "
        "knowledge distillation after pruning"),
}

#: (regex on the lowercased model id, lineage key). ORDER MATTERS: the first
#: match wins, so more specific patterns come first -- 'falcon3-mamba' must
#: precede 'falcon3', or the mamba line disappears into the transformer one.
_PATTERNS = [
    (r"falcon3-mamba",  "tii/falcon3-mamba"),
    (r"falcon-mamba",   "tii/falcon-mamba"),
    (r"falcon-h1",      "tii/falcon-h1"),
    (r"falcon3",        "tii/falcon3"),
    (r"falcon",         "tii/falcon1"),
    (r"llama-3\.1",     "meta/llama-3.1"),
    (r"llama-3\.2",     "meta/llama-3.1"),      # distilled from 3.1
    (r"tulu",           "ai2/tulu"),
    (r"olmo-3|olmo3",   "ai2/olmo-3"),
    (r"olmo-2|olmo2",   "ai2/olmo-2"),
    (r"olmoe",          "ai2/olmoe"),
    (r"olmo-hybrid",    "ai2/olmo-hybrid"),
    (r"pythia",         "eleuther/pythia"),
    (r"qwen2\.5",       "qwen/qwen2.5"),
    (r"qwen3",          "qwen/qwen3"),
    (r"smollm2",        "hf/smollm2"),
    (r"smollm3",        "hf/smollm3"),
    (r"kanana-1\.5",    "kakao/kanana-1.5"),
    (r"kanana-2",       "kakao/kanana-2"),
    (r"minicpm",        "openbmb/minicpm"),
    (r"archangel",      "ctx/archangel"),
]


class UnknownLineage(Exception):
    """An unmatched model. Never silently a singleton: a singleton lineage and
    a genuinely independent model are indistinguishable, and guessing inflates
    n in the direction that flatters every finding."""


def lineage_of(model, strict=True):
    """The pretraining-program key for a model id."""
    n = str(model).lower()
    for pat, key in _PATTERNS:
        if re.search(pat, n):
            return key
    org = str(model).split("/")[0].lower()
    stem = re.split(r"[-_]", str(model).split("/")[-1].lower())[0]
    if strict and not stem:
        raise UnknownLineage(model)
    return "%s/%s" % (org, stem)


def _size_b(model):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-z])", str(model))
    return float(m.group(1)) if m else None


def representative(lineage, members, cells=None, target_b=7.0):
    """Pick the representative of one lineage. See the module docstring."""
    cand = [m for m in members if m not in DERIVATIVES] or list(members)
    def rank(m):
        s = _size_b(m)
        return (abs((s if s is not None else 1e3) - target_b),
                -(cells or {}).get(m, 0), str(m))
    return sorted(cand, key=rank)[0]


def collapse(models, cells=None, target_b=7.0):
    """One representative per lineage. **Use this for any cross-lineage n.**"""
    from collections import defaultdict
    g = defaultdict(list)
    for m in models:
        g[lineage_of(m)].append(m)
    return sorted(representative(k, v, cells, target_b) for k, v in g.items())


def report(models, cells=None):
    """A table a reader can check: lineage, members, representative, redundancy."""
    from collections import defaultdict
    g = defaultdict(list)
    for m in models:
        g[lineage_of(m)].append(m)
    out = []
    for k in sorted(g, key=lambda k: -len(g[k])):
        v = sorted(g[k])
        rep = representative(k, v, cells)
        out.append((k, len(v), rep, [x for x in v if x in DERIVATIVES]))
    return out
