"""The canonical model file: one row per model, typed edges beside it.

    uv run .venv/bin/python scripts/build_model_registry.py --dry-run
    uv run .venv/bin/python scripts/build_model_registry.py

Writes `data/model_registry.json` on the pattern of `prompt_categorisation.json`
(RH's instruction, docket [983]-[988]).

WHY IT EXISTS. Model facts lived in five places and disagreed: MODEL_FAMILIES,
a JSON cache of it, registry.py's NICKNAMES, cjk_coverage.csv, and the docket.
The cache was written on 26 June and `Registry.__init__` reads it IF IT EXISTS,
so a stale file permanently outranked the live taxonomy -- 59 models against
112, covering 41 of the 103 in the frozen spec. That is a cache-invalidation
bug wearing a knowledge gap's clothes, and this file is regenerated rather than
read-if-present for exactly that reason.

MEASURED FIELDS NAME THEIR PRODUCERS ([984].1). Nothing here takes a fresh
ad-hoc read where a sweep exists. `_provenance` maps every measured field to the
script and artifact that produced it, WITH the library versions the sweep ran
under, because tonight established that tokenizer measurements are
version-dependent -- transformers refuses .bin below torch 2.6, and the same
checkpoints load on one machine and not another.

THREE SOURCE KINDS, and the distinction is the point:

    declared    a fact about OUR taxonomy      family, position, architecture
    derived     computed from a declared fact  stage from the model name
    measured    a fact about the MODEL         vocab_size, weights_format, cjk_tier

An architecture claim was asserted from a spec read and was wrong; a stage field
says `dpo` for all four archangel arms because it was derived from a family
label rather than from the name that carries the method. Marking which fields
describe our declarations and which describe the models is what separates those
two failures from each other.

STATUS CARRIES ITS REASON AND ITS SCOPE. `"100 of 103"` becomes a QUERY over
`status`, not a sentence someone must remember to write -- and every EXCLUDED
row carries `pending_repair`, because tonight needed the difference between
"excluded" and "excluded until the repair pass" twice in one evening.

TWO LISTS. A model has several outgoing edges at once -- its training parent AND
its size sibling AND its data-ablation twin -- so relations cannot be fields on
the row. Direction is declared per relation type in `_schema`, because an edge
label that does not state its convention invites a silent inversion.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES, PATH_DATA  # noqa: E402
from malign_logits.registry import NICKNAMES  # noqa: E402

OUT = os.path.join(PATH_DATA, "model_registry.json")
# HAND-CURATED FACTS NO ARTIFACT CAN REGENERATE -- org_type, country, scale.
# The first run of this builder DESTROYED them: the file it overwrote was both a
# stale derived cache AND a store of hand-entered values, and "regenerate, never
# read-if-exists" is right for the first and fatal for the second. They were
# recovered from git and now live in their own file, which is merged and never
# rebuilt. `country` is not incidental -- the Chinese-family contrast and the
# country confound both read it.
CURATED = os.path.join(PATH_DATA, "model_curated.json")
# Registry rows that exist for BEAM-LABEL RESOLUTION and are not on the grid
# roster. They were appended to the artifact AFTER a build, so they carried no
# status fields and three tests failed on KeyError -- and a rebuild would have
# deleted them, because this producer builds from its own sources and had no
# way to learn they existed. A POPULATION THAT ENTERS AN ARTIFACT BY A PATH THE
# PRODUCER DOES NOT KNOW IS ONE THE PRODUCER WILL SILENTLY DROP.
EXTRA_MODELS = os.path.join(PATH_DATA, "registry_extra_models.json")
# THE ROSTER. Completeness is asked of the object, never of the execution plan --
# a plan that shrinks is not a roster that shrank, and "82 of 82 ACTIVE" was this
# file cheerfully reporting a full house against a spec the night had narrowed.
SPEC = os.path.join(PATH_DATA, "grid_roster.json")

# ---- measured inputs: (artifact, producer script, key column) ----
SWEEPS = {
    "weights": ("weights_audit.csv", "scripts/weights_audit.py"),
    "cjk": ("cjk_coverage.csv", "scripts/build_cjk_coverage.py"),
    "bos": ("bos_resolution.csv", "scripts/bos_resolution_sweep.py"),
    "roundtrip": ("tokenizer_roundtrip.csv", "scripts/tokenizer_roundtrip_sweep.py"),
}

FULL = 979          # a complete arm; the grid scores 979 prompts per model
# SSM/hybrid, deferred 2026-07-31. Originally booked as "compute-bound, wants a
# faster card" -- the right SYMPTOM (GPU pinned at 100%) with the wrong CAUSE.
# UTILIZATION AT 100% DOES NOT NAME ITS CAUSE: a naive state-space scan launches
# thousands of tiny kernels and reads identically on the gauge to saturated
# arithmetic. A faster card would have been $24 spent proving it.
SSM_DEFERRED = {
    "tiiuae/falcon-mamba-7b", "tiiuae/falcon-mamba-7b-instruct",
    "tiiuae/Falcon3-Mamba-7B-Base", "tiiuae/Falcon3-Mamba-7B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Base", "tiiuae/Falcon-H1-1.5B-Instruct",
    "tiiuae/Falcon-H1-7B-Base", "tiiuae/Falcon-H1-7B-Instruct"}
OOM_32B = {"allenai/Olmo-3.1-32B-Instruct-SFT",
           "allenai/Olmo-3.1-32B-Instruct-DPO"}

POSITIONS = ("base", "ego", "superego", "reinforced_superego",
             "reasoning", "reasoning_base")

# POSITION IS OUR STRUCTURAL SLOT; STAGE IS THE PROCEDURE THAT PRODUCED IT.
# They are not the same field and collapsing them is what made all four
# archangel arms read `dpo`.
POSITION_STAGE = {"base": "base", "ego": "sft", "superego": "dpo",
                  "reinforced_superego": "rlvr", "reasoning": "reasoning",
                  "reasoning_base": "base"}

# DERIVED FROM THE MODEL NAME, which carries the method the family label lost.
# archangel_sft-kto_pythia2-8b is a KTO model; the position field says superego
# for all four, and superego->dpo then erases the one variable the cell exists
# to vary.
#
# `\b` DOES NOT WORK HERE and the first version of this line silently did
# nothing: the ids read `archangel_sft-kto_pythia2-8b`, and `_` is a word
# character, so there is no boundary between `kto` and `_`. All four arms kept
# reading `dpo` and the fix looked applied. A negative lookahead for a letter is
# the correct assertion.
METHOD_IN_NAME = re.compile(r"sft-(dpo|kto|ppo|slic)(?![a-z])", re.I)

# AND A BARE `sft` IS ITSELF A METHOD THE NAME CARRIES.
#
# The regex above only recognises `sft-<method>`, so a checkpoint whose name
# says SFT and nothing else fell through to POSITION_STAGE. In a 2-LAYER
# family the single aligned member sits at `superego` by convention -- that is
# the structural slot, not a claim about training -- and superego->dpo then
# staged three Tulu-3 SFT DATA-ABLATION checkpoints as preference-optimised.
# Found by lacan at docket [4718]; it inverted an SFT-vs-DPO comparison on the
# one base carrying several recipes.
#
# The negative lookahead is load-bearing TWICE: `sft` must not be followed by
# a method (or `archangel_sft-dpo` would stage as sft, erasing the variable
# those four cells exist to vary) and must not be followed by a letter (or
# `softmax`-like substrings would match).
#
# POSITION IS NOT A STAGE. `archangel_sft-kto` is position=superego stage=kto
# and always was; the fallback treating superego as dpo is a guess for models
# whose name says nothing, and it should be reached as rarely as possible.
SFT_IN_NAME = re.compile(r"sft(?!-(?:dpo|kto|ppo|slic))(?![a-z])", re.I)

# DECLARED METHOD, for checkpoints whose NAME carries none and whose vendor
# documents one. The fallback superego->dpo is "a guess for models whose name
# says nothing", and these are the cases where the guess is known to be WRONG.
#
# Kanana 1.5: `stage=dpo` was FACTUALLY BACKWARDS. Kakao abandoned DPO for the
# 1.5 line and moved to ON-POLICY RL. tech.kakao.com/posts/707 lists the three
# changes as "On-policy 강화학습법의 도입" (introduction of on-policy RL),
# scalar -> generative reward model, and combination with verifiable reward
# functions; posts/716 names them retrospectively as RLVR (math/code) and
# RLGRM (general conversation) applied "SFT 이후" -- after SFT. The DPO the
# blog mentions belongs to the 1.0 recipe it is describing moving away from,
# which is the conflation any generic search falls into because the 1.5 card
# cites the 1.0 arXiv paper.
#
# **`ppo` HERE IS A BUCKET, NOT A NAMED ALGORITHM.** The source says
# "PPO 등의 online learning 학습법" -- on-policy methods SUCH AS PPO -- and never
# names the algorithm used for the released checkpoint. `ppo` is the closest
# value in the schema's vocabulary and is right about the family of method;
# `dpo` was wrong about the family, which is the distinction that matters when
# the project's central finding is that SFT and DPO do different work.
#
# Kanana 2: the card states "pretrained from scratch on TPU clusters and
# further improved through post-training with supervised fine-tuning and
# reinforcement learning". Same bucket, same caveat -- RL is declared, the
# algorithm is not.
METHOD_DECLARED = {
    "kakaocorp/kanana-1.5-8b-instruct-2505": "ppo",
    "kakaocorp/kanana-2-3b-instruct": "ppo",
    #: Phi-4-reasoning-plus is GRPO, not DPO. Report S4: "Starting from a
    #: strong SFT model, i.e., Phi-4-reasoning, additional GRPO training for
    #: only 90 steps". `grpo` is not in the schema vocabulary; `ppo` is the
    #: on-policy bucket and is right about the FAMILY of method where `dpo`
    #: (offline, preference-pair) is wrong about it.
    "microsoft/Phi-4-reasoning-plus": "ppo",
    #: SmolLM3's aligned arm is an APO SOUP -- anchored preference optimisation,
    #: then a 0.9/0.1 LINEAR MERGE. The merge is not a training step at all, so
    #: no stage value is honest about the edge; `ppo`-family is closer than
    #: `dpo` because APO is on-policy-anchored, and the family comment carries
    #: the merge caveat. **Do not pool this arm with DPO arms silently.**
    "HuggingFaceTB/SmolLM3-3B": "ppo",
}

# DECLARED. Not inferable from any artifact we hold, and asserting it from a
# spec read is the error this column exists to make visible.
ARCHITECTURE = {
    "tiiuae/falcon-mamba-7b": "ssm",
    "tiiuae/falcon-mamba-7b-instruct": "ssm",
    "tiiuae/Falcon3-Mamba-7B-Base": "ssm",
    "tiiuae/Falcon3-Mamba-7B-Instruct": "ssm",
    "tiiuae/Falcon-H1-1.5B-Base": "hybrid",
    "tiiuae/Falcon-H1-1.5B-Instruct": "hybrid",
    "tiiuae/Falcon-H1-7B-Base": "hybrid",
    "tiiuae/Falcon-H1-7B-Instruct": "hybrid",
    "RWKV/rwkv-4-7b-pile": "linear_attn_rnn",
    "RWKV/rwkv-raven-7b": "linear_attn_rnn",
    "allenai/OLMoE-1B-7B-0125": "moe",
    "allenai/OLMoE-1B-7B-0125-SFT": "moe",
    "allenai/OLMoE-1B-7B-0125-DPO": "moe",
    "allenai/OLMoE-1B-7B-0125-Instruct": "moe",
}


# SCALE LADDERS, DECLARED. `smaller_version_of` carries the scale question --
# one org, one recipe, three sizes, which is a far better design for "does the
# effect scale" than any cross-org comparison. Declared rather than inferred
# from the family-key prefix, because name-pattern inference produced two
# defects today (a \b that matched nothing, an architecture default asserted
# from a spec read).
#
# GENERATION IS DECLARED PER RUNG, and it is why this list is pairs rather than
# keys ([1116].2). `olmo-tiny` is OLMo-**2**; `olmo` and `olmo-32b` are OLMo-**3**
# — a different pretraining corpus and a different post-training recipe, a year
# apart. A single `smaller_version_of` relation over this ladder therefore emitted
# edges that cross a model GENERATION and called them a SCALE contrast, which is
# not what "same family, different scale" names. Split below.
#
# Declared, never parsed: `OLMo-2-0425-1B` and `Olmo-3-1025-7B` differ by one
# character in the position a parser would read, and name-pattern inference is
# retired for fact fields.
SCALE_LADDERS = [
    [("olmo-tiny", "OLMo-2"), ("olmo", "OLMo-3"), ("olmo-32b", "OLMo-3")],
    # falcon3-7b was COMPLETE IN THE STORE the whole time and simply never
    # declared ([1121].2). The ladder was not wrong about any edge it drew; it
    # was SILENT ABOUT AN EDGE IT NEVER DREW, which no assertion about existing
    # edges can catch. Hence the completeness test below.
    [("falcon3-1b", "Falcon3"), ("falcon3-3b", "Falcon3"),
     ("falcon3-7b", "Falcon3"), ("falcon3-10b", "Falcon3")],
    [("qwen-tiny", "Qwen2.5"), ("qwen", "Qwen2.5")],
]
#: DECLARED NON-RUNGS. The completeness test asks: is every same-org family that
#: differs in scale from another either ON a ladder or declared not a rung? Keyed
#: by FAMILY rather than by pair, because the reasons are properties of the family
#: (it is an architecture variant, a recipe variant, a data ablation, a different
#: base lineage) and pairwise declaration would need 52 entries saying twelve things.
#:
#: "Considered and excluded" and "never noticed" must be DIFFERENT STATES. That
#: distinction is the entire lesson of falcon3-7b, which was neither.
NOT_A_SCALE_RUNG = {
    "olmo-hybrid": "architecture contrast at one scale, not a scale rung",
    "olmo-think": "training-recipe contrast (Think) at one scale",
    "olmoe": "MoE against dense at one scale",
    "tulu": "Llama base despite the allenai org -- a different lineage",
    "tulu-no-safety": "data ablation of tulu, not a scale rung",
    "tulu-sft-nomath": "data ablation of tulu",
    "tulu-sft-nopersona": "data ablation of tulu",
    "tulu-sft-nowildchat": "data ablation of tulu",
    "qwen3": "Qwen3 -- a later GENERATION than the Qwen2.5 ladder, not a rung on it",
    "smol": "SmolLM2 against SmolLM3 is a generation change",
    "smol3": "SmolLM3 against SmolLM2 is a generation change",
    "falcon-h1-1.5b": "hybrid architecture; UNSCORED (SSM fast path unavailable)",
    "falcon-h1-7b": "hybrid architecture; UNSCORED",
    "falcon-mamba": "SSM architecture; UNSCORED",
    "falcon3-mamba": "SSM architecture against dense Falcon3; UNSCORED",
    "deepseek-7b": "different lineage from the deepseek-org distill families",
    "llama": "Llama-3.1 8B; its 70B sibling is NOT_IN_GRID, so no rung exists yet",
    "ct-llm": "2B model, not a rung on any declared series",
    "map-neo": "7B model, not a rung on any declared series",
    # Falcon-1 (2023) against Falcon3 (2024) is a GENERATION change at the same
    # org, the same call already made for qwen3 and smol/smol3 above. These
    # rows exist only for beam-label resolution (data/registry_extra_models.json)
    # and carry no measured params, so a Falcon-1/Falcon3 pair would read as a
    # scale contrast between a sized family and an unsized one.
    "falcon": "Falcon-1 7B (2023) -- a different GENERATION from the Falcon3 "
              "series, not a rung on it; off-roster, beam-label rows only",
}
#: family key -> declared generation. EMPTY MEANS UNDECLARED, NEVER "the same":
#: absence is what let two generations sit on one ladder unremarked.
GENERATION = {k: g for ladder in SCALE_LADDERS for k, g in ladder}


# ONE FIELD, TWO VOCABULARIES. The pre-existing curated data used
# corporate/state; the sourced pass used company/government. Both are correct
# English and they are the same concepts, so the values are normalised to the
# declared enum rather than the enum widened to hold synonyms -- a value list
# that admits two words for one thing cannot be asserted against.
#
# This survived my own enum assertion because that test checked position,
# architecture, stage, weights_format and status, and NOT org_type. A test
# that names its fields explicitly omits silently.
ORG_TYPE_ALIASES = {"corporate": "company", "state": "government"}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def load_csv(fn):
    p = os.path.join(PATH_DATA, fn)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return {r["model"]: r for r in csv.DictReader(fh)}


# PARAMS THE NAME CANNOT YIELD. Declared, because THREE ATTEMPTS AT A RULE BROKE
# THREE DIFFERENT FAMILIES:
#     naive `\d+b`                 archangel_sft-kto_pythia2-8b -> 8B   (it is 2.8B)
#     hyphen-as-decimal            rwkv-4-7b -> 4.7B, OLMo-2-0425-1B -> 425.1B
#     + require a letter before    Falcon3-1B -> 3.1B  (the 3 is a GENERATION)
# `pythia2-8b` and `Falcon3-1B` are syntactically identical and semantically
# opposite, so no pattern separates them. This is lacan's own rule arriving on
# my code: name-pattern inference produced two defects today, and trying to
# rescue it produced two more.
PARAMS_OVERRIDE = {
    "ContextualAI/archangel_sft_pythia2-8b": 2.8,
    "ContextualAI/archangel_sft-dpo_pythia2-8b": 2.8,
    "ContextualAI/archangel_sft-kto_pythia2-8b": 2.8,
    "ContextualAI/archangel_sft-ppo_pythia2-8b": 2.8,
    "ContextualAI/archangel_sft-slic_pythia2-8b": 2.8,
}


def params_from_name(mid):
    """(display, billions). None when the name does not carry a size.

    Overrides first; then the plain `<n>b` / `<n>m` forms. NO hyphen-as-decimal
    rule -- see PARAMS_OVERRIDE for why it cannot exist.
    """
    if mid in PARAMS_OVERRIDE:
        v = PARAMS_OVERRIDE[mid]
        return f"{v}B", v
    s = mid.lower()
    m = re.findall(r"[-_/](\d+(?:\.\d+)?)b(?![a-z0-9])", s) or \
        re.findall(r"(\d+(?:\.\d+)?)b(?![a-z0-9])", s)
    if m:
        return f"{m[-1]}B", float(m[-1])
    m = re.findall(r"(\d+)m(?![a-z0-9])", s)
    if m:
        return f"{m[-1]}M", float(m[-1]) / 1000
    return "", None


def build():
    spec = json.load(open(SPEC))
    spec_rows = spec["spec"] if isinstance(spec, dict) else spec
    in_spec = {r["model"] for r in spec_rows}
    sw = {k: load_csv(v[0]) for k, v in SWEEPS.items()}
    _c = json.load(open(CURATED)) if os.path.exists(CURATED) else {}
    cur = _c.get("models", {})
    org_facts = _c.get("orgs", {})
    param_facts = _c.get("params", {})

    rows, relations = {}, []
    #: (child, relation) -> chosen parent, resolved once after all families are
    #: walked. A ModelFamily is a comparison structure and several may name the
    #: same endpoint; descent is not a per-family fact.
    _lineage = {}

    def _prefix(a, b):
        """Shared-prefix length of two model ids, for deterministic parent choice."""
        a, b = a.split("/")[-1], b.split("/")[-1]
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    for fam_key, fam in sorted(MODEL_FAMILIES.items()):
        arms = {p: getattr(fam, p, None) for p in POSITIONS}
        base_id = arms.get("base")
        for pos, mid in arms.items():
            if not mid:
                continue
            r = rows.setdefault(mid, {"model_id": mid})
            # A MODEL CAN SIT IN TWO FAMILIES (a shared base). The row records
            # the first family that claims it and the rest become edges, rather
            # than the row silently taking whichever family sorted last.
            if "family" not in r:
                w = sw["weights"].get(mid, {})
                cj = sw["cjk"].get(mid, {})
                bo = sw["bos"].get(mid, {})
                rt = sw["roundtrip"].get(mid, {})
                method = METHOD_IN_NAME.search(mid)
                r.update(
                    nickname=NICKNAMES.get(mid, ""),
                    family=fam_key, position=pos,
                    #: DECLARED beats derived-from-name beats slot default.
                    #: The slot default is explicitly "a guess for models whose
                    #: name says nothing" and must lose to a vendor statement.
                    stage=(METHOD_DECLARED.get(mid)
                           or (method.group(1).lower() if method
                               else "sft" if SFT_IN_NAME.search(mid)
                               else POSITION_STAGE.get(pos, ""))),
                    org=mid.split("/")[0],
                    params=params_from_name(mid)[0],
                    params_b=params_from_name(mid)[1],
                    architecture=ARCHITECTURE.get(mid, "transformer"),
                    tokenizer_class=rt.get("tokenizer", ""),
                    #: **TWO VOCABULARY NUMBERS, AND THEY ARE NOT THE SAME
                    #: FACT.** This one is the CJK survey's, i.e.
                    #: `tokenizer.vocab_size`, which EXCLUDES added tokens. It
                    #: is kept because it is what the tokenizer reports. It is
                    #: NOT the number that governs an out-of-range embedding
                    #: assert -- that is the config's, carried separately in
                    #: `vocab_size_config` by scripts/probe_model_requirements.py.
                    #: 43 of 70 probed models differ between the two, and the
                    #: pair that cost 85 sites (llama-7b > beaver-7b-v1.0,
                    #: 32000 vs 32001) looked IDENTICAL on this field.
                    vocab_size=int(cj["vocab_size"]) if cj.get("vocab_size") else None,
                    cjk_tier=cj.get("tier", ""),
                    cjk_chars=int(cj["cjk_chars"]) if cj.get("cjk_chars") else None,
                    weights_format=w.get("weights_format", ""),
                    # A REAL BOOLEAN. It was the STRING "true"/"false", so
                    # `if row["index_present"]` was True for "false" as well --
                    # one field, two types, waiting.
                    index_present=({"true": True, "false": False}
                                   .get(w.get("index_present", ""), None)),
                    needs_torch=w.get("needs_torch", ""),
                    bos_stratum=bo.get("stratum", ""),
                    bos_resolver=bo.get("resolver", ""),
                    loader_override=(bo.get("loader", "")
                                     if bo.get("loader") not in ("", "AutoTokenizer")
                                     else ""),
                    # fam_key, NOT fam -- `fam` is the ModelFamily object and
                    # .get() on it returns "" for every row without erroring,
                    # which would ship an all-empty column that reads as
                    # "no generation declared anywhere".
                    generation=GENERATION.get(fam_key, ""),
                    in_grid_spec=mid in in_spec,
                    status="ACTIVE", exclusion_reason="", pending_repair=None,
                )
                r.update(cur.get(mid, {}))          # curated wins; never derived
                if r.get("org_type") in ORG_TYPE_ALIASES:
                    r["org_type"] = ORG_TYPE_ALIASES[r["org_type"]]
                # ORG-LEVEL curated facts, applied per row. Only non-empty
                # values are written: an org whose country is deliberately
                # blank (a multinational collective) must stay blank rather
                # than acquire "" as though it were unknown-by-omission.
                of = org_facts.get(mid.split("/")[0], {})
                for k in ("country", "org_type"):
                    if of.get(k):
                        r[k] = of[k]
                # SOURCED params override every parse. bloom-7b1 is 7.069B by
                # its own card -- the row I declined to guess, and the guess
                # would have been 7.1.
                pf = param_facts.get(mid)
                if pf:
                    r["params_b"] = pf["b"]
                    r["params"] = f"{pf['b']}B"
            else:
                relations.append({"parent": mid, "child": fam_key,
                                  "relation": "also_member_of"})
        # TRAINING-AXIS EDGES: CHAINED, NOT STAR-SHAPED, and typed by the
        # child's STAGE rather than its position.
        #
        # Star shape hung every aligned arm off the BASE, so the SFT step was
        # invisible to a traversal and "what sequence of training produced this
        # checkpoint" could not be answered from the relations at all. The
        # parent is now the previous stage where one exists.
        #
        # And the label follows the stage: the four archangel arms differ only
        # by preference method, so an edge typed from `position` called kto_of
        # dpo_of and contradicted the node it pointed at.
        prev = base_id
        for pos in ("ego", "superego", "reinforced_superego"):
            mid_pos = arms.get(pos)
            if not mid_pos:
                continue
            if prev:
                st = rows[mid_pos].get("stage") or POSITION_STAGE[pos]
                #: **A CHECKPOINT HAS ONE LINEAGE PARENT PER RELATION TYPE.**
                #: A ModelFamily is a COMPARISON structure -- `tulu-no-safety`
                #: names the MAIN DPO as its superego so the ablation has an
                #: endpoint to be read against -- and chaining it as descent
                #: turns that into a claim the ablation arm PRODUCED that DPO.
                #: It did not; AI2 released no DPO for the ablation arms. The
                #: result was two `dpo_of` parents for Llama-3.1-Tulu-3-8B-DPO,
                #: which made `Registry.base_of` raise AMBIGUOUS and silently
                #: removed tulu from every roster that resolves a base.
                #:
                #: Resolved by NAME PROXIMITY, not by file order, which this
                #: registry exists to forbid: the true parent shares a longer
                #: prefix with the child than a sibling ablation does
                #: (`...Tulu-3-8B-SFT` vs `...Tulu-3-8B-SFT-no-safety-data`
                #: against child `...Tulu-3-8B-DPO`). Deterministic, and it
                #: generalises to any suite where one endpoint is shared.
                key = (mid_pos, f"{st}_of")
                cand = _lineage.get(key)
                if cand is None or _prefix(prev, mid_pos) > _prefix(cand, mid_pos):
                    _lineage[key] = prev
            prev = mid_pos
        if base_id and arms.get("reasoning"):
            relations.append({"parent": base_id, "child": arms["reasoning"],
                              "relation": "reasoning_of"})
    #: emitted AFTER every family is walked, so the winner is chosen against all
    #: candidates rather than against whichever family came first.
    for (child, rel), parent in sorted(_lineage.items()):
        relations.append({"parent": parent, "child": child, "relation": rel})
    return rows, relations, sw


def lateral(rows, relations):
    """The edges ModelFamily has no slot for -- and the schema's payoff."""
    by_base = defaultdict(list)
    for mid, r in rows.items():
        for fk, fam in MODEL_FAMILIES.items():
            if fam.name == r["family"] or fk == r["family"]:
                if getattr(fam, "base", None):
                    by_base[fam.base].append(mid)
    # same_base_as: the separator cell becomes a query instead of four reads
    for base, members in by_base.items():
        sibs = sorted(set(members) - {base})
        for i, a in enumerate(sibs):
            for b in sibs[i + 1:]:
                relations.append({"parent": a, "child": b,
                                  "relation": "same_base_as"})
    # scale ladders: ORDERED at [984].5 and absent from the first build.
    # Emitted between consecutive rungs AND transitively, so "every olmo" is one
    # traversal rather than a hand-assembly from three unconnected family keys.
    fam_of = {}
    for mid, r in rows.items():
        fam_of.setdefault(r.get("family"), []).append(mid)
    # THE RELATION IS SPLIT BY GENERATION ([1116].2). `smaller_version_of`
    # conflated "smaller sibling in the same release" with "smaller predecessor",
    # so a pair count taken off it answered a question the clause did not ask:
    # 6 pairs, of which only 4 were same-generation. A clause that says
    # "same family, different SCALE" may only draw on the first kind.
    for ladder in SCALE_LADDERS:
        present = [(f, g) for f, g in ladder if fam_of.get(f)]
        for i, (small, g_small) in enumerate(present):
            for big, g_big in present[i + 1:]:
                rel = ("smaller_sibling_of" if g_small == g_big
                       else "smaller_predecessor_of")
                for a in fam_of[small]:
                    for b in fam_of[big]:
                        if rows[a].get("position") == rows[b].get("position"):
                            relations.append({"parent": a, "child": b,
                                              "relation": rel})

    # data_ablation_of: the pair that can say WHICH training data
    for mid in rows:
        if "no-" in mid and "Tulu" in mid:
            full = re.sub(r"-no-[a-z]+-data", "", mid)
            if full in rows:
                relations.append({"parent": full, "child": mid,
                                  "relation": "data_ablation_of"})
    return relations


def main(a):
    rows, relations, sw = build()
    relations = lateral(rows, relations)

    # ---- EXCLUSIONS ARE PASS-SCOPED ([984].3) ----
    # A model requiring torch >= 2.6 is not defective; it was unloadable ON THE
    # BOX THE v3 GRID RAN ON, which had 2.5.1. Writing that as a property of the
    # model would be the mistake this field exists to prevent -- tonight every
    # exclusion looked permanent and every one was a version floor.
    # EXCLUSION IS DECIDED BY THE STORE, NOT BY THE WEIGHTS FORMAT. `needs_torch`
    # is a property of the CHECKPOINT; `status` is a question about whether we
    # HAVE the data. The two coincided only while the box sat below the floor.
    # The repair pass scored all thirteen under torch 2.6, so keying exclusion
    # off the format would report models as missing while their cells sit in the
    # store -- the completeness query answering from a cause instead of a fact.
    from malign_logits.cache import get_cache as _gc
    # ── DECLARED EXTRA ROWS ──────────────────────────────────────────────
    # Added here, BEFORE the status loop, so they go through exactly the same
    # NOT_IN_GRID stamping as any other off-roster row. Adding them after would
    # reproduce the defect this file exists to fix.
    if os.path.exists(EXTRA_MODELS):
        _x = json.load(open(EXTRA_MODELS))
        for mid, meta in (_x.get("models") or {}).items():
            if mid in rows:
                continue
            r = dict(model_id=mid, nickname=NICKNAMES.get(mid, ""),
                     family=meta.get("family", ""), position="",
                     stage=meta.get("stage", ""), org=mid.split("/")[0],
                     params="", params_b=None, architecture="",
                     tokenizer_class="", vocab_size=None, cjk_tier="",
                     cjk_chars=None, weights_format="", index_present=None,
                     needs_torch="", bos_stratum="", bos_resolver="",
                     loader_override="", generation="",
                     in_grid_spec=False,
                     status="ACTIVE", exclusion_reason="", pending_repair=None)
            _cj = json.load(open(CURATED)) if os.path.exists(CURATED) else {}
            r.update((_cj.get("models") or {}).get(mid, {}))   # curated wins
            _of = (_cj.get("orgs") or {}).get(mid.split("/")[0], {})
            for _k in ("country", "org_type"):
                if _of.get(_k):
                    r[_k] = _of[_k]
            # ONE FIELD, TWO VOCABULARIES -- the curated data says
            # corporate/state and the schema declares company/government. The
            # main loop normalises; the first version of this block did not,
            # and shipped two undeclared enum values into a rebuilt artifact.
            if r.get("org_type") in ORG_TYPE_ALIASES:
                r["org_type"] = ORG_TYPE_ALIASES[r["org_type"]]
            rows[mid] = r
        print(f"  + {len(_x.get('models') or {})} declared extra row(s) "
              f"(beam-label resolution; off-roster)")

    _cm = _gc()
    # CELL count per model, one pass, through the declared schema. A hand-built
    # key here would go stale at the next rekey exactly as fourteen others did.
    scored = _cm.count_by("true_word_probs", "model")

    n_exc = 0
    for mid, r in rows.items():
        # COVERAGE IS MEASURED FOR EVERY ROW, so absence of the field can never be
        # read as zero-by-luck and a KeyError can never stand in for "no data".
        n = scored.get(mid, 0)
        r["cells_in_store"] = n
        if not r.get("in_grid_spec"):
            # NEVER ASKED IS NOT THE SAME ANSWER AS ASKED AND ANSWERED. These nine
            # rows kept the constructor's default ACTIVE, so a registry-wide
            # `status == "ACTIVE"` returned 102 while 93 models were scored -- the
            # roster-scoped judgment applied to the registry-scoped population. The
            # buckets still summed to the whole, which is why a partition test
            # passes over it: a partition is not a semantics.
            r["status"] = "NOT_IN_GRID"
            r["exclusion_reason"] = ("not on the v3 grid roster; never asked. Not a "
                                     "failure and not a gap in coverage")
            continue
        # COVERAGE DECIDES STATUS; the REASON is looked up afterwards. Keying
        # status off a cause misses every model that went missing for a
        # different one -- the first version excluded the thirteen torch-floor
        # checkpoints and silently passed Falcon x8 and the two 32B arms, which
        # are equally unscored and merely fail for other reasons.
        if n >= FULL:
            continue
        w = sw["weights"].get(mid, {})
        if w.get("needs_torch") == "2.6":
            reason = ("safetensors shards present but index absent; falls back "
                      "to the .bin index, refused below torch 2.6"
                      if w.get("weights_format") == "mixed"
                      else "bin-only checkpoint; refused below torch 2.6")
        elif mid in SSM_DEFERRED:
            reason = ("SSM/hybrid: the mamba-ssm / causal-conv1d fast-path CUDA "
                      "kernels are UNAVAILABLE at torch 2.6, so transformers "
                      "falls back to a naive scan -- 0.068 p/s against 0.9-2.5 "
                      "for dense transformers, ~32 h for the eight. A source "
                      "build succeeded and the extension still fails to load "
                      "(undefined symbol c10::cuda::c10_cuda_check_"
                      "implementation); the version that works needs torch "
                      "2.13, which breaks the compiled triple and sits above "
                      "nothing the .bin floor needs. NOT a card problem. "
                      "Deferred at docket [1144] to the F35 book-phase "
                      "decision; 68 partial cells preserved unmerged at "
                      "data/twp_phasefalcon/")
        elif mid in OOM_32B:
            reason = ("CUDA OOM at load: 64 GB of fp16 weights plus transient "
                      "peaks against 79.15 GiB usable; wants >80 GB, not a "
                      "newer torch")
        else:
            reason = f"incomplete: {n} of {FULL} cells"
        r["status"] = "EXCLUDED"
        r["exclusion_reason"] = reason
        r["pending_repair"] = True
        r["excluded_from"] = "grid_v3"
        n_exc += 1

    # dedupe; an edge asserted twice is not two edges
    seen, uniq = set(), []
    for e in relations:
        k = (e["parent"], e["child"], e["relation"])
        if k not in seen and e["parent"] in rows and e["child"] in rows:
            seen.add(k)
            uniq.append(e)

    doc = {
        "_schema": {
            "note": ("Regenerate; never read-if-exists. A cache that can outrank "
                     "its source is how 59 models shadowed 112 for five weeks."),
            "fields": {
                "family": {"source": "declared", "from": "MODEL_FAMILIES"},
                "position": {"source": "declared",
                             "values": list(POSITIONS)},
                "stage": {"source": "derived",
                          "values": ["base", "sft", "dpo", "kto", "ppo", "slic",
                                     "rlvr", "reasoning", "instruct"],
                          "note": ("from the MODEL NAME where it carries the "
                                   "method, else from position. The four "
                                   "archangel arms differ only by method and a "
                                   "position-derived stage called all four dpo.")},
                "country": {"source": "curated (sourced)",
                            "note": ("blank where ONE country would mislead -- "
                                     "RWKV, bigscience and m-a-p are multinational, "
                                     "and huggyllama is a re-uploader, not an "
                                     "originator")},
                "org_type": {"source": "curated (sourced)",
                             "values": ["company", "academic", "nonprofit",
                                        "collective", "individual", "government"],
                             "note": ("`government` was added because the original "
                                      "five had no slot for a state research "
                                      "institute -- TII is part of the Abu Dhabi "
                                      "government's ATRC and HF's own badge calls "
                                      "it a company")},
                "architecture": {"source": "declared",
                                 "values": ["transformer", "ssm", "hybrid",
                                            "linear_attn_rnn", "moe"]},
                "vocab_size": {"source": "measured"},
                "cjk_tier": {"source": "measured"},
                "weights_format": {"source": "measured",
                                   "values": ["safetensors", "bin", "mixed",
                                              "none", "unknown"]},
                "index_present": {"source": "measured",
                                  "note": ("sharded checkpoints only; decides "
                                           "loadability where format does not")},
                "needs_torch": {"source": "measured",
                                "note": "transformers refuses .bin below 2.6"},
                "status": {"source": "declared",
                           "values": ["ACTIVE", "EXCLUDED", "NOT_IN_GRID"],
                           "note": ("scoped to the ROSTER. ACTIVE and EXCLUDED are "
                                    "answers about coverage and only a model that "
                                    "was asked can hold one; NOT_IN_GRID is the "
                                    "registry rows the v3 grid never asked, which "
                                    "is not a coverage gap.")},
                "pending_repair": {"source": "declared",
                                   "note": ("EXCLUDED rows only. 'excluded' and "
                                            "'excluded until the repair pass' "
                                            "are different facts.")},
            },
            "relations": {
                "field_names": ["parent", "child", "relation"],
                # A WORKED SENTENCE PER TYPE, not a direction flag. lacan's own
                # two founding examples required OPPOSITE readings of one
                # convention -- "X is the kto_of Y" and "X is the
                # smaller_version_of Y" cannot both be true under one rule. A
                # sentence naming both roles cannot be read backwards.
                "reads": {
                    "sft_of": "{child} is the SFT-tuned version of {parent}",
                    "dpo_of": "{child} is the DPO-aligned version of {parent}",
                    "kto_of": "{child} is the KTO-aligned version of {parent}",
                    "ppo_of": "{child} is the PPO-aligned version of {parent}",
                    "slic_of": "{child} is the SLiC-aligned version of {parent}",
                    "rlvr_of": "{child} is the RLVR-tuned version of {parent}",
                    "instruct_of": "{child} is the instruct version of {parent}",
                    "reasoning_of": "{child} is the reasoning-distilled version of {parent}",
                    # SPLIT AT [1116].2. One relation cannot carry both readings:
                    # a smaller sibling shares a pretraining corpus and a recipe,
                    # a smaller predecessor does not, and only the first is the
                    # "same family, different scale" a scale clause may draw on.
                    "smaller_sibling_of": ("{parent} is a smaller-parameter version of "
                                           "{child} FROM THE SAME RELEASE (same declared "
                                           "generation)"),
                    "smaller_predecessor_of": ("{parent} is a smaller-parameter model from "
                                               "an EARLIER GENERATION than {child} -- scale "
                                               "AND release differ, so this pair is not a "
                                               "scale contrast"),
                    "data_ablation_of": "{child} is {parent} trained without some data",
                    "same_base_as": "{parent} and {child} share a base checkpoint",
                    "also_member_of": "{parent} is also a member of family {child}",
                },
                "symmetric": ["same_base_as"],
            },
        },
        "_provenance": {
            "built_by": "scripts/build_model_registry.py",
            "head": sh("git rev-parse --short HEAD"),
            "measured_from": {
                k: {"artifact": f"data/{v[0]}", "producer": v[1],
                    "rows": len(sw[k])} for k, v in SWEEPS.items()},
            "measurement_context": {
                "note": ("tokenizer and weight measurements are library-version "
                         "dependent; the sweeps ran locally, the v3 grid ran on "
                         "transformers 5.14.1 / torch 2.5.1"),
            },
        },
        "models": [rows[k] for k in sorted(rows)],
        "relations": sorted(uniq, key=lambda e: (e["relation"], e["parent"])),
    }

    print(f"models {len(doc['models'])}   relations {len(doc['relations'])}")
    print(f"  in the frozen spec: {sum(1 for r in doc['models'] if r['in_grid_spec'])}"
          f" / {len(json.load(open(SPEC))['spec'])}")
    import collections
    print(f"  stages: {dict(collections.Counter(r['stage'] for r in doc['models']))}")
    arch = {r["model_id"].split("/")[-1]: r["stage"]
            for r in doc["models"] if "archangel_sft-" in r["model_id"]}
    print("  archangel (the separator cell -- these MUST differ):")
    for k, v in sorted(arch.items()):
        print(f"    {k:<38}{v}")
    if len(set(arch.values())) != len(arch):
        print("    !! STAGES NOT DISTINCT -- the method is the experiment here")
    print(f"  relations: {dict(collections.Counter(e['relation'] for e in doc['relations']))}")
    act = sum(1 for r in doc["models"] if r["in_grid_spec"] and r["status"] == "ACTIVE")
    print(f"\n  COMPLETENESS IS A QUERY, not a sentence: "
          f"{act} of {sum(1 for r in doc['models'] if r['in_grid_spec'])} ACTIVE"
          f"   ({n_exc} EXCLUDED, all pending_repair)")
    # THE DENOMINATOR IS PART OF THE ANSWER. Reporting the roster count without the
    # registry count invites the reading that the registry IS the roster, which is
    # how nine never-asked models sat at ACTIVE.
    nig = sum(1 for r in doc["models"] if r["status"] == "NOT_IN_GRID")
    print(f"  registry holds {len(doc['models'])}; {nig} NOT_IN_GRID (never asked, "
          f"not a coverage gap)")
    assert act + n_exc + nig == len(doc["models"]), "status buckets do not partition"
    assert all(r["status"] != "ACTIVE" or r["in_grid_spec"] for r in doc["models"]), (
        "a model outside the roster is marked ACTIVE; never-asked is not answered")
    assert all("cells_in_store" in r for r in doc["models"]), (
        "coverage absent on some row; absence would be read as zero")
    #: **CARRY FORWARD THE FIELDS THIS BUILDER DOES NOT PRODUCE.**
    #: `vocab_size_config`, `requires_pip` and `load_note` are written by
    #: scripts/probe_model_requirements.py, which reads repo listings and
    #: configs. A rebuild that dropped them would silently un-learn four
    #: pair-killing facts (tiktoken, sentencepiece, mamba kernels, and the
    #: +1 vocabulary that asserts) and nothing downstream would raise --
    #: the preflight would simply report every pair clean.
    PROBE_FIELDS = ("vocab_size_config", "requires_pip", "load_note")
    if os.path.exists(OUT):
        try:
            prev = {m["model_id"]: m for m in json.load(open(OUT))["models"]}
        except Exception:
            prev = {}
        carried = 0
        for r in doc["models"]:
            old_row = prev.get(r["model_id"], {})
            for f in PROBE_FIELDS:
                if f in old_row and f not in r:
                    r[f] = old_row[f]
                    carried += 1
        if carried:
            print("  carried %d probe-written field(s) forward; re-run "
                  "scripts/probe_model_requirements.py --write to refresh"
                  % carried)
    if a.dry_run:
        print("\nDRY RUN -- nothing written")
        return 0
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(ap.parse_args()))
