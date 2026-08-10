"""Assertions on `data/model_registry.json`, in the commit that founds it.

The prompt file needed thirty assertions and a full day to become trustworthy.
These exist so the model file does not repeat that, and every one was watched to
FAIL before being allowed to pass -- a check whose green means only that it
declined to look is the failure mode this whole apparatus replaces.

The file is regenerated, never read-if-exists: `Registry.__init__` read a cache
written on 26 June IF IT EXISTED, so 59 models permanently shadowed 112 and
covered 41 of the 103 in the frozen spec. A cache that can outrank its source is
not a cache.
"""
import itertools
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "data", "model_registry.json")
SPEC = os.path.join(HERE, "data", "grid_roster.json")


@pytest.fixture(scope="module")
def doc():
    if not os.path.exists(PATH):
        pytest.skip("model_registry.json not built")
    with open(PATH) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def ids(doc):
    return {m["model_id"] for m in doc["models"]}


def test_every_declared_enum_value_is_used_and_every_used_value_declared(doc):
    """A value list nobody checks is documentation, not a schema."""
    f = doc["_schema"]["fields"]
    # org_type AND country ARE IN THIS LIST BECAUSE THEY WERE NOT. The field
    # carried `corporate` from one source and `company` from another -- one
    # field, two vocabularies -- and this test did not look because it named
    # its fields and org_type was not among them. A test that enumerates
    # omits silently.
    for field in ("position", "architecture", "stage", "weights_format",
                  "status", "org_type"):
        declared = set(f[field].get("values", []))
        if not declared:
            continue
        used = {m.get(field) for m in doc["models"] if m.get(field)}
        assert used <= declared, f"{field}: undeclared values {used - declared}"


def test_archangel_stages_are_distinct(doc):
    """THE ONE CELL WHERE THE METHOD IS THE EXPERIMENT.

    Four arms share a base and one SFT checkpoint and differ only by preference
    method. A stage derived from `position` called all four `dpo`, erasing the
    single variable the cell exists to vary. Watched to fail against the
    position-derived form.
    """
    arms = {m["model_id"]: m["stage"] for m in doc["models"]
            if "archangel_sft-" in m["model_id"]}
    assert len(arms) == 4, f"expected 4 archangel arms, found {len(arms)}"
    assert len(set(arms.values())) == 4, f"stages not distinct: {arms}"
    assert set(arms.values()) == {"dpo", "kto", "ppo", "slic"}, arms


def test_every_spec_model_has_a_row(doc, ids):
    with open(SPEC) as fh:
        s = json.load(fh)
    spec_ids = {r["model"] for r in (s["spec"] if isinstance(s, dict) else s)}
    assert spec_ids <= ids, f"missing rows: {sorted(spec_ids - ids)[:5]}"


def test_relation_endpoints_exist(doc, ids):
    for e in doc["relations"]:
        assert e["parent"] in ids, f"dangling parent {e['parent']}"
        assert e["child"] in ids, f"dangling child {e['child']}"


def test_relation_types_are_declared(doc):
    declared = set(doc["_schema"]["relations"]["reads"])
    used = {e["relation"] for e in doc["relations"]}
    assert used <= declared, f"undeclared relation types: {used - declared}"


def test_relation_direction_is_stated(doc):
    """An edge label without a direction convention invites a silent inversion.

    And the convention must name the FIELDS it applies to: this file's edges
    were written source/target while `Relation` declares parent/child, so the
    two disagreed and the assertions read a key that was not there.
    """
    rel = doc["_schema"]["relations"]
    assert rel.get("reads"), "no per-type reading sentences"
    assert rel.get("field_names") == ["parent", "child", "relation"]
    for e in doc["relations"]:
        assert set(e) == {"parent", "child", "relation"}, e


def test_hierarchical_relations_are_acyclic(doc):
    """A model cannot be its own training ancestor."""
    hier = {"sft_of", "dpo_of", "rlvr_of", "reasoning_of"}
    adj = {}
    for e in doc["relations"]:
        if e["relation"] in hier:
            adj.setdefault(e["parent"], []).append(e["child"])
    seen, stack = set(), set()

    def walk(n):
        if n in stack:
            raise AssertionError(f"cycle through {n}")
        if n in seen:
            return
        seen.add(n)
        stack.add(n)
        for m in adj.get(n, []):
            walk(m)
        stack.discard(n)

    for n in list(adj):
        walk(n)


def test_no_model_holds_two_positions(doc):
    from collections import Counter
    c = Counter(m["model_id"] for m in doc["models"])
    assert not [k for k, v in c.items() if v > 1], "duplicate model rows"


def test_excluded_rows_carry_a_reason_and_a_scope(doc):
    """`excluded` and `excluded until the repair pass` are different facts.

    Tonight needed that distinction twice: every exclusion looked permanent and
    every one turned out to be a torch version floor.
    """
    for m in doc["models"]:
        if m["status"] == "EXCLUDED":
            assert m["exclusion_reason"], f"{m['model_id']}: no reason"
            assert m["pending_repair"] is not None, f"{m['model_id']}: no scope"
            assert m.get("excluded_from"), f"{m['model_id']}: no pass scope"


def test_completeness_is_a_query(doc):
    """The claim the file exists to make answerable without anyone writing it."""
    in_spec = [m for m in doc["models"] if m["in_grid_spec"]]
    active = [m for m in in_spec if m["status"] == "ACTIVE"]
    excluded = [m for m in in_spec if m["status"] == "EXCLUDED"]
    #: **THE ROSTER IS A VIEW AND THE RECEIPT IS HISTORY ([5296]).** `phi-4` and
    #: `phi-4-reasoning` are named by `grid_roster.json` -- the v3 grid really
    #: did ask them -- and were later removed because neither has a base model,
    #: so no base->aligned pair exists (RH, 2026-08-10). The receipt is not
    #: rewritten to hide that; the rows carry status REMOVED instead.
    #:
    #: So the denominator of a COVERAGE question is the roster rows still on the
    #: roster, not every name the receipt mentions. The old form asserted
    #: active+excluded == in_spec and would now read the deliberate removal as
    #: two missing answers.
    history = [m for m in in_spec if m["status"] in ("REMOVED", "REPLACED", "GATED")]
    assert len(in_spec) == 103
    assert len(active) + len(excluded) + len(history) == len(in_spec)
    assert all(m["exclusion_reason"] for m in history), (
        "a removed name without a reason is indistinguishable from an oversight")
    # every exclusion in this pass is the torch floor and every one is repairable
    assert all(m["pending_repair"] for m in excluded)


def test_status_is_roster_scoped_and_never_asked_is_not_an_answer(doc):
    """A model the grid never asked cannot hold an answer about coverage.

    Nine registry rows sit outside the v3 roster and kept the constructor's default
    ACTIVE until 2026-07-31, so a registry-wide `status == "ACTIVE"` returned 102
    while 93 models were scored. The buckets summed to the whole the entire time —
    a partition test passes over this happily, because a partition is not a
    semantics. The invariant that catches it is the SCOPE:

        ACTIVE or EXCLUDED  =>  in_grid_spec

    and it holds at any roster size, so it cannot go stale and cannot fire when the
    remaining ten are finally scored.
    """
    HISTORY = ("REMOVED", "REPLACED", "GATED")
    for m in doc["models"]:
        if m["status"] in ("ACTIVE", "EXCLUDED"):
            assert m["in_grid_spec"], (
                f"{m['model_id']} is {m['status']} but was never on the roster; "
                "never-asked is not answered")
        elif m["status"] in HISTORY:
            #: NAME HISTORY IS SCOPE-FREE ([5296]). A removed name may or may not
            #: have been on the roster -- phi-4 was, gpt-sw3-6.7b was not -- so
            #: `in_grid_spec` is whatever the receipt says and is not asserted.
            #: What IS required is the reason, because that is the only thing
            #: separating "considered and dropped" from "quietly vanished".
            assert m["exclusion_reason"], f"{m['model_id']}: {m['status']} without a reason"
        else:
            assert m["status"] == "NOT_IN_GRID", f"{m['model_id']}: {m['status']}"
            assert not m["in_grid_spec"]

    # COVERAGE IS PRESENT ON EVERY ROW, so absence can never be read as zero. The
    # off-roster rows had no `cells_in_store` key at all, which is how a consumer
    # gets a KeyError where it wanted a nought.
    assert all("cells_in_store" in m for m in doc["models"])
    #: **OFF-ROSTER CELLS ARE LABELLED, NOT FORBIDDEN.** This asserted
    #: `cells_in_store == 0` for every NOT_IN_GRID row, on the reasoning that a
    #: model nobody asked cannot have been answered. The store says otherwise:
    #: 19 off-roster rows hold cells (salamandra, Lucie, gemma-2, granite, jais,
    #: llm-jp, the 70B pair, Teuken, croissant, recurrentgemma), and phi-4 holds
    #: 2,647 while being deliberately REMOVED. **The stash is the population and
    #: the roster is a view of it**, so data outside the view is the normal
    #: state, not a contradiction. What must never happen is a row whose
    #: coverage is unreadable -- hence the key check above, which is the real
    #: invariant this block was reaching for.
    assert all(isinstance(m["cells_in_store"], int) for m in doc["models"]), (
        "a row's coverage is not a number; absence would be read as zero")


def test_measured_fields_name_their_producers(doc):
    """A measured field whose producer is a shell history is not measured."""
    prov = doc["_provenance"]["measured_from"]
    for k, v in prov.items():
        assert v["producer"].startswith("scripts/"), k
        assert os.path.exists(os.path.join(HERE, v["producer"])), v["producer"]
        assert os.path.exists(os.path.join(HERE, v["artifact"])), v["artifact"]
        assert v["rows"] > 0, f"{k}: producer artifact is empty"


def test_weights_format_mixed_records_which_index(doc):
    """`mixed` alone cannot explain a load failure; the index decides.

    mistral-7b-sft-beta publishes both formats and only the .bin index, so it
    falls back and is refused while usable safetensors sit unreachable.
    """
    for m in doc["models"]:
        if m.get("weights_format") == "mixed":
            assert isinstance(m.get("index_present"), bool), m["model_id"]


def test_stage_is_never_guessed_for_unknown_procedures(doc):
    """A guessed `dpo` is worse than an honest label.

    One-step families whose procedure is not public must not acquire a
    procedure name by inference; an analysis branching on stage would treat
    the guess as a fact.
    """
    declared = set(doc["_schema"]["fields"]["stage"]["values"])
    for m in doc["models"]:
        assert m["stage"] in declared or m["stage"] == "", m["model_id"]


# ── environment-conditional load facts ────────────────────────────────────

ENVPATH = os.path.join(HERE, "data", "model_load_environments.json")


@pytest.fixture(scope="module")
def envdoc():
    if not os.path.exists(ENVPATH):
        pytest.skip("model_load_environments.json not present")
    with open(ENVPATH) as fh:
        return json.load(fh)


def test_every_observation_names_a_declared_environment(envdoc):
    """An outcome without an environment is not an observation.

    AmberSafe failed and then loaded on the SAME BOX, so the model id alone
    cannot key the fact.
    """
    envs = set(envdoc["environments"])
    for o in envdoc["observations"]:
        assert o["environment"] in envs, o


def test_load_observations_reference_real_models(envdoc, ids):
    for o in envdoc["observations"]:
        assert o["model_id"] in ids, o["model_id"]
    for m in envdoc["predicted_untested"]["model_ids"]:
        assert m in ids, m


def test_predictions_agree_with_the_weights_audit(envdoc):
    """The predicted-untested list is derived, so it must match its producer."""
    import csv as _csv
    with open(os.path.join(HERE, "data", "weights_audit.csv")) as fh:
        w = {r["model"]: r for r in _csv.DictReader(fh)}
    for m in envdoc["predicted_untested"]["model_ids"]:
        assert w[m]["needs_torch"] == "2.6", m


def test_torch_floor_population_is_fully_accounted(envdoc):
    """Every model the audit says needs torch>=2.6 is observed OR predicted.

    Absence of an observation must never read as success.
    """
    import csv as _csv
    with open(os.path.join(HERE, "data", "weights_audit.csv")) as fh:
        need = {r["model"] for r in _csv.DictReader(fh)
                if r["needs_torch"] == "2.6"}
    seen = {o["model_id"] for o in envdoc["observations"]
            if o["outcome"] == "load_failed"}
    seen |= set(envdoc["predicted_untested"]["model_ids"])
    assert need <= seen, f"unaccounted: {sorted(need - seen)}"


# ── defects lacan found by building against the file ──────────────────────

def test_edge_type_agrees_with_the_child_node(doc):
    """An edge must not assert a method its own endpoint denies.

    kto/ppo/slic arms travelled as `dpo_of` while their `stage` said otherwise
    -- three edges contradicting the nodes they connect.
    """
    stage = {m["model_id"]: m["stage"] for m in doc["models"]}
    METHOD = {"sft_of", "dpo_of", "kto_of", "ppo_of", "slic_of", "rlvr_of"}
    for e in doc["relations"]:
        if e["relation"] in METHOD:
            assert e["relation"] == f"{stage[e['child']]}_of", (
                f"{e['relation']} points at a node whose stage is "
                f"{stage[e['child']]}: {e}")


def test_params_match_the_base_across_training_edges(doc):
    """A 2.8B base cannot have 8B children.

    `archangel_sft-kto_pythia2-8b` IS pythia-2.8b -- the decimal point is
    written as a hyphen -- and a naive parse gave 8B for five rows. This finds
    that class without anyone reading the values.
    """
    pb = {m["model_id"]: m.get("params_b") for m in doc["models"]}
    TRAIN = {"sft_of", "dpo_of", "kto_of", "ppo_of", "slic_of", "rlvr_of",
             "instruct_of"}
    for e in doc["relations"]:
        if e["relation"] in TRAIN:
            a, b = pb.get(e["parent"]), pb.get(e["child"])
            if a and b:
                assert abs(a - b) < 1e-6, (
                    f"{e['relation']}: parent {a}B vs child {b}B -- "
                    f"training does not change parameter count ({e})")


SCALE_RELS = ("smaller_sibling_of", "smaller_predecessor_of")


def test_scale_ladder_relation_exists(doc):
    """Ordered in the founding commit and absent from the first build.

    It carries the scale question: one org, one recipe, three sizes. SPLIT at
    [1116].2 — see the next test for why one relation could not carry it.
    """
    lad = [e for e in doc["relations"] if e["relation"] in SCALE_RELS]
    assert lad, "the scale-ladder relations were ordered and are missing"
    assert not [e for e in doc["relations"] if e["relation"] == "smaller_version_of"], (
        "smaller_version_of is retired; it conflated a scale contrast with a "
        "generation change")
    pb = {m["model_id"]: m.get("params_b") for m in doc["models"]}
    for e in lad:
        a, b = pb.get(e["parent"]), pb.get(e["child"])
        if a and b:
            assert a < b, f"{e['relation']} points the wrong way: {e}"


def test_a_scale_edge_never_crosses_a_generation(doc):
    """"Same family, different SCALE" may not be drawn from a different RELEASE.

    `smaller_version_of` emitted 18 edges over two ladders, and one ladder crossed
    OLMo-2 to OLMo-3 — a different pretraining corpus and post-training recipe, a
    year apart. Counting family pairs off it gave 6 where the clause's own words
    allow 4, which is the difference between a claim proceeding and a claim being
    ruled underpowered ([1114]/[1116].1).

    The invariant is the SPLIT, not either count: a sibling edge has ONE
    generation on both ends, a predecessor edge has TWO, and neither may be
    empty — an undeclared generation is what let two of them sit on one ladder
    unremarked. It holds at any roster size.
    """
    gen = {m["model_id"]: m.get("generation", "") for m in doc["models"]}
    seen = set()
    for e in doc["relations"]:
        if e["relation"] not in SCALE_RELS:
            continue
        seen.add(e["relation"])
        g0, g1 = gen.get(e["parent"], ""), gen.get(e["child"], "")
        assert g0 and g1, (
            f"{e}: a scale edge with an undeclared generation — absence must "
            "never be read as 'the same release'")
        if e["relation"] == "smaller_sibling_of":
            assert g0 == g1, f"sibling edge crosses a generation: {e} ({g0} vs {g1})"
        else:
            assert g0 != g1, (
                f"predecessor edge does not cross a generation: {e} — it is a "
                "sibling and belongs under the other relation")
    assert seen == set(SCALE_RELS), (
        f"only {seen} present; if a ladder lost its generation split the "
        "conflation is back")


def test_every_relation_type_states_how_to_read_it(doc):
    """A direction flag needs a reader to remember what 'forward' means.

    lacan's two founding examples required OPPOSITE readings of one convention,
    so each type carries a sentence naming both roles.
    """
    reads = doc["_schema"]["relations"]["reads"]
    used = {e["relation"] for e in doc["relations"]}
    for r in used:
        assert r in reads, f"{r} has no reads sentence"
        assert "{parent}" in reads[r] and "{child}" in reads[r], reads[r]


def test_symmetric_relations_are_symmetric(doc):
    sym = set(doc["_schema"]["relations"].get("symmetric", []))
    pairs = {(e["parent"], e["child"]) for e in doc["relations"]
             if e["relation"] in sym}
    for a, b in pairs:
        assert (b, a) in pairs or a < b, (
            f"symmetric relation stored one-way and unordered: {a} {b}")


def test_index_present_is_a_boolean_not_a_string(doc):
    """One field, two types, waiting: `if row["index_present"]` was True for
    the STRING "false"."""
    for m in doc["models"]:
        v = m.get("index_present")
        assert v is None or isinstance(v, bool), (m["model_id"], type(v))


def test_every_scale_ladder_candidate_is_declared_or_excluded(doc):
    """A RELATION CAN BE WRONG BY SILENCE ([1122].4).

    `falcon3-7b` was complete in the store, same generation as its three siblings,
    and simply absent from the declared ladder. Every assertion here checked the
    edges that WERE drawn — direction, endpoints, generation consistency — and none
    could catch an edge that was never drawn. A dormant clause's revival turned on
    the resulting pair count.

    The fix is a completeness assertion over the DECLARED set: any group of families
    sharing an org and a declared generation, at two or more parameter counts, is
    either on a ladder or named in NOT_A_SCALE_RUNG. "Considered and excluded" and
    "never noticed" must be different states, which is exactly what the silence
    destroyed.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bmr", os.path.join(HERE, "scripts", "build_model_registry.py"))
    bmr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bmr)

    on_ladder = {k for lad in bmr.SCALE_LADDERS for k, _ in lad}
    fams = {}
    for m in doc["models"]:
        f = m.get("family")
        if f:
            fams.setdefault(f, {"org": m.get("org"), "gen": m.get("generation", ""),
                                "params": set()})["params"].add(m.get("params"))
    missing = []
    for a, b in itertools.combinations(sorted(fams), 2):
        ia, ib = fams[a], fams[b]
        if ia["org"] != ib["org"]:
            continue
        if ia["params"] == ib["params"]:          # not a scale contrast
            continue
        if a in on_ladder and b in on_ladder:
            continue
        if a in bmr.NOT_A_SCALE_RUNG or b in bmr.NOT_A_SCALE_RUNG:
            continue
        missing.append(f"{a}/{b}")
    assert not missing, (
        "same-org multi-scale family groups neither on a ladder nor declared "
        f"NOT_A_LADDER: {sorted(missing)} — add the rung or state why it is not one")
