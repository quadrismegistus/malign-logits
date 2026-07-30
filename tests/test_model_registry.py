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
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "data", "model_registry.json")
SPEC = os.path.join(HERE, "data", "grid_spec.json")


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
    for field in ("position", "architecture", "stage", "weights_format", "status"):
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
        assert e["source"] in ids, f"dangling source {e['source']}"
        assert e["target"] in ids, f"dangling target {e['target']}"


def test_relation_types_are_declared(doc):
    declared = set(doc["_schema"]["relations"]["values"])
    used = {e["relation"] for e in doc["relations"]}
    assert used <= declared, f"undeclared relation types: {used - declared}"


def test_relation_direction_is_stated(doc):
    """An edge label without a direction convention invites a silent inversion."""
    assert doc["_schema"]["relations"].get("direction")


def test_hierarchical_relations_are_acyclic(doc):
    """A model cannot be its own training ancestor."""
    hier = {"sft_of", "dpo_of", "rlvr_of", "reasoning_of"}
    adj = {}
    for e in doc["relations"]:
        if e["relation"] in hier:
            adj.setdefault(e["source"], []).append(e["target"])
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
    assert len(in_spec) == 103
    assert len(active) + len(excluded) == len(in_spec)
    # every exclusion in this pass is the torch floor and every one is repairable
    assert all(m["pending_repair"] for m in excluded)


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
            assert m.get("index_present") in ("true", "false"), m["model_id"]


def test_stage_is_never_guessed_for_unknown_procedures(doc):
    """A guessed `dpo` is worse than an honest label.

    One-step families whose procedure is not public must not acquire a
    procedure name by inference; an analysis branching on stage would treat
    the guess as a fact.
    """
    declared = set(doc["_schema"]["fields"]["stage"]["values"])
    for m in doc["models"]:
        assert m["stage"] in declared or m["stage"] == "", m["model_id"]
