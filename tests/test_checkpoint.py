"""Tests for Checkpoint and Family. Each was watched to fail."""
import pytest

from malign_logits.checkpoint import Checkpoint
from malign_logits.family import Family

KTO = "ContextualAI/archangel_sft-kto_pythia2-8b"


def test_stage_separates_what_position_cannot():
    """THE SEPARATOR CELL. Four checkpoints share a base AND an SFT arm and differ only
    in preference method. `position` calls all four `superego`; if `stage` did too, the
    one cell that can distinguish "the first stage did the work" from "the second stage
    was weak" would be unaddressable."""
    stages, positions = set(), set()
    for k in ("archangel-dpo", "archangel-kto", "archangel-ppo", "archangel-slic"):
        f = Family(k)
        stages.add(f.preference.stage)
        positions.add("superego" if f.superego else None)
    assert stages == {"dpo", "kto", "ppo", "slic"}
    assert positions == {"superego"}, "position must NOT distinguish them; that is the point"


def test_dpo_is_none_where_the_method_is_not_dpo_and_preference_still_finds_it():
    """Asking by stage when you mean method is the reason both vocabularies exist."""
    f = Family("archangel-kto")
    assert f.dpo is None
    assert f.preference is not None and f.preference.stage == "kto"
    assert f.superego == f.preference


def test_three_arm_and_complete_are_different_questions():
    """22 families have the shape a staged decomposition needs; 5 lose an arm to a
    loader failure. Counting the first and quoting it as the second overstates the
    roster by five families."""
    three = Family.all(is_three_arm=True)
    complete = Family.all(is_three_arm=True, is_complete=True)
    assert len(three) > len(complete), (
        "if these are equal no family has an excluded arm and the fixture is stale")
    assert len(three) == 22 and len(complete) == 17


def test_excluded_arms_surface_with_a_reason_and_a_repair_flag():
    """"Excluded" and "excluded until the repair pass" are different facts."""
    bad = [f for f in Family.all() if f.excluded]
    assert bad, "no family has an excluded arm; the load-failure ledger is missing"
    cp, (reason, pending) = bad[0].excluded[0]
    assert reason, "an exclusion without a reason is an absence, not a record"


def test_checkpoint_reads_the_artifact_and_does_not_guess():
    cp = Checkpoint(KTO)
    assert cp.stage == "kto" and cp.position == "superego"
    assert cp.architecture == "transformer" and cp.vocab_size == 50254
    with pytest.raises(AttributeError):
        cp.archtiecture          # a typo must raise, not return None


def test_unregistered_checkpoint_is_known_false_rather_than_a_crash():
    cp = Checkpoint("not/a-real-model")
    assert cp.is_known is False
    assert cp.record == {}


def test_all_filters_on_any_registry_field():
    ssm = Checkpoint.all(architecture="ssm", in_grid_spec=True)
    assert len(ssm) == 4, "two pure-SSM families, base+aligned each"
    assert all(c.architecture == "ssm" for c in ssm)
    assert Checkpoint.counts("status")["EXCLUDED"] == 13


def test_family_of_refuses_a_shared_base():
    """pythia-2.8b bases all four archangel families; picking one would be a coin flip."""
    assert Family.of("EleutherAI/pythia-2.8b") is None
    assert Family.of(KTO).key == "archangel-kto"
