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
    """Having the SHAPE a staged decomposition needs is not having the DATA. Counting
    the first and quoting it as the second overstates the roster by however many
    families are missing an arm.

    THIS TEST NO LONGER PINS EITHER COUNT, and the reason is a defect it had until
    2026-07-31. It asserted `len(three) > len(complete)` — i.e. that SOME family is
    incomplete — plus the literal pair 22/17. Both are roster-state, not invariants:

      the literal pair went stale the moment the repair pass recovered arms (17 -> 21),
      which is the shallow half; and

      the `>` would FAIL WHEN THE ROSTER FINALLY COMPLETES. A test that fires on success
      is worse than one that fires on staleness, because the correct response is to
      delete it and nobody deletes a red test without reading it first.

    What is invariant is that the two filters AGREE WITH THE EXCLUSION LEDGER: the gap
    between them is exactly the number of three-arm families carrying an excluded arm.
    That holds at a gap of five, of one, and of zero.

    Observed 2026-07-31 at 91,421 payloads / 93 models: three=22, complete=21, gap=1
    (was 22/17/5 pre-repair). Recorded, not asserted.
    """
    three = Family.all(is_three_arm=True)
    complete = Family.all(is_three_arm=True, is_complete=True)

    assert {f.key for f in complete} <= {f.key for f in three}, (
        "complete must be a SUBSET of three-arm; if it is not, `is_complete` is not "
        "narrowing the same population and the two filters do not compose")

    missing_an_arm = [f for f in three if f.excluded]
    assert len(three) - len(complete) == len(missing_an_arm), (
        f"the gap between three-arm ({len(three)}) and complete ({len(complete)}) must "
        f"equal the three-arm families with an excluded arm ({len(missing_an_arm)}). "
        f"A mismatch means `is_complete` and the exclusion ledger disagree about the "
        f"same fact, which is the defect this test exists to catch.")


def test_excluded_arms_surface_with_a_reason_and_a_repair_flag():
    """"Excluded" and "excluded until the repair pass" are different facts."""
    bad = [f for f in Family.all() if f.excluded]
    if not bad:
        # An empty ledger is a legitimate END STATE, not a missing artifact. This used to
        # `assert bad`, which would have failed on the day the roster completed -- the
        # same fire-on-success defect corrected in
        # test_three_arm_and_complete_are_different_questions.
        pytest.skip("no family currently has an excluded arm; nothing to check the "
                    "shape of. Not a failure: the ledger is empty because the repair "
                    "pass succeeded.")
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
    """The subject is the FILTER, not the roster.

    This test asserted `counts("status")["EXCLUDED"] == 13` until 2026-07-31 — a roster
    fixture bolted onto an API test, off-topic for the thing being checked and stale the
    moment the repair pass scored those thirteen (13 -> 10). The structural claim it was
    reaching for is already made by
    test_excluded_arms_surface_with_a_reason_and_a_repair_flag.

    So the assertions here are now about filtering behaviour only, and none of them move
    when the roster does.
    """
    ssm = Checkpoint.all(architecture="ssm", in_grid_spec=True)
    assert ssm, "no SSM checkpoint in the spec; the architecture field or the spec moved"
    assert all(c.architecture == "ssm" for c in ssm), "a filter returned a non-match"

    # a filter must NARROW, and two filters must narrow at least as much as one
    all_ssm = Checkpoint.all(architecture="ssm")
    assert len(ssm) <= len(all_ssm)

    # every status bucket must be non-empty of the thing it names, and the buckets must
    # partition the roster -- a count that does not sum to the whole is a dropped row
    counts = Checkpoint.counts("status")
    for status, n in counts.items():
        got = Checkpoint.all(status=status)
        assert len(got) == n, f"counts({status})={n} but all(status={status}) gave {len(got)}"
    assert sum(counts.values()) == len(Checkpoint.all()), (
        "status counts do not sum to the roster; some checkpoint has a status the "
        "counter did not bucket")


def test_family_of_refuses_a_shared_base():
    """pythia-2.8b bases all four archangel families; picking one would be a coin flip."""
    assert Family.of("EleutherAI/pythia-2.8b") is None
    assert Family.of(KTO).key == "archangel-kto"
