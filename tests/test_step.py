"""Tests for Step and Cell. Each was watched to fail."""
import pytest

from malign_logits.checkpoint import Checkpoint
from malign_logits.movement import CANONICAL, DRAW
from malign_logits.step import Step

AMBER, CHAT, SAFE = "LLM360/Amber", "LLM360/AmberChat", "LLM360/AmberSafe"
ANGRY = "She was so angry she wanted to"


def _step():
    s = Step(Checkpoint(AMBER), Checkpoint(CHAT))
    if not s.prompts:
        pytest.skip("amber cells not in the store")
    return s


def test_direction_distinguishes_reverse_from_lateral():
    """A bare forward/reverse flag would erase the difference. kto against dpo is not a
    step backwards through training -- the two are alternatives at one point, and calling
    that 'reverse' would invent an ordering the training never had."""
    assert Step(Checkpoint(AMBER), Checkpoint(CHAT)).direction == "forward"
    assert Step(Checkpoint(CHAT), Checkpoint(AMBER)).direction == "reverse"
    lat = Step(Checkpoint("ContextualAI/archangel_sft-kto_pythia2-8b"),
               Checkpoint("ContextualAI/archangel_sft-dpo_pythia2-8b"))
    assert lat.direction == "lateral"


def test_a_reverse_step_is_allowed_and_stamped():
    """Teacher-forcing base->sft then sft->base is real work here, so refusing would
    block the experiment. The guard is that every result carries its direction."""
    s = Step(Checkpoint(CHAT), Checkpoint(AMBER))
    if not s.prompts:
        pytest.skip("amber cells not in the store")
    m = s.cell(ANGRY).movement(CANONICAL)
    assert m is not None
    assert m.diagnostics["direction"] == "reverse"


def test_label_is_derived_from_the_endpoints_not_declared():
    assert _step().label == "base->sft"
    assert Step.of("amber", "ego", "superego").label == "sft->dpo"


def test_prompts_is_the_intersection_never_one_arms_list():
    """Taking either arm alone measures a different population — the defect that dropped
    65% of amber's cells by iterating a registry instead of the store."""
    s = _step()
    from malign_logits.step import _scored
    a, b = _scored(AMBER), _scored(CHAT)
    assert set(s.prompts) == a & b
    assert len(s.prompts) <= min(len(a), len(b))


def test_cell_carries_its_stratification():
    """cell.prompt.domain is one attribute away, so stratifying before the statistic is
    the path of least resistance rather than a rule someone remembers."""
    c = _step().cell(ANGRY)
    assert c.is_present
    assert c.domain == "violence" and c.language == "en"
    assert c.record()["domain"] == "violence"


def test_the_rule_is_named_at_the_call_site_and_the_rules_disagree():
    c = _step().cell(ANGRY)
    canon, draw = c.movement(CANONICAL), c.movement(DRAW)
    assert len(draw.risers) > len(canon.risers), (
        "DRAW has no renormalisation null so it must admit more risers; if not, the "
        "fixture no longer distinguishes the rules")
    assert canon.diagnostics["rule"] == "canonical"


def test_mixed_rule_version_raises_rather_than_being_averaged():
    """A v1 arm against a v3 arm books an instrument change as training movement."""
    from malign_logits import cell as cellmod
    c = _step().cell(ANGRY)
    c.__dict__["pre"] = type(c.post)(probs={"a": 0.5}, residual=0.5, rule_version=1)
    with pytest.raises(ValueError, match="rule_version mismatch"):
        c.movement(CANONICAL)
    assert c.movement(CANONICAL, allow_mixed=True) is not None


def test_chain_gives_the_sequence_the_relations_cannot():
    """The registry's edges are star-shaped from the base, so the SFT arm is invisible
    to a traversal even though it is a declared node."""
    labels = [s.label for s in Step.chain("amber")]
    assert labels == ["base->sft", "sft->dpo"]
