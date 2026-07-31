"""Tests for malign_logits.prompts.

Each was watched to fail. The translation tests exist because the first implementation
trusted the `<id>_zh` naming convention and silently returned None for six rows.
"""
import pytest

from malign_logits.prompts import Prompt, PromptGroup, Prompts


def test_translation_survives_the_id_convention_having_exceptions():
    """380 of 386 Chinese rows are keyed `<english_id>_zh`. SIX ARE NOT, because the
    keying pass used the id of whichever TWIN it inherited from. `e6_water_M`'s Chinese
    row is `violence_explicit_5_zh`, so an id-only lookup finds nothing and an analysis
    trusting it reports no translation where one exists.
    """
    p = Prompt("e6_water_M")
    assert Prompt.find(p.text) is not None
    t = p.translation
    assert t is not None, "the group+role fallback must find the off-convention row"
    assert t.id == "violence_explicit_5_zh"
    assert t.language == "zh"
    assert t.group_role == p.group_role, "the fallback matches on ROLE, not position"


def test_translation_round_trip_lands_on_a_row_with_the_same_text():
    """Not necessarily the SAME row: a dual-identity prompt's Chinese counterpart points
    back at whichever twin it was keyed from. Identity of TEXT is the guarantee."""
    p = Prompt("e6_water_M")
    back = p.translation.english
    assert back is not None and back.text == p.text


def test_partner_refuses_to_guess_in_a_triple():
    """Two members: unambiguous. Three: silently picking one of two poles is how sign
    errors happen, so it returns None and the caller must use .group.

    The two-member example was `e6_water_M` until 2026-07-31, when the cessation set
    retired. A FIXTURE NAMING A SPECIFIC PROMPT IS A BET THAT THE PROMPT SURVIVES —
    the subject here is partner resolution, not that prompt, so it now names a live
    pair and the assertion is about the API rather than the catalogue.
    """
    assert Prompt("e1_bill_M").partner.id == "e1_bill_U"
    triple = PromptGroup("f11_love")
    assert len(triple) > 2
    assert triple.members[0].partner is None


def test_lookup_by_id_not_by_text_and_duplicates_are_visible():
    """61 texts carry more than one row. A text-keyed dict takes whichever came last --
    the defect that reported 48 group disagreements where the true figure was 1."""
    d = Prompt.find("My boss takes credit for my work in every meeting. I should")
    assert d.duplicates, "this text is dual-identity; if not, the fixture is stale"
    assert Prompt(d.id).id == d.id
    assert len(Prompts.matching(d.text)) == len(d.duplicates) + 1


def test_where_is_active_by_default_and_says_so_when_asked_otherwise():
    """Forgetting the status filter once reported source=OTHER as 55 where it was 4."""
    active = Prompts.where(source="OTHER")
    everything = Prompts.where(source="OTHER", status=None)
    assert len(everything) > len(active), (
        "if these are equal the fixture no longer distinguishes the default")
    assert all(p.is_active for p in active)


def test_a_logical_prompt_announces_that_its_text_is_not_feedable():
    b = Prompt("BOS")
    assert b.is_logical
    assert b.text.startswith("<<<LOGICAL:")
    assert not Prompt("e6_water_M").is_logical


def test_unknown_field_raises_instead_of_returning_none():
    """A typo should be a mistake, not a silent null."""
    with pytest.raises(AttributeError):
        Prompt("e6_water_M").doamin


def test_group_contrast_and_its_zh_image():
    g = PromptGroup("f11_love")
    assert g.contrast and "/" in g.contrast
    z = g.translation
    assert z is not None and z.id.endswith("_zh")
    assert g.roles == z.roles, "the Chinese image must carry the same role census"
