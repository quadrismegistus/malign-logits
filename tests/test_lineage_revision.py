"""The `repo@revision` convention: stripping, refusing, and the only two parses.

Registrar's condition on [5402]. M05 writes 95 checkpoints as `repo@revision`
so ClickHouse's ReplacingMergeTree -- which dedupes on a sorting key containing
`model` -- keeps them apart. Every test here is a failure that would NOT raise
on fleet day.
"""
import pathlib
import re

import pytest

from malign_logits.lineage import (REV_SEP, UnmappedModel, base_model_of,
                                   base_of, lineage_of, revision_of)

REPO = "allenai/Olmo-3-1025-7B"
CKPT = REPO + REV_SEP + "stage1-step1000"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_a_checkpoint_has_its_repos_lineage():
    """**The inflation guard.** An unstripped id@rev is absent from
    model_to_lineage, and consumers that treat an unmapped model as its OWN
    lineage (registration_e does, in so many words) would read 95 checkpoints of
    4 repos as 95 INDEPENDENT LINEAGES -- silently, in the direction that
    flatters every finding."""
    assert lineage_of(CKPT) == lineage_of(REPO)


def test_an_unknown_repo_still_refuses():
    """Stripping must not weaken the guard: the stripped repo is still looked
    up, and a repo nobody mapped still raises rather than defaulting."""
    with pytest.raises(UnmappedModel):
        lineage_of("nonexistent/model" + REV_SEP + "step1")
    with pytest.raises(UnmappedModel):
        lineage_of("nonexistent/model")


def test_the_two_sanctioned_parses():
    assert base_model_of(CKPT) == REPO
    assert base_model_of(REPO) == REPO           # a bare id is unchanged
    assert revision_of(CKPT) == "stage1-step1000"
    assert revision_of(REPO) is None             # None MEANS the default
    assert base_of(CKPT) == base_of(REPO)


def test_a_revision_containing_the_separator_splits_once_only():
    """`split(REV_SEP, 1)`, not `split(REV_SEP)`. A revision is free to contain
    the separator; the repo is everything before the FIRST one."""
    odd = REPO + REV_SEP + "weird@rev"
    assert base_model_of(odd) == REPO
    assert revision_of(odd) == "weird@rev"


def test_no_hand_rolled_split_outside_the_helpers():
    """Six call sites is six chances to write `[-1]` instead of `[0]`.

    **The first version of this guard flagged its own rationale** -- a comment
    naming `split("@")` in order to forbid it -- so comment lines are ignored. A
    guard that fires on the documentation of the rule it enforces is a guard the
    next person in a hurry deletes.
    """
    pat = re.compile(r"""split\(\s*['"]@['"]""")
    offenders = []
    for f in sorted((ROOT / "malign_logits").rglob("*.py")):
        if f.name == "lineage.py":
            continue                      # the helpers are allowed to parse
        for n, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if pat.search(code):
                offenders.append("%s:%d" % (f.relative_to(ROOT), n))
    assert not offenders, (
        "hand-rolled revision parse outside lineage.py; use base_model_of() or "
        "revision_of(): %s" % offenders)
