"""`repo@revision` must reach the STORES and never the Hub.

M05 names its 95 checkpoints `repo@revision` so ClickHouse's ReplacingMergeTree,
which dedupes on a sorting key containing `model`, keeps them apart. Docket
[5398]/[5400]: RH's fix, after two seats had scoped a sorting-key migration on 47M
live rows for what is a naming problem.

**EVERY FAILURE THIS FILE GUARDS IS SILENT.** None of them raises on fleet day:

    from_pretrained("org/repo@step1000")   asks the Hub for a repo that does not
                                           exist -- this one is loud, and it is the
                                           only loud one
    LOADER_OVERRIDE.get("org/repo@rev")    misses, so a tokenizer override does not
                                           fire and the cells are subtly wrong
    purge_model("org/repo@rev")            matches no hub directory and collects
                                           NOTHING, printing nothing, on a run whose
                                           95 checkpoints are ~1.3 TB
    lineage_of("org/repo@rev")             absent from the map; consumers that
                                           default an unmapped model to its own
                                           lineage read 95 checkpoints of 4 repos as
                                           95 INDEPENDENT LINEAGES

The last is the one that reaches a published number, and it inflates n in the
direction that flatters.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits.lineage import base_model_of, revision_of

CKPT = "allenai/Olmo-3-1025-7B@stage1-step1000"
REPO = "allenai/Olmo-3-1025-7B"


def test_split_round_trip():
    assert base_model_of(CKPT) == REPO
    assert revision_of(CKPT) == "stage1-step1000"


def test_bare_id_is_unchanged():
    """The 400,644 cells written before this convention keep their keys."""
    assert base_model_of(REPO) == REPO
    assert revision_of(REPO) is None


def test_revision_may_contain_hyphens_and_digits():
    m = "allenai/Olmo-3-7B-Think-SFT@1e-4-step1000"
    assert base_model_of(m) == "allenai/Olmo-3-7B-Think-SFT"
    assert revision_of(m) == "1e-4-step1000"


def test_purge_strips_the_suffix_and_is_not_a_silent_noop(tmp_path, monkeypatch, capsys):
    """The disk failure: purge that matches nothing, prints nothing, frees nothing."""
    from malign_logits import twp

    hub = tmp_path / "hub"
    d = hub / "models--allenai--Olmo-3-1025-7B"
    (d / "blobs").mkdir(parents=True)
    (d / "blobs" / "w.bin").write_bytes(b"x" * 16)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(hub)
                        if p.endswith("huggingface/hub") else os.path.expanduser(p))

    twp.purge_model(CKPT)
    assert not d.exists(), "a checkpoint id must purge its repo's cache"
    assert "purged" in capsys.readouterr().out


def test_purge_announces_when_nothing_matched(tmp_path, monkeypatch, capsys):
    """Success and failure printed the same nothing; on a disk-bound run that is
    the whole outcome."""
    from malign_logits import twp

    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(hub)
                        if p.endswith("huggingface/hub") else os.path.expanduser(p))
    twp.purge_model("nobody/nothing@rev")
    assert "nothing matched" in capsys.readouterr().out


def test_load_tokenizer_never_hands_the_suffix_to_huggingface(monkeypatch):
    """The suffix is ours. The Hub has no such repo."""
    from malign_logits import twp

    seen = {}

    class FakeTok:
        @classmethod
        def from_pretrained(cls, mid, **kw):
            seen["mid"], seen["revision"] = mid, kw.get("revision")
            return "tok"

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTok)
    twp.load_tokenizer(CKPT)
    assert seen["mid"] == REPO, f"HuggingFace was asked for {seen['mid']!r}"
    assert seen["revision"] == "stage1-step1000"


def test_explicit_revision_wins_over_the_suffix(monkeypatch):
    """A caller that already resolved a registry pin keeps its behaviour."""
    from malign_logits import twp

    seen = {}

    class FakeTok:
        @classmethod
        def from_pretrained(cls, mid, **kw):
            seen["mid"], seen["revision"] = mid, kw.get("revision")
            return "tok"

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTok)
    twp.load_tokenizer(CKPT, revision="pinned-sha")
    assert seen["revision"] == "pinned-sha"


def test_lineage_of_answers_at_the_repo_grain():
    """Ruled [5402]: STRIP, do not raise. Raising is only a guard if callers
    propagate it, and [5384] measured that they do not -- they default an unmapped
    model to its own lineage, so raising CAUSES the inflation it looks like it
    prevents."""
    from malign_logits.lineage import UnmappedModel, lineage_of

    try:
        want = lineage_of(REPO)
    except UnmappedModel:
        import pytest
        pytest.skip("%s is not in the lineage map on this checkout" % REPO)
    assert lineage_of(CKPT) == want

    #: and stripping does not weaken the guard: an unknown REPO still refuses
    try:
        lineage_of("nonexistent/model@step1")
    except UnmappedModel:
        pass
    else:
        raise AssertionError("an unknown repo must still raise")


def test_no_hand_rolled_at_splits_outside_the_helpers():
    """[5402].2 -- six call sites is six chances to write `split("@")[-1]`."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    allowed = {root / "malign_logits" / "lineage.py", pathlib.Path(__file__).resolve()}
    pat = re.compile(r"""split\(\s*['"]@['"]""")
    bad = []
    for p in list((root / "malign_logits").rglob("*.py")) + \
             list((root / "scripts").rglob("*.py")):
        if p.resolve() in allowed:
            continue
        for i, ln in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            #: comments and docstrings NAME the forbidden form in order to forbid
            #: it -- the first version of this guard flagged its own rationale
            if pat.search(ln) and not ln.lstrip().startswith(("#", "*", '"', "'")):
                bad.append(f"{p.relative_to(root)}:{i}")
    assert not bad, ("parse `repo@revision` with base_model_of/revision_of, "
                     "not by hand: " + ", ".join(bad))
