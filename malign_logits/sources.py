"""The payload source registry. ONE list, imported by both ingesters.

    from malign_logits.sources import twp_sources, resolve_source

    for path, label in twp_sources():   # ACTIVE only, in a stable order
        ...

WHY THIS EXISTS. `twp_ingest.py` took `--src <one directory>` chosen by the
operator per run; `ch_ingest.py` carried a hardcoded list of three. Nothing
enforced that the two covered the same set, and by 2026-08-10 they did not:
the stash held **307,891 cells and ClickHouse 273,723**, a gap of 38,451
concentrated in six `twp_*` directories one ingester had never heard of
([5297]).

Neither ingester was wrong. Each did what it was told. **The list was the
artifact nobody owned**, and a list that lives in an operator's memory diverges
silently the first time two operators remember differently.

RETIRED IS A STATUS, NOT A PREFIX CONVENTION. `RETIRED-20260803-falcon-nan`
holds the SAME two models, the SAME 5,166 cells and IDENTICAL filenames as
`falcon_h1_repair` -- it is the all-NaN Falcon shard from [3015] and the repair
supersedes it. Any rule of the form "ingest every directory containing twp
jsonl" pulls it back in, and because `source` is part of the twp key it would
sit alongside the repair rather than visibly overwriting it. So retirement is
declared here with its reason and its superseder, and `resolve_source` RAISES
on a retired path rather than skipping it: a silent skip and a silent include
are both ways of not being told.

PENDING IS ALSO A STATUS. A directory whose provenance has not been confirmed
by the seat that produced it is not ingested and is not quietly dropped. It is
listed, with the question attached.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (relative path, source label, status, note)
#:
#: `label` is what lands in the `source` column and in the stash key, so it is
#: part of the data's identity: **renaming a label re-partitions the store.**
#: Statuses: ACTIVE ingest it | RETIRED refuse it | PENDING ask first.
TWP_SOURCES = [
    #: M05 REVISION PRE-FLIGHT, 3 cells on a spare checkpoint. Registered rather
    #: than side-loaded: the point of the pre-flight is that the REAL ingest path
    #: keeps `repo@revision` apart, and an ingest bypassing this registry would
    #: test a path the fleet does not use. `stage1-step10000` is deliberately NOT
    #: one of M05's 43 revisions, so no pre-flight cell can collide with a fleet
    #: cell. Retire once the fleet has landed.
    ("data/raw/preflight_m05_revision", "preflight_m05_revision", "ACTIVE",
     "M05 repo@revision pre-flight: 3 cells, Olmo-3-1025-7B@stage1-step10000"),
    #: **THE M05 ACQUISITION FLEET, ONE LABEL PER BOX.** Five boxes each rsync
    #: into their own directory so two writers can never touch one filename --
    #: the collision that truncated 14 files in the beam campaign. They are
    #: declared here BEFORE the ingest runs, because a bare sweep visits the
    #: declared list and nothing else: an undeclared directory is not an error,
    #: it is 54,492 cells that silently do not exist. That is how a whole jais
    #: model pair went missing with 878 MB of `.f16` sitting on disk ([5315]).
    #:
    #: The model strings here carry `repo@revision` for all but the three mains.
    ("data/raw/m05/m05_0", "m05_0", "ACTIVE", "M05 acquisition fleet, shard 0"),
    ("data/raw/m05/m05_1", "m05_1", "ACTIVE", "M05 acquisition fleet, shard 1"),
    ("data/raw/m05/m05_2", "m05_2", "ACTIVE", "M05 acquisition fleet, shard 2"),
    ("data/raw/m05/m05_3", "m05_3", "ACTIVE", "M05 acquisition fleet, shard 3"),
    ("data/raw/m05/m05_4", "m05_4", "ACTIVE", "M05 acquisition fleet, shard 4"),
    #: **THE PYTHIA LADDER, ONE LABEL PER BOX.** A SEPARATE STUDY from M05
    #: (registrar [5425](b)): different lab, tokenizer and corpus, and its
    #: per-rung arm is pretraining only. Declared here before the ingest for
    #: the same reason the m05 shards were -- a bare sweep visits the declared
    #: list and nothing else, so an undeclared directory is not an error, it is
    #: cells that silently do not exist.
    ("data/raw/pythia/py_0", "py_0", "ACTIVE", "Pythia ladder, shard 0"),
    ("data/raw/pythia/py_1", "py_1", "ACTIVE", "Pythia ladder, shard 1"),
    ("data/raw/pythia/py_2", "py_2", "ACTIVE", "Pythia ladder, shard 2"),
    ("data/raw/pythia/py_3", "py_3", "ACTIVE", "Pythia ladder, shard 3"),
    ("data/raw/pythia/py_4", "py_4", "ACTIVE", "Pythia ladder, shard 4"),
    ("data/raw/cloud_run_20260801", "cloud_run_20260801", "ACTIVE",
     "the 2026-08-01 cloud run; the bulk of the store"),
    ("data/f11_twp", "f11_twp", "ACTIVE",
     "F11 twp payloads; SPLIT ACROSS TWO DIRECTORIES, see cache.logit_path"),
    ("data/f11_twp_delta", "f11_twp_delta", "ACTIVE",
     "the F11 delta run"),
    ("data/raw/falcon_h1_repair", "falcon_h1_repair", "ACTIVE",
     "supersedes RETIRED-20260803-falcon-nan for the two Falcon-H1-7B arms"),

    ("data/raw/RETIRED-20260803-falcon-nan", "falcon_nan", "RETIRED",
     "all-NaN Falcon-H1-7B shard [3015]. Same two models, same 5,166 cells and "
     "identical filenames as falcon_h1_repair, which supersedes it."),

    #: Confirmed at [5298] as completed cloud runs of 07-08 Aug, none scratch,
    #: none superseded. Their names are box/shard labels rather than experiment
    #: names, which is why they read as temporary and were not in any registry.
    ("data/raw/twp_twp_00", "twp_twp_00", "ACTIVE", "3 boxes of one sharded run, 08-07"),
    ("data/raw/twp_twp_01", "twp_twp_01", "ACTIVE",
     "628 cells absent from the stash are NOT a partial run: four models whose "
     "prompts were refused at encoding, 572 of them Teuken's two arms at "
     "exactly 314 each -- the documented tokenizer mangling. Re-running "
     "reproduces the same skips; the identical 314 on both arms is the tell."),
    ("data/raw/twp_twp_02", "twp_twp_02", "ACTIVE", "3 boxes of one sharded run, 08-07"),
    ("data/raw/twp_w2_gpu2", "twp_w2_gpu2", "ACTIVE", "second wave, 08-07"),
    ("data/raw/twp_w2_jais", "twp_w2_jais", "ACTIVE", "the jais pair, needed its own box"),
    ("data/raw/twp_lineages_v2", "twp_lineages_v2", "ACTIVE", "08-07, one file"),

    #: ── THE CENSUS, EIGHT PER-BOX SUBDIRECTORIES ONE LEVEL DOWN ──────
    #:
    #: **NAMED, NEVER GLOBBED.** `twp_fill_rsync_loop.sh` puts one destination
    #: per box deliberately -- four loops into one directory is four writers on
    #: one filename the moment two boxes touch the same model. So a scan over
    #: `data/raw/*/` sees the whole census as ONE 536-cell directory, which is
    #: exactly how ~100,000 cells stayed invisible to one store ([5298]). A
    #: convention about nesting is a convention until something enforces it.
    ("data/raw/twp_fill/twpfill0", "twpfill0", "ACTIVE", ""),
    ("data/raw/twp_fill/twpfill1", "twpfill1", "ACTIVE", ""),
    ("data/raw/twp_fill/twpfill2", "twpfill2", "ACTIVE", ""),
    ("data/raw/twp_fill/twpfill3", "twpfill3", "ACTIVE",
     "holds the COMPLETE kanana-1.5-8b-instruct-2505 battery (2,579 cells); "
     "supersedes twpfill0's 364-cell partial on those cells [5304]"),
    ("data/raw/twp_fill/twpilm", "twpilm", "ACTIVE", "the internlm2 lineage, own box"),
    ("data/raw/twp_fill/twpssm", "twpssm", "ACTIVE", ""),
    ("data/raw/twp_fill/twp70b", "twp70b", "ACTIVE", "the 70B pair"),

    ("data/raw/twp_fill/_unsharded_box0", "unsharded_box0", "RETIRED",
     "the PRE-SPLIT unsharded run. 4,406 cells, ALL 4,406 also present in a "
     "sharded box, ZERO unique -- measured, not assumed [5303]. Excluding it "
     "loses nothing and removes it from three of the four collision groups. "
     "Not a precedence question: a fallback implies it covers something, and "
     "it covers nothing."),
    ("data/raw/twp_fill", "twp_fill_loose", "RETIRED",
     "the one loose google__recurrentgemma-9b-it.jsonl at the top of the "
     "census directory, 536 cells from the pre-split run and a SUBSET of "
     "_unsharded_box0, which is itself a subset of the shards. Doubly "
     "subsumed. `source` is in the twp key, so ingesting it would sit a third "
     "copy beside the others rather than overwrite [5299]."),
    #: **THE VERSE FLEET, ONE LABEL FOR THE MERGED TREE.** 250 rungs, 455,000
    #: rows, 13-14 Aug 2026, eight boxes. ONE label rather than M05's one-per-box:
    #: that pattern exists so two LIVE writers cannot touch one filename, and
    #: these instances are destroyed. Per-box provenance is preserved in the
    #: unmerged `data/raw/verse_fleet/<instance_id>/` directories.
    #:
    #: **The merged tree is hard links SELECTED ON ROW COUNT (== 1,820), and that
    #: selection is the point rather than a tidying step.** Three partial files
    #: sit in the unmerged tree -- a 959-row `pythia-6.9b@step8000` from the box
    #: whose four-hour rental window expired mid-write (runbook 2.27), and two
    #: more from a window when a duplicate assignment briefly had two boxes on
    #: the same models. A walk taking the first file it found would ingest a
    #: partial over a complete rung, which is the shape that cost
    #: `opening_matched` its construction ([5874]). Rebuild with
    #: `scripts/verse_fleet_merge.py`; it REFUSES if any declared rung lacks a
    #: complete file.
    #:
    #: Prompt fidelity verified before declaring: all 456,130 records, all 1,786
    #: distinct prompts verbatim against the specs, max length 297 -- this runner
    #: writes `prompt` as its own field rather than inside an id.
    ("data/raw/verse_fleet_merged", "verse_fleet", "ACTIVE",
     "Verse fleet: 250 rungs merged from 8 boxes, complete files only (1,820 rows)"),
]


class RetiredSource(Exception):
    """Raised when a retired directory is asked for. Never silently skipped."""


def twp_sources(status="ACTIVE", absolute=True):
    """[(path, label)] for the given status, in declaration order.

    Declaration order, not sorted order: the order sources are ingested in
    decides which one a merge keeps when a cell appears in two, so it is part
    of the protocol rather than a presentation detail.
    """
    out = []
    for rel, label, st, _note in TWP_SOURCES:
        if st != status:
            continue
        out.append((os.path.join(ROOT, rel) if absolute else rel, label))
    return out


def resolve_source(path_or_label):
    """(path, label) for one source. RAISES on retired, KeyError on unknown."""
    key = str(path_or_label).rstrip("/")
    base = os.path.basename(key)
    for rel, label, st, note in TWP_SOURCES:
        if key in (rel, label) or base in (os.path.basename(rel), label):
            if st == "RETIRED":
                raise RetiredSource(
                    "%s is RETIRED and must not be ingested: %s" % (label, note))
            return os.path.join(ROOT, rel), label
    raise KeyError(
        "%r is not a declared payload source. Add it to TWP_SOURCES with a "
        "status and a note; an undeclared source is how the two stores "
        "diverged by 38,451 cells." % path_or_label)


def status_table():
    """One line per source, for a run log. Printed, not inferred."""
    w = max(len(r[1]) for r in TWP_SOURCES)
    return "\n".join(
        "  %-9s %-*s  %s" % (st, w, label, note)
        for _rel, label, st, note in TWP_SOURCES)


def payload_files(path, ext=".f16"):
    """Real payloads in one source directory. Zero-length files are NOT payloads.

    **A ZERO-BYTE `.f16` IS NOT A CANDIDATE**, and there are 20 of them across
    the declared sources. Eight of the ten unresolved logit collisions at
    [5321] were a real payload against one of these, which is not a contest
    between two observations -- it is a file that holds nothing.

    THE RULE LIVES HERE RATHER THAN IN THE RESOLVER, deliberately. A zero-byte
    file should be invisible to the INDEXER too, not only to the tie-break;
    otherwise the indexer keeps offering candidates the resolver keeps
    rejecting and the two disagree about what exists. It is a property of the
    file, not of a collision.

    TWO CAUSES, ONE REMEDY, AND THEY LOOK IDENTICAL FROM THE FILESYSTEM:

        aborted write    Pharia, Baichuan2, internlm2 -- the model failed to
                         load, so nothing was ever written
        refusal that     the two Falcon Mamba instruct arms -- the model loaded
        worked           and the tokenizer round-trip guard refused all 12
                         prompts offered, and the runner had already opened the
                         file

    The second is a guard doing its job and leaving an artifact that looks like
    a crash. Distinguishing them matters for what you go and fix; it does not
    change what to do with the file.
    """
    import glob as _glob
    import os as _os
    out = []
    for f in sorted(_glob.glob(_os.path.join(path, "*" + ext))):
        if _os.path.basename(f).endswith(".hidden" + ext):
            continue                      # hidden states are not a distribution
        if _os.path.getsize(f) == 0:
            continue                      # not a payload
        out.append(f)
    return out


def empty_payloads():
    """[(label, filename)] for every zero-length payload. Reported, not hidden."""
    import glob as _glob
    import os as _os
    out = []
    for path, label in twp_sources():
        for f in sorted(_glob.glob(_os.path.join(path, "*.f16"))):
            if _os.path.getsize(f) == 0:
                out.append((label, _os.path.basename(f)))
    return out
