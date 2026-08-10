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

    #: PENDING: present in the stash, absent from ClickHouse, and not declared
    #: anywhere as sources of record. Asked at [5297]; not ingested until the
    #: producing seat confirms each is intended for the record rather than a
    #: scratch or superseded run.
    ("data/raw/twp_twp_00", "twp_twp_00", "PENDING", "7,749 cells; 7,367 absent from CH"),
    ("data/raw/twp_twp_01", "twp_twp_01", "PENDING",
     "10,332 cells; 9,925 absent from CH AND 628 absent from the stash -- the "
     "only directory where both stores are behind, so possibly a partial run"),
    ("data/raw/twp_twp_02", "twp_twp_02", "PENDING", "5,166 cells; 4,896 absent from CH"),
    ("data/raw/twp_w2_gpu2", "twp_w2_gpu2", "PENDING", "7,749 cells; 7,398 absent from CH"),
    ("data/raw/twp_w2_jais", "twp_w2_jais", "PENDING", "5,166 cells; all absent from CH"),
    ("data/raw/twp_lineages_v2", "twp_lineages_v2", "PENDING", "2,583 cells; 2,448 absent from CH"),
    ("data/raw/twp_fill", "twp_fill", "PENDING",
     "536 cells; absent from BOTH stores. The newest directory on disk."),
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
