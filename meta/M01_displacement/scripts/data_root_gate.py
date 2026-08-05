"""data_root_gate.py — REFUSE A RUN POINTED AT THE WRONG DATA ROOT.

**WHY IT EXISTS.** Pinning `largeliterarymodels` to a tag ([4594]) made the
import resolve into `site-packages`. `STASH_PATH` is derived from the package
location, so **the stash, the batch ledger and the raw sidecars moved with the
code** — into a directory the campaign's 9.0 GB of paid annotation history was
not in. A 24-item run there reported **24/24, zero errors, `certify_raw()`
complete: True**, and `cache_hit_rate` 0.0% — every receipt true, about the
wrong volume.

Upstream fixed the derivation (`pin-2026-08-05.2-data-root`): `LITMOD_DATA_DIR`
first, then a non-empty package-relative `data/` OUTSIDE site-packages, else
`~/.largeliterarymodels/data`, and data inside site-packages is refused with a
warning naming both paths.

**BUT THE REMEDY IS AN ENVIRONMENT VARIABLE, AND A VARIABLE IS PER-SHELL.**

    the old defect: *the pin governs which CODE runs, and each COMMAND LINE
      has to remember the prefix.*
    this one:       *the env var governs which DATA is seen, and each SHELL
      has to remember the export.*

**Same shape, one plane over.** A new terminal, a cron entry, a subprocess with
a scrubbed environment, or another seat's session all lose it — and **nothing
in the package can distinguish an empty root that is wrong from an empty root
that is legitimately fresh.** A gate can, because a gate is told which root the
run declares.

**WHAT IT CANNOT DO.** It checks that the run is pointed at the root it says it
is. It cannot tell you that root is the right one for the registration, and it
cannot tell you the data in it is what you think. **It answers "am I where I
said I would be", which is the question the receipts could not answer.**
"""
import os
import sys


class DataRootRefusal(SystemExit):
    pass


def resolve_stash_path():
    """The path the installed library will actually use, asked of the library
    rather than recomputed here. **Recomputing it would make this gate a
    second implementation of the thing it guards** — the defect that put
    `departed` and `tail_excess` on two different definitions."""
    from largeliterarymodels import llm
    return os.path.realpath(llm.STASH_PATH)


def gate_data_root(declared_root, *, require_nonempty=True, verbose=True):
    """Refuse unless the library's stash resolves under `declared_root`.

    `declared_root` is the registration's own statement of where its data
    lives — a literal in the producer, not read from the environment, so the
    thing being checked and the thing checking it cannot move together.
    """
    declared = os.path.realpath(os.path.expanduser(declared_root))
    stash = resolve_stash_path()
    env = os.environ.get("LITMOD_DATA_DIR")

    if verbose:
        print("=== DATA ROOT GATE")
        print("  declared root      %s" % declared)
        print("  library STASH_PATH %s" % stash)
        print("  LITMOD_DATA_DIR    %s" % (env if env else "<unset>"))

    if not os.path.isdir(declared):
        raise DataRootRefusal(
            "REFUSING: declared data root does not exist: %s" % declared)

    if os.path.commonpath([stash, declared]) != declared:
        raise DataRootRefusal(
            "REFUSING: the library will write to %s, which is NOT under the "
            "declared root %s. Set LITMOD_DATA_DIR=%s before running. A run "
            "from here would be cold against the declared history and would "
            "report its 0%% cache-hit rate honestly."
            % (stash, declared, declared))

    #: **AN EMPTY ROOT PASSES A PATH CHECK AND FAILS THE PURPOSE.** The
    #: failure that cost us was a *correct-looking* run against nothing; a
    #: gate that only compares strings would have passed it too.
    if require_nonempty:
        try:
            n = len(os.listdir(stash))
        except FileNotFoundError:
            n = 0
        if n == 0:
            raise DataRootRefusal(
                "REFUSING: %s resolves under the declared root but is EMPTY. "
                "Either this is a genuinely fresh root — pass "
                "require_nonempty=False and say so in the registration — or "
                "the run is about to re-pay for annotations that already "
                "exist elsewhere." % stash)
        if verbose:
            print("  stash task dirs    %d" % n)

    if verbose:
        print("  data root          PASS")
    return {"declared_root": declared, "stash_path": stash,
            "litmod_data_dir": env}


def _selftest():
    """**POSITIVE CONTROL.** A gate that has only ever passed is an untested
    belief. Both refusal paths are exercised against constructed states."""
    import tempfile
    real = resolve_stash_path()
    print("=== SELFTEST")
    ok = []

    #: (1) a root the stash is NOT under must refuse
    with tempfile.TemporaryDirectory() as tmp:
        try:
            gate_data_root(tmp, verbose=False)
            ok.append("**FAIL: accepted a root the stash is not under**")
        except DataRootRefusal as e:
            ok.append("refused a wrong root: %s" % str(e)[:64])

    #: (2) an EMPTY root that the stash IS under must refuse
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "stash")
        os.makedirs(empty)
        import largeliterarymodels.llm as _l
        saved = _l.STASH_PATH
        try:
            _l.STASH_PATH = empty
            try:
                gate_data_root(tmp, verbose=False)
                ok.append("**FAIL: accepted an empty stash**")
            except DataRootRefusal as e:
                ok.append("refused an empty stash: %s" % str(e)[:64])
        finally:
            _l.STASH_PATH = saved

    for line in ok:
        print("  %s" % line)
    bad = [x for x in ok if x.startswith("**FAIL")]
    print("  selftest %s" % ("**FAILED**" if bad else "PASS — both refusal paths bite"))
    print("\n  live resolution at this seat: %s" % real)
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    root = os.environ.get("LITMOD_DATA_DIR") or os.path.expanduser(
        "~/github/largeliterarymodels/data")
    try:
        gate_data_root(root)
    except DataRootRefusal as e:
        print(e)
        sys.exit(1)
