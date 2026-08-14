"""What can move underneath a producer without touching its own git history.

    uv run python scripts/producer_closure.py meta/M06_generation/scripts
    uv run python scripts/producer_closure.py --all          # every producer
    uv run python scripts/producer_closure.py <file.py>      # one, verbose

WHY THIS EXISTS RATHER THAN A FOURTH GREP. On 2026-08-14 four seats
enumerated the ways a producer's inputs can move while its own history looks
clean. The list grew LIBRARY -> STORE -> SIBLING MODULE -> SHARED CONSTANT in
about an hour, and every value was found by a seat auditing its own folder
after another seat named the previous one. **An enumeration that grows each
time someone looks is not a specification, and greping for the current list
is guaranteed to be one value behind.**

Three of those four are the same object seen from different sides: a library,
a sibling module and a shared constant are all NODES IN A TRANSITIVE IMPORT
CLOSURE. Compute the closure once and all three are covered, including the
values nobody has named yet, because a fifth flavour of "something I import
moved" is still an import.

    library     import malign_logits.fields  -> in the closure
    sibling     import a_dose_response       -> in the closure
    constant    from plot_x import FRAG      -> in the closure
    store       SELECT ... FROM twp_words    -> NOT a file; reported separately

THE STORE IS THE ONE THIS CANNOT CLOSE, and it is flagged rather than
silently omitted. ClickHouse is not in anybody's import graph, so a producer
whose closure is clean can still be reading rows inserted after its artifact
was written. The `store` column says which closures reach the store at all;
those need the insertion-time check for the cells actually read, which no
static tool can do for them.

    grep for a NAME              answers "does this file say X"
    this                         answers "what does this file DEPEND ON"

That distinction is dario's, from the same thread, and it is why the schema
false positive happened: `FROM malign_logits.twp_words` is a name-match for
an import and is not one. Here imports come from `ast`, so a schema string
cannot be mistaken for a dependency and a lowercase SQL `from` cannot either.

WHAT THE VERDICTS MEAN
    STALE     a closure member is NEWER than the artifact. The artifact may
              have been produced under a different definition. Re-run or
              justify.
    ok        every closure member is older than the artifact. Necessary, not
              sufficient -- see the store column.
    no-artifact   the producer's output path could not be resolved, so
              nothing was compared. **This is not a pass.** It is reported
              as its own state precisely so it cannot read as one.

CHANGE TIME IS max(git commit time, mtime), which is deliberately the
EARLIER-flagging of the two. A checkout resets mtimes so mtime alone can
read as ancient; an uncommitted edit has no commit time at all so git alone
misses live work. Taking the max means a file counts as changed if EITHER
witness says so, which biases toward false STALE rather than false ok.
"""
import argparse
import ast
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: A closure member reading the STORE. Deliberately NOT `malign_logits.`,
#: which matches both `import malign_logits.fields` (a LIBRARY exposure) and
#: `FROM malign_logits.twp_words` (a STORE exposure) -- conflating them is the
#: schema false positive that started this, one level down.
#: SQL context, never the bare word. `"clickhouse"` anywhere in a string also
#: matches this file's own help text ("closure reaches ClickHouse via ..."),
#: so the checker classified itself as a store reader even after docstrings
#: were excluded. The binary name is matched EXACTLY instead, which a prose
#: sentence containing the word cannot satisfy.
STORE_MARK = ("from malign_logits.", "join malign_logits.",
              "into malign_logits.", "table malign_logits.",
              "clickhouse client")
STORE_EXACT = ("clickhouse",)
#: the names whose VALUES are markers rather than uses of them. Listed
#: explicitly and not matched by suffix: the suffix version covered
#: STORE_MARK, I moved the bare token into STORE_EXACT, and the exclusion
#: silently stopped covering it -- the repair reproducing the defect.
MARKER_NAMES = ("STORE_MARK", "STORE_EXACT")
#: calls whose path argument is a WRITE. The write is the disambiguator
#: (malign's, [6100]): a producer that only READS os.path.join(OUTD, x) is
#: not its author, and comparing that file's age to the closure is a
#: comparison between two things neither of which produced the other.
WRITE_CALLS = ("to_parquet", "to_csv", "to_json", "to_feather", "savefig",
               "write_text", "write_bytes", "imsave")
#: and the mirror set, for the FIFTH exposure: a producer whose input is
#: another producer's ARTIFACT. Not a library, not a store, not a sibling
#: module, not a shared constant -- a data file, which is in no import graph
#: at all, so the closure check above cannot see it either.
READ_CALLS = ("read_parquet", "read_csv", "read_json", "read_feather",
              "read_text", "read_bytes", "load", "imread")
SKIP_DIRS = (".git", ".venv", "node_modules", "__pycache__", ".migrate_")


def produced_time(path):
    """When an ARTIFACT was written. mtime only, deliberately.

    NOT max(mtime, commit) the way `change_time` does it for dependencies.
    `mediation_pairs.parquet` was written 08-13 22:13 and committed 08-14
    13:37; taking the max dates it to the COMMIT, so a seat committing an old
    artifact makes it look newer than every dependency and the staleness
    disappears. The two times want opposite biases:

        a DEPENDENCY should look as NEW as possible   -> max, over-flags
        an ARTIFACT should look as OLD as possible    -> mtime, over-flags

    Both errors then run toward false STALE rather than false ok, which is
    the only direction a guard may be wrong in.
    """
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def change_time(path, _cache={}):
    """max(last commit touching it, mtime) -- see the module docstring."""
    if path in _cache:
        return _cache[path]
    mt = os.path.getmtime(path) if os.path.exists(path) else 0.0
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%ct", "--", path],
                           cwd=ROOT, capture_output=True, text=True, timeout=20)
        gt = float(r.stdout.strip() or 0)
    except Exception:
        gt = 0.0
    _cache[path] = max(mt, gt)
    return _cache[path]


def last_commit(path, _cache={}):
    """The commit that last touched `path`, or "" if none."""
    if path in _cache:
        return _cache[path]
    try:
        r = subprocess.run(["git", "rev-list", "-1", "HEAD", "--", path],
                           cwd=ROOT, capture_output=True, text=True, timeout=20)
        _cache[path] = r.stdout.strip()
    except Exception:
        _cache[path] = ""
    return _cache[path]


def same_commit(a, b):
    """Were these two files last touched by the SAME commit?

    If so, no staleness is expressible between them: **within one commit the
    file mtimes are arbitrary**, so a producer that writes two files in one
    run can flag either against the other depending on which got written a
    second sooner. @registrar found three of five M05 flags were exactly
    this, and it is worse than a coincidence -- it is a GUARANTEED false
    positive, because `content_moved` then compares the dependency against
    the commit BEFORE the shared one, where its content really was different.
    """
    ca, cb = last_commit(a), last_commit(b)
    return bool(ca) and ca == cb


def content_moved(dep, at_t):
    """Did `dep`'s CONTENT change after time `at_t`? -> True/False/None.

    **A TIMESTAMP IS NOT A CHANGE, and that distinction cost the campaign an
    hour on 2026-08-14.** A seat reported 212,776 changed mass values in a
    parquet; the file was byte-unstable because its rows come out in a
    different order each run, and the values had never moved at all. An
    mtime is weaker evidence still: a commit, a touch, or an identical
    rewrite all move it while the bytes stay put.

    So a STALE verdict from times alone is an ORDERING claim, not a change
    claim. This resolves it against content: the blob at the last commit
    on or before `at_t`, against the blob now.

        True   content differs -- the dependency really did move
        False  identical bytes -- the ordering is real and means nothing
        None   no history to compare (untracked, ignored, or born after
               the artifact); UNVERIFIABLE, which is not the same as clean
    """
    try:
        r = subprocess.run(
            ["git", "rev-list", "-1", "--before=@%d" % int(at_t), "HEAD",
             "--", dep], cwd=ROOT, capture_output=True, text=True, timeout=20)
        commit = r.stdout.strip()
        if not commit:
            return None
        rel = os.path.relpath(dep, ROOT)
        old = subprocess.run(["git", "rev-parse", "%s:%s" % (commit, rel)],
                             cwd=ROOT, capture_output=True, text=True, timeout=20)
        if old.returncode != 0:
            return None
        new = subprocess.run(["git", "hash-object", dep], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        if new.returncode != 0:
            return None
        return old.stdout.strip() != new.stdout.strip()
    except Exception:
        return None


def resolve(name, from_dir):
    """A dotted module name -> a repo-local .py file, or None.

    Candidate roots are the importer's OWN directory (which is how sibling
    imports resolve, and how a `sys.path.insert(ROOT/scripts)` resolves too),
    then scripts/, then the repo root for packages like malign_logits.
    """
    parts = name.split(".")
    for base in (from_dir, os.path.join(ROOT, "scripts"), ROOT):
        p = os.path.join(base, *parts)
        for cand in (p + ".py", os.path.join(p, "__init__.py")):
            if os.path.exists(cand):
                return os.path.realpath(cand)
    return None


def imports_of(path):
    """Every module name this file imports, from the AST.

    ast.walk rather than module-level iteration, deliberately: a function-body
    import is a dependency exactly as much as a top-of-file one, and this repo
    has several (a `from clean_dream_text import clean` inside a function).
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except Exception:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            #: level > 0 is a relative import; module may be None for `from . import x`
            if n.level == 0 and n.module:
                out.append(n.module)
    return out


def closure(path):
    """Transitive repo-local import closure of one producer, excluding itself."""
    seen, stack = set(), [os.path.realpath(path)]
    start = os.path.realpath(path)
    while stack:
        cur = stack.pop()
        for name in imports_of(cur):
            r = resolve(name, os.path.dirname(cur))
            if r and r not in seen and r != start:
                seen.add(r)
                stack.append(r)
    return sorted(seen)


def touches_store(path):
    """Does this file QUERY ClickHouse (as opposed to importing the library)?

    THE MARKER IS SOUGHT IN STRING LITERALS ONLY, via the AST. A text grep
    for `from malign_logits.` matches `from malign_logits.cache import ...`,
    which is a LIBRARY import and a different exposure entirely -- I wrote
    STORE_MARK with a comment claiming to have separated the two and then
    conflated them anyway, flagging `m06_style.py` as a store reader on the
    strength of an import. SQL lives in strings; imports are AST nodes. The
    two cannot collide once you stop looking at the file as text.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except Exception:
        return False
    #: docstrings are bare string EXPRESSION STATEMENTS and are excluded --
    #: several files in this repo (including this one) discuss ClickHouse in
    #: prose without ever querying it. SQL is never a bare statement.
    doc = {id(n.value) for n in ast.walk(tree)
           if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
           and isinstance(n.value.value, str)}
    #: AND A DEFINITION OF THE MARKER IS NOT A USE OF IT. This file assigns
    #: the marker strings to STORE_MARK, so on its first validation run the
    #: checker classified ITSELF as a store reader -- the instrument inside
    #: the population it measures, which is why the known-negative case was
    #: in the fixture at all.
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in MARKER_NAMES
                        for t in n.targets)):
            doc |= {id(c) for c in ast.walk(n.value)}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in doc):
            s = n.value.lower()
            if any(m in s for m in STORE_MARK) or s.strip() in STORE_EXACT:
                return True
    return False


def _consts(tree, selfpath):
    """Module-level NAME -> absolute path, for plain literals and joins.

    Both directories and files: `OUTD = os.path.join(ROOT, "meta/.../results")`
    is the common shape here and the artifact is only assembled at the call
    site, so the directory constant has to be carried to get there.
    """
    out = {}
    for n in tree.body:
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        t = n.targets[0]
        if not isinstance(t, ast.Name):
            continue
        v = _path_expr(n.value, out, selfpath)
        if v:
            out[t.id] = v
    return out


def _path_expr(node, consts, selfpath):
    """A path-shaped AST expression -> an absolute path string, or None.

    Handles the `HERE = os.path.dirname(os.path.abspath(__file__))` /
    `ROOT = os.path.dirname(HERE)` idiom every producer in this repo opens
    with. Without it ROOT is unresolvable, so OUTD is, so the artifact is --
    and the checker returns `no-artifact` for the entire repo while looking
    like it ran. That was the first fixture's finding.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        s = node.value
        if not s or s.startswith(("http", "SELECT", "select")):
            return None
        return s if os.path.isabs(s) else os.path.join(ROOT, s)
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return selfpath
        return consts.get(node.id)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        fn = node.func.attr
        if fn in ("dirname", "abspath", "realpath", "expanduser") and node.args:
            inner = _path_expr(node.args[0], consts, selfpath)
            if not inner:
                return None
            return os.path.dirname(inner) if fn == "dirname" else inner
        if fn == "join":
            parts = []
            for a in node.args:
                p = _path_expr(a, consts, selfpath)
                if p is None:
                    return None
                #: a bare relative literal was absolutised against ROOT by the
                #: Constant branch; inside a join it is a SEGMENT, so undo that
                if (isinstance(a, ast.Constant) and not os.path.isabs(a.value)):
                    p = a.value
                parts.append(p)
            return os.path.join(*parts) if parts else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _path_expr(node.left, consts, selfpath)
    return None


def path_literals(path):
    """Every existing repo file this module NAMES anywhere, however used.

    Deliberately over-inclusive, and only applied to closure MEMBERS. The
    precise `reads_of` cannot see `malign_logits/prompts.py`'s catalogue: it
    does `json.load(open(_path()))` where `_path()` is a FUNCTION returning
    the join, which needs interprocedural resolution this does not have.

    Over-inclusion is the correct direction here. @malign's ordering: a false
    positive is noise and somebody checks it; a false negative is silence and
    nobody does. A library that names a data file is a library that probably
    reads it, and being wrong costs a flag.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except Exception:
        return []
    selfpath = os.path.realpath(path)
    consts = _consts(tree, selfpath)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "join":
            p = _path_expr(n, consts, selfpath)
            if p and os.path.isfile(p):
                out.append(p)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and "/" not in n.value and n.value.count(".") == 1 \
                and n.value.rsplit(".", 1)[-1] in DATA_EXT:
            #: A BARE FILENAME, RESOLVED BY UNIQUE BASENAME. The shape is
            #: `os.path.join(PATH_DATA, "prompt_categorisation.json")` where
            #: PATH_DATA arrives via `from . import PATH_DATA` inside a
            #: function body -- a constant in ANOTHER module, which needs
            #: cross-module resolution to follow. Matching the basename
            #: against the repo costs nothing and is exact when the name is
            #: unique; ambiguous names are skipped rather than guessed.
            hit = _basename_index().get(n.value)
            if hit and len(hit) == 1:
                out.append(hit[0])
    return sorted(set(out))


DATA_EXT = ("json", "csv", "parquet", "jsonl", "gz", "txt", "yml", "yaml")


def _basename_index(_cache={}):
    """basename -> [absolute paths], for files under data/ and meta/."""
    if _cache:
        return _cache
    for base in ("data", "meta"):
        for dirpath, dirnames, files in os.walk(os.path.join(ROOT, base)):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(SKIP_DIRS) and d != "raw"]
            for f in files:
                if f.rsplit(".", 1)[-1] in DATA_EXT:
                    _cache.setdefault(f, []).append(os.path.join(dirpath, f))
    return _cache


def reads_of(path):
    """Every existing file this producer READS, resolved statically.

    Same machinery as `artifacts_of`, opposite direction. A read that lands
    on another producer's write is a DATA-FILE dependency: it does not appear
    in any import graph, so the transitive closure above is blind to it, and
    it is the most common dependency in this repo by a wide margin.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except Exception:
        return []
    selfpath = os.path.realpath(path)
    consts = _consts(tree, selfpath)
    found = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else "")
        p = None
        if name == "open" and n.args:
            mode = ""
            if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                mode = str(n.args[1].value)
            for kw in n.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if not mode.startswith(("w", "a", "x")):
                p = _path_expr(n.args[0], consts, selfpath)
        elif name in READ_CALLS and n.args:
            p = _path_expr(n.args[0], consts, selfpath)
        if p and "%" not in p and os.path.isfile(p):
            found.append(p)
    return sorted(set(found))


def artifacts_of(path):
    """Every existing file this producer WRITES, resolved statically.

    The write is what makes an artifact the producer's own -- a file it merely
    reads is somebody else's output and its age says nothing about this
    producer's closure. Unresolvable or not-yet-existing paths are dropped
    silently here and surface as `no-artifact`, which is reported as its own
    state rather than as a pass.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except Exception:
        return []
    selfpath = os.path.realpath(path)
    consts = _consts(tree, selfpath)
    found = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else "")
        p = None
        if name in ("open",) and n.args:
            mode = ""
            if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                mode = str(n.args[1].value)
            for kw in n.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if mode.startswith(("w", "a", "x")):
                p = _path_expr(n.args[0], consts, selfpath)
        elif name in WRITE_CALLS and n.args:
            p = _path_expr(n.args[0], consts, selfpath)
        if not p:
            continue
        if "%" in p:                       # "p_on_passages%s.json"
            import glob as _g
            found += [q for q in _g.glob(p.replace("%s", "*").replace("%d", "*"))
                      if os.path.isfile(q)]
        elif os.path.isfile(p):
            found.append(p)
    return sorted(set(found))


def audit(paths):
    #: PASS 1: who writes what, so a read can be attributed to a producer.
    #: The artifact graph only exists once every writer is known, which is
    #: why this cannot be done one producer at a time the way the import
    #: closure can.
    writer = {}
    for p in paths:
        for a in artifacts_of(p):
            writer.setdefault(a, []).append(os.path.relpath(p, ROOT))

    rows = []
    for p in paths:
        cl = closure(p)
        arts = artifacts_of(p)
        rds = reads_of(p)
        #: a read that lands on another producer's write. Self-reads are
        #: excluded: a producer that reads back its own output is resuming,
        #: not depending on someone else.
        me = os.path.relpath(p, ROOT)
        dataedge = [(r, w) for r in rds for w in writer.get(r, []) if w != me]
        store = [c for c in cl if touches_store(c)]
        if touches_store(p):
            store = [p] + store
        newest, newest_t = None, 0.0
        for c in cl:
            t = change_time(c)
            if t > newest_t:
                newest, newest_t = c, t
        #: an artifact is stale if a closure member changed AFTER it was
        #: written. Every artifact is checked, not the first or the newest:
        #: one producer writing a .json and a .parquet can have re-run for
        #: one and not the other.
        #: the artifact edge counts toward staleness exactly as an import
        #: edge does: an input that moved after the output was written is an
        #: input that moved, whether it is code or data.
        for r, _w in dataedge:
            t = change_time(r)
            if t > newest_t:
                newest, newest_t = r, t
        #: A TIME ORDERING IS PROMOTED TO A CHANGE ONLY BY CONTENT, and
        #: same-commit pairs are excluded before that even runs. Both gates
        #: are per-ARTIFACT: `stale` used to be the timestamp list and the
        #: verdict the content one, so the per-artifact "<- STALE" marker
        #: printed a different answer from the verdict two lines below it.
        #: EVERY moved input is reported, not the newest one. Reporting only
        #: `newest_dep` named `movement.py` for @dario's confirmed instance
        #: when the driver was the prompt catalogue: the flag was right and
        #: the diagnosis sent the reader to a file that had nothing to do
        #: with it. A single summary is an undeclared choice, made in the
        #: field most likely to be read as the answer.
        #: AND THE READS OF EVERY MODULE IN THE CLOSURE, not only the
        #: producer's own. `t_fans.py` imports `malign_logits.prompts`, which
        #: is "a readable view onto data/prompt_categorisation.json" and
        #: json.loads it at run time. The module had not moved since 07-30;
        #: the catalogue moved 08-10, after the 08-06 artifact. **That is the
        #: mechanism behind the campaign's only confirmed Class 4 instance**,
        #: and the import closure and the artifact edge are both blind to it:
        #: the file is read by a DEPENDENCY, not by the producer.
        inputs = list(cl) + [r for r, _w in dataedge]
        inputs += reads_of(p)
        for m in cl:
            inputs += path_literals(m)
        inputs = sorted(set(inputs))
        moved = None
        stale, dismissed, movers = [], [], []
        for a in arts:
            at = produced_time(a)
            hits, why = [], set()
            for dep in inputs:
                if change_time(dep) <= at:
                    continue
                if same_commit(a, dep):
                    why.add("same-commit")
                    continue
                m = content_moved(dep, at)
                if m is False:
                    why.add("identical-bytes")
                    continue
                hits.append((os.path.relpath(dep, ROOT), m))
                moved = True if m is True else (moved if moved is not None else m)
            if hits:
                stale.append(a)
                movers.append([a if not a.startswith(ROOT)
                               else os.path.relpath(a, ROOT),
                               [h for h, _m in hits]])
            elif why:
                dismissed.append((a, "/".join(sorted(why))))
        if not arts:
            verdict = "no-artifact"
        elif stale and moved is True:
            verdict = "STALE"
        elif stale:
            verdict = "unverified"    # no history to settle it
        elif dismissed:
            verdict = "ordering"      # flagged on time, dismissed on content
        else:
            verdict = "ok"
        rows.append({"producer": os.path.relpath(p, ROOT),
                     "n_deps": len(cl),
                     "deps": [os.path.relpath(c, ROOT) for c in cl],
                     "newest_dep": os.path.relpath(newest, ROOT) if newest else None,
                     "newest_dep_t": newest_t or None,
                     "artifacts": [os.path.relpath(a, ROOT) for a in arts],
                     "stale": [os.path.relpath(a, ROOT) for a in stale],
                     "store": bool(store),
                     "store_via": [os.path.relpath(c, ROOT) for c in store],
                     "content_moved": moved,
                     "dismissed": [[a, why] for a, why in dismissed],
                     "movers": movers,
                     "n_dataedge": len(dataedge),
                     "dataedge": [[os.path.relpath(r, ROOT), w]
                                  for r, w in dataedge],
                     "verdict": verdict})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="a .py file or a directory")
    ap.add_argument("--all", action="store_true", help="every .py under scripts/ and meta/*/scripts/")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    def every_producer():
        out = []
        for d in [os.path.join(ROOT, "scripts")] + sorted(
                os.path.join(ROOT, "meta", m, "scripts")
                for m in os.listdir(os.path.join(ROOT, "meta"))
                if os.path.isdir(os.path.join(ROOT, "meta", m, "scripts"))):
            out += sorted(os.path.join(d, f) for f in os.listdir(d)
                          if f.endswith(".py"))
        return out

    if a.all:
        paths = every_producer()
    elif a.target and os.path.isdir(a.target):
        paths = sorted(os.path.join(a.target, f) for f in os.listdir(a.target)
                       if f.endswith(".py"))
    elif a.target:
        paths = [a.target]
    else:
        return ap.print_help() or 2

    #: THE ARTIFACT GRAPH NEEDS THE WHOLE POPULATION, and a narrow target
    #: silently loses it: run on one file, `writer` holds only that file's
    #: own writes, every data edge disappears and the verdict comes back
    #: `ok`. `m06_mediation_read.py` read ok alone and STALE in its folder,
    #: same instant -- a guard passing because it could not see, which is
    #: the shape this whole check exists to catch. Scope the population to
    #: the repo always; report only what was asked for.
    rows = [r for r in audit(sorted(set(every_producer()) | set(
        os.path.realpath(p) for p in paths)))
        if os.path.join(ROOT, r["producer"]) in
        {os.path.realpath(p) for p in paths}]
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0

    import time as _t

    def when(rel):
        """Dependency time: max(mtime, commit)."""
        return _t.strftime("%Y-%m-%d %H:%M",
                           _t.localtime(change_time(os.path.join(ROOT, rel))))

    def when_art(rel):
        """Artifact time: mtime. Must match what the verdict was computed
        from -- printing change_time beside a STALE mark showed 13:37 for a
        file the verdict had treated as 22:13, so the number on screen did
        not explain the mark next to it."""
        return _t.strftime("%Y-%m-%d %H:%M",
                           _t.localtime(produced_time(os.path.join(ROOT, rel))))

    if len(rows) == 1:
        r = rows[0]
        print("%s" % r["producer"])
        print("  closure    %d repo-local modules" % r["n_deps"])
        for d in r["deps"]:
            mark = "  <- READS STORE" if d in r["store_via"] else ""
            print("     %-56s %s%s" % (d, when(d), mark))
        print("  writes     %d resolvable artifact(s)" % len(r["artifacts"]))
        for a in r["artifacts"]:
            print("     %-56s %s%s"
                  % (a, when_art(a), "  <- STALE" if a in r["stale"] else ""))
        if r["dataedge"]:
            print("  reads      %d artifact(s) written by another producer"
                  % len(r["dataedge"]))
            for rd, w in r["dataedge"]:
                print("     %-56s %s\n       written by %s" % (rd, when(rd), w))
        print("  verdict    %s" % r["verdict"])
        if r["verdict"] == "STALE":
            print("             newest input %s (%s)"
                  % (r["newest_dep"], when(r["newest_dep"])))
        if r["store"]:
            print("\n  STORE EXPOSURE: closure reaches ClickHouse via %s"
                  % (", ".join(r["store_via"]) or "itself"))
            print("  A clean closure does NOT clear this. ClickHouse is in no")
            print("  import graph, so check INSERTION TIME for the cells read.")
        return 0

    by = collections.Counter(r["verdict"] for r in rows)
    print("%d producers | %s" % (len(rows), dict(by)))
    imp = sum(1 for r in rows if r["n_deps"])
    dat = sum(1 for r in rows if r["n_dataedge"])
    print("  IMPORT edge  (library / sibling / shared constant): %d producers" % imp)
    print("  ARTIFACT edge (reads another producer's output):    %d producers" % dat)
    print("  STORE        (not a file; needs insertion time):    %d producers"
          % sum(1 for r in rows if r["store"]))
    print("  neither import nor artifact edge:                   %d producers"
          % sum(1 for r in rows if not r["n_deps"] and not r["n_dataedge"]))
    print("\n  %-52s %5s %5s %5s %6s %-9s"
          % ("producer", "imp", "data", "arts", "store", "verdict"))
    for r in sorted(rows, key=lambda r: (r["verdict"] != "STALE",
                                         -(r["n_deps"] + r["n_dataedge"]))):
        if not r["n_deps"] and not r["n_dataedge"] and r["verdict"] == "no-artifact":
            continue
        print("  %-52s %5d %5d %5d %6s %-9s"
              % (r["producer"], r["n_deps"], r["n_dataedge"],
                 len(r["artifacts"]), "yes" if r["store"] else "-",
                 r["verdict"]))
    for r in rows:
        for a in r["stale"]:
            print("\n  STALE  %s" % a)
            print("         written %s, but %s changed %s"
                  % (when(a), r["newest_dep"], when(r["newest_dep"])))
    print("\n  (producers with no repo-local deps AND no resolvable artifact"
          " are omitted: nothing was measured about them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
