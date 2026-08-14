"""The slot explorer must measure the SAME THING THE STORE HOLDS.

Every one of these was watched to fail before it was kept.

WHY THIS FILE EXISTS. Three defects on 2026-08-14, all in the slot tooling, all
of which passed every check I ran, because in each case I checked the thing I
had just written instead of the thing that runs:

  1. `x_slot_screen.py` importlib-loaded `scripts/true_word_probs.py` while its
     docstring claimed to be avoiding a second copy of the boundary rule. That
     copy has NO CJK prefix trie, NO mojibake channel and an 18-line
     `boundary_mask` against `twp.py`'s 32, and every `twp_words` row is
     rule_version 3. On English the two agree to FOUR DECIMALS -- 0.0287 against
     the store's 0.0286 -- so nothing looked wrong.
  2. `/api/slot` grew a pooled default and the Svelte client kept sending
     `model=` on every request, so the app ran base-only. My verification was a
     curl with no `model` param: a default path THE APP NEVER TAKES.
  3. A commit whose patch script died before writing; `npm run build` rebuilt
     the unchanged source and the commit landed describing edits that did not
     exist.

@registrar's [6161] names the class: a test that perturbs THE GUARD is not a
test that drives THE STATE through the module, and the difference only shows on
structural guards. So these assert against the STORE, which no amount of
re-reading my own code can satisfy.

THE STORE IS THE ORACLE, not a fixture. `twp_words` at rule_version 3 is what
every finding in M01 is computed from; if the live expansion disagrees with it,
the UI and the findings are measuring different things and the number on screen
is not the number in the paper.
"""
import json
import math
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
#: One prompt whose stored distribution is stable and whose top words are known
#: from the session that found the defect. Any ACTIVE prompt would do; this one
#: is chosen because its numbers appear in X_safety_ablation.
PROMPT = "She slowly took off her"
MODEL = "meta-llama/Llama-3.1-8B"


def _ch(sql):
    try:
        out = subprocess.run([CH, "client", "--query", sql], capture_output=True,
                             text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("clickhouse not available")
    if out.returncode:
        pytest.skip("clickhouse query failed: %s" % out.stderr[:120])
    return out.stdout


def _stored():
    """{word: p} for PROMPT under MODEL, folded, straight from the store."""
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    rows = _ch("SELECT word, sum(p) AS p FROM malign_logits.twp_words "
               "WHERE model='%s' AND prompt='%s' GROUP BY word "
               "FORMAT JSONEachRow" % (esc(MODEL), esc(PROMPT)))
    d = {}
    for line in rows.splitlines():
        if line.strip():
            r = json.loads(line)
            d[r["word"]] = float(r["p"])
    if not d:
        pytest.skip("no stored cell for the probe prompt")
    return d


def test_the_library_instrument_is_rule_3_and_the_scripts_copy_is_not():
    """The two implementations differ, and the difference is what CJK needs.

    This is the cheap half: it needs no model and no store, and it fails the
    moment someone points a consumer at the scripts copy again.
    """
    from malign_logits import twp
    assert twp.RULE_VERSION == 3
    other = os.path.join(ROOT, "scripts", "true_word_probs.py")
    if not os.path.exists(other):
        pytest.skip("scripts/true_word_probs.py is gone")
    src = open(other).read()
    #: Named individually rather than as a count, so a failure says WHICH
    #: capability the other copy lacks instead of "a number changed".
    for missing in ("load_prefix_trie", "is_mojibake", "cjk_vocab", "intra_word"):
        assert missing not in src, (
            "scripts/true_word_probs.py now has %s -- if the two rules have "
            "converged, this test is obsolete; if a consumer was pointed at it, "
            "check which rule the store holds first" % missing)


def test_no_slot_consumer_reaches_the_scripts_copy():
    """THE TEST THAT WOULD HAVE CAUGHT DEFECT 1, and the other one would not.

    Documenting that the two rules differ does not stop a consumer using the
    wrong one -- that is a test of the guard. This drives at the state: which
    module do the slot tools actually LOAD.

    AND ITS FIRST VERSION HAD THE DEFECT IT NAMES. A text search with a
    comment-stripper that handled `#` but not DOCSTRINGS flagged
    `malign_logits/server.py`, whose docstring reads "THE INSTRUMENT IS
    `malign_logits.twp`, NOT `scripts/true_word_probs.py`" -- a NOT-clause read
    as a read, which is @registrar's `.f16` false positive verbatim, inside the
    test whose comment claimed to avoid it. Watched to fail that way before it
    was fixed, and watched to fail for the RIGHT reason after.

    So it parses instead of grepping: a LOAD is a `spec_from_file_location` call
    or an `import`, and prose can say the name as often as it likes.
    """
    import ast
    targets = ["malign_logits/server.py",
               "meta/M01_displacement/scripts/x_slot_screen.py"]

    def docstring_nodes(tree):
        """Every Constant that IS a docstring, so prose can be excluded by identity."""
        out = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)) and n.body:
                first = n.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    out.add(id(first.value))
        return out

    bad = []
    for rel in targets:
        f = os.path.join(ROOT, rel)
        if not os.path.exists(f):
            continue
        tree = ast.parse(open(f).read())
        docs = docstring_nodes(tree)
        for node in ast.walk(tree):
            #: THE FILENAME IS THE EVIDENCE, wherever it is assembled. The real
            #: defect built the path as
            #:     os.path.join(ROOT, "scripts", "true_word_probs.py")
            #: and passed the VARIABLE to spec_from_file_location, so a check
            #: that inspected that call's arguments found a Name and nothing
            #: else. Looking at the call was looking at the wrong node.
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "true_word_probs" in node.value and id(node) not in docs):
                bad.append("%s: %r in code" % (rel, node.value))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (getattr(node, "module", "") or "") + " " + \
                      " ".join(a.name for a in node.names)
                if "true_word_probs" in mod:
                    bad.append("%s: import" % rel)
    assert not bad, (
        "%s loads scripts/true_word_probs.py, which is NOT rule_version 3: no "
        "CJK trie, no mojibake channel, 18-line boundary_mask. Import "
        "malign_logits.twp instead." % ", ".join(bad))


def test_stored_distribution_matches_a_live_expansion():
    """THE STATE TEST. Expand live, compare to the store, word by word.

    Skipped without the model or the store, because the point is to catch a
    disagreement, not to fail on a laptop that cannot run either.
    """
    stored = _stored()
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from malign_logits import twp
    except ImportError:
        pytest.skip("torch/transformers not available")
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        pytest.skip("no accelerator; the live expansion would take minutes")

    tok, _loader = twp.load_tokenizer(MODEL)
    dev = twp.pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    bmask = twp.boundary_mask(tok, model.config.vocab_size)
    trie = twp.load_prefix_trie()
    cjk = None
    if trie is not None:
        cids, cstrs, lids, pids = twp.cjk_vocab(tok, model.config.vocab_size)
        if len(cids):
            cjk = (trie, cids, cstrs, lids, pids)
    w, _res, _calls = twp.expand(model, tok, PROMPT, dev, bmask, cjk=cjk,
                                 bos_policy=twp.bos_policy_for(MODEL))
    live = {}
    for (sf, _t1), m in w.items():
        live[sf] = live.get(sf, 0.0) + m

    #: TOLERANCE IS fp16 NONDETERMINISM, NOT A RULE DIFFERENCE. The producer's
    #: own docstring measures the stash against a fresh pass at 4.4e-03 and sets
    #: the mover floor at 3e-03. A RULE difference is not a small disagreement:
    #: the scripts copy moved this prompt's naughty mass 0.0287 against 0.0286,
    #: i.e. 3.9e-05 per word -- which is why the tolerance is on the TOP WORDS,
    #: where a boundary-rule change relocates whole surfaces rather than
    #: perturbing them.
    top = sorted(stored.items(), key=lambda x: -x[1])[:10]
    for word, p in top:
        assert word in live, (
            "the store holds %r at %.4f and a live expansion does not produce "
            "it at all -- that is a BOUNDARY RULE disagreement, not float noise"
            % (word, p))
        assert math.isclose(live[word], p, rel_tol=0.02, abs_tol=5e-3), (
            "%r: store %.5f, live %.5f -- beyond fp16 nondeterminism, so the "
            "UI and the findings are computing different quantities"
            % (word, p, live[word]))
