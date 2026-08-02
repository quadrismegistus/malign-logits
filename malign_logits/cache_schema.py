"""Declared key schemas for the stashes — data, not code.

WHY THIS FILE EXISTS
--------------------
`CacheManager` grew 26 `get_`, 26 `set_` and 19 `has_` methods against 3
`count_` and 2 `iter_`. The triples are valuable: they put a key shape in ONE
place so no caller hand-builds a key. Three things they do not do, and all
three cost us on 2026-08-02:

**1. THE SCHEMA IS CODE, NOT DATA.** `_twp_key` is a function; `get_beam_words`
inlines its own dict; twenty-six methods each re-express a key shape in their
body. A shape expressed only as code cannot be inspected, validated, diffed or
migrated programmatically — only read by eye and edited by hand. The
2026-07-30 rekey had to be chased through every call site.

**2. THE SURFACE IS INCOMPLETE AND THE GAPS ARE SILENT.** With no typed way to
ask *what does this stash contain*, fourteen sites across `scripts/` and
`malign_logits/` dropped to `cm._stash(...)` and rebuilt keys by hand. That is
not timidity; there was no method to call. A hand-built key is a schema claim
made outside the class that owns the schema.

**3. THE INVARIANTS LIVE IN COMMENTS BESIDE A METHOD**, not attached to the
stash — so *"theta is keyed because two beam widths once coexisted"* is
findable only by someone who already knows to look.

SCOPE, DELIBERATE
-----------------
**`true_word_probs` and `logits` are declared here.** The other twenty-five
migrate when they are next touched. A cache refactor that competes with the paper is the
wrong trade; this one is declared because it blocks a live decision (whether a
run/rule dimension belongs in the key) and it serves as the worked example.

Adding a stash here is a DECLARATION, and a declaration is a claim: it says
these fields identify a measurement and anything not listed does not.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class KeySchema:
    """What identifies one entry in a stash.

    `fields` is the key's shape, in a fixed order so a rendered key is stable.
    `defaults` supplies values a caller may omit. `notes` carries the reason a
    field is keyed — the part that is usually lost and is usually what matters.
    """

    name: str
    fields: List[str]
    defaults: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    notes: Dict[str, str] = field(default_factory=dict)
    doc: str = ""

    def build(self, **kw):
        """kwargs -> the key dict. Raises on anything undeclared or missing.

        REFUSING AN UNDECLARED FIELD IS THE POINT. A typo'd or invented kwarg
        would otherwise produce a key that is merely *different* — it would
        write successfully, read back as absent, and look like missing data.
        """
        unknown = [k for k in kw if k not in self.fields]
        if unknown:
            raise KeyError(
                f"{self.name}: undeclared key field(s) {unknown}; "
                f"declared fields are {self.fields}")
        out = {}
        for f in self.fields:
            if f in kw and kw[f] is not None:
                out[f] = kw[f]
            elif f in self.defaults:
                out[f] = self.defaults[f]
            elif f in self.required:
                raise KeyError(f"{self.name}: key field {f!r} is required")
            else:
                raise KeyError(f"{self.name}: key field {f!r} has no value "
                               f"and no default")
        return out

    def matches(self, key, **filters):
        """Does `key` satisfy these filters?

        `None` means DO NOT FILTER on that field — NOT "match the default".
        The difference is load-bearing: `mode=None` returns raw and chat and
        think together, which is right for a census and wrong for an analysis,
        and a caller who means raw must say `mode="raw"`.
        """
        d = dict(key) if not isinstance(key, dict) else key
        for f, want in filters.items():
            if want is None:
                continue
            if f not in self.fields:
                raise KeyError(f"{self.name}: cannot filter on undeclared "
                               f"field {f!r}; declared are {self.fields}")
            if str(d.get(f)) != str(want):
                return False
        return True


TRUE_WORD_PROBS = KeySchema(
    name="true_word_probs",
    fields=["model", "prompt", "theta", "mode"],
    defaults={"theta": 0.001, "mode": "raw"},
    required=["model", "prompt"],
    doc="Exact P(next WORD) by threshold-bounded token-tree expansion. Value: "
        "{rows: [{word, t1, p}], residual: {tail, drop, open, total}, "
        "batches, rule_version, dict_sha, rule_commits}.",
    notes={
        "theta":
            "KEYED. `beam_words` put `n` in its key and two beam widths (200 "
            "and 1000) then coexisted across 70+ models on different "
            "unrecoverable scales, silently mixed by any reader that did not "
            "filter. Theta plays the same structural role — but unlike a beam "
            "width it is a PRINCIPLED floor: expanding every token above theta "
            "is complete for every word above theta, and the unexpanded mass "
            "is reported as residual rather than divided away.",
        "mode":
            "ALWAYS PRESENT since 2026-07-30; previously omitted when raw, "
            "which gave raw keys a different SHAPE and made raw IMPLICIT. "
            "AND MODE IS NOT ONE DIMENSION: raw and chat are two framings of "
            "ONE stimulus, while continue and think prepend 'Continue this "
            "text:' and therefore measure a DIFFERENT stimulus. Anything that "
            "groups on this field must not pool across that boundary.",
        "model": "Full HuggingFace id, never a family key or short label.",
        "prompt": "The exact stimulus string. Hashed on disk by hashstash.",
    },
)

# ── THE PENDING CHANGE, DECLARED RATHER THAN REMEMBERED ──────────────────
#
# 2026-08-02: two runs of this pipeline — 979 prompts (v3) and 2,583 prompts
# (v4), different instances, a day apart — were compared on their 963 shared
# prompts across 95 models: 89,704 cells, 100.00% IDENTICAL, zero differences.
# So the cell is reproducible given the rule, and `run_id` does NOT belong in
# the key: it would store N identical copies and force every reader to choose a
# run, manufacturing the ambiguity the beam stash suffers from rather than
# removing it. (Beams DO differ run to run, which is exactly why `model` — the
# run anchor — is keyed there. The key holds what the value depends on.)
#
# WHAT DOES belong is the RULE. `rule_version` and `dict_sha` currently live in
# the VALUE, so two cells computed under different boundary rules COLLIDE, and
# the later silently overwrites the earlier. The producer's own comment says
# why `dict_sha` and not `rule_version` alone: "A different word list is a
# different boundary rule wearing the same version number."
#
# NOT APPLIED YET, and the reason is data, not doubt: 93,216 resident entries
# carry keys without these fields, so flipping this without a migration makes
# every one unreadable. The cheap moment is a clear-and-reingest, where the
# stash is rewritten anyway.
TRUE_WORD_PROBS_WITH_RULE = KeySchema(
    name="true_word_probs",
    fields=["model", "prompt", "theta", "mode", "rule_version", "dict_sha"],
    defaults={"theta": 0.001, "mode": "raw"},
    required=["model", "prompt", "rule_version", "dict_sha"],
    doc="As TRUE_WORD_PROBS, with the BOUNDARY RULE keyed so two rules cannot "
        "collide. `run_id` deliberately stays in the value: the cell is "
        "reproducible given the rule (89,704/89,704 across two runs).",
    notes=dict(TRUE_WORD_PROBS.notes, **{
        "rule_version":
            "KEYED. Was value-only, so a re-run under a changed rule "
            "overwrote the old cell and the artifact could say which rule "
            "made a cell but never hold both.",
        "dict_sha":
            "KEYED alongside rule_version because the DICTIONARY IS PART OF "
            "THE RULE and rule_version can go stale relative to a word-list "
            "swap. A different word list is a different boundary rule wearing "
            "the same version number.",
    }),
)

LOGITS = KeySchema(
    name="logits",
    fields=["model", "prompt", "mode", "dtype"],
    defaults={"mode": "raw"},
    required=["model", "prompt", "dtype"],
    doc="Full-vocabulary next-token logits at the last position. Value: "
        "{logits: ndarray, dtype, vocab_size, torch_version, "
        "transformers_version, device, stamped_at}. THE VALUE CARRIES ITS OWN "
        "PROVENANCE -- the archived store held bare ndarrays and could not say "
        "what produced any of them.",
    notes={
        "mode":
            "ALWAYS PRESENT. The archived stash omitted it when raw "
            "(`if mode != 'raw': key['mode'] = mode`, copy-pasted across "
            "get/set/has), which gave raw keys a DIFFERENT SHAPE from moded "
            "ones and made raw IMPLICIT: 31,402 bare {model, prompt} entries "
            "against 21,398 {mode, model, prompt}, with a pre-mode entry "
            "indistinguishable from a raw one. Identical to the defect "
            "true_word_probs was re-keyed to remove on 2026-07-30; logits was "
            "scoped out of that migration and the defect grew.",
        "dtype":
            "KEYED, and this is the rule_version analogue. m04_rescore.py "
            "states it: *A DTYPE DIFFERENCE IS A LOGIT DIFFERENCE, and this "
            "campaign's quantity IS a next-token probability.* The archived "
            "store mixed float16 and float32 with nothing in key or value "
            "recording which -- so unlike true_word_probs, where the collision "
            "had not yet happened, here it already had.",
        "model": "Full HuggingFace id, never a family key or short label.",
        "prompt": "The exact stimulus string.",
    },
)

SCHEMAS = {
    "true_word_probs": TRUE_WORD_PROBS,
    "logits": LOGITS,
}


def schema_for(stash):
    """The declared schema, or None for a stash not yet migrated.

    Returning None rather than raising is deliberate: twenty-six stashes are
    undeclared by design, and the generic engine must degrade to the untyped
    path for them rather than refuse.
    """
    return SCHEMAS.get(stash)
