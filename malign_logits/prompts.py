"""A readable view onto `data/prompt_categorisation.json`.

    from malign_logits.prompts import Prompt, Prompts

    p = Prompt("e6_water_M")
    p.domain, p.language, p.finding, p.slot        # fields
    p.partner                                      # the UNMARKED arm of its pair
    p.group                                        # every member, keyed by role
    p.translation                                  # its Chinese counterpart

    Prompts.where(domain="violence", language="zh")
    Prompts.groups(finding="F11")

The catalogue is a frozen artifact guarded by thirty assertions; this module never
writes to it. It exists because the RELATIONS in that file -- pair partner, group
membership, translation -- were being re-derived by hand in every script that needed
them, four times in one evening, and got it wrong twice.

THREE THINGS IT DELIBERATELY MAKES HARD TO GET WRONG, each of them a defect from
2026-07-30:

  KEYED BY prompt_id, NEVER BY TEXT. 61 prompt STRINGS carry more than one row -- one
  prompt serving two designs. A dict keyed by text keeps whichever row came last, which
  reported "48 group disagreements" where the true figure was 1. `Prompt("e1_credit_M")`
  is unambiguous; `Prompt.find(text)` returns the ranked pick and `Prompts.matching(text)`
  returns all of them, so the ambiguity is visible when it matters.

  ACTIVE BY DEFAULT. Every count wants the status filter and forgetting it once produced
  "source=OTHER: 55" where the answer was 4. `Prompts.where()` returns active rows;
  `status=None` opts into everything and `status="RETIRED"` asks for the rest.

  A LOGICAL PROMPT IS NOT A STRING. `<<<LOGICAL:BOS>>>` is a sentinel that must never
  reach a tokenizer -- `p.is_logical` says so, and `p.text` still returns the sentinel
  rather than pretending it is feedable.
"""
from __future__ import annotations

import json
import os
import re
from functools import cached_property

CAT = None      # resolved lazily so importing this module never touches disk
_ROWS = None
_BY_ID = None
_BY_TEXT = None

SENTINEL = re.compile(r"^<<<LOGICAL:[A-Z0-9_]+>>>$")
STATUS_RANK = {"ACTIVE": 0, "DISPUTED": 1, "RETIRED": 2}


def _path():
    from . import PATH_DATA
    return os.path.join(PATH_DATA, "prompt_categorisation.json")


def _load(force=False):
    """Load the catalogue once. 1,189 rows; held in memory, not re-read per lookup."""
    global _ROWS, _BY_ID, _BY_TEXT
    if _ROWS is not None and not force:
        return
    doc = json.load(open(_path()))
    _ROWS = doc["prompts"]
    _BY_ID = {r["prompt_id"]: r for r in _ROWS}
    _BY_TEXT = {}
    for r in _ROWS:
        _BY_TEXT.setdefault(r.get("prompt"), []).append(r)


def reload():
    """Re-read the catalogue. Only needed if the file changed under a live session."""
    _load(force=True)


def _rank(row):
    """ACTIVE beats DISPUTED beats RETIRED, then grouped, then role-bearing.

    The pick used wherever one text maps to several rows. A plain `dict[text] = row`
    takes whichever came last, and that arbitrary choice is what turned one real
    disagreement into a reported forty-eight.
    """
    return (STATUS_RANK.get(row.get("status"), 3),
            0 if row.get("group_id") else 1,
            0 if row.get("group_role") else 1)


class Prompt:
    """One row of the catalogue. Identity is `prompt_id`, never the text."""

    def __init__(self, prompt_id):
        _load()
        if prompt_id not in _BY_ID:
            raise KeyError(f"no prompt with id {prompt_id!r}")
        self.id = prompt_id
        self._row = _BY_ID[prompt_id]

    # -- identity ---------------------------------------------------------
    def __repr__(self):
        t = (self.text or "")[:38]
        return f"Prompt({self.id!r}, {self.status}, {t!r})"

    def __eq__(self, other):
        return isinstance(other, Prompt) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

    # -- fields, straight through -----------------------------------------
    @property
    def text(self):
        return self._row.get("prompt")

    def __getattr__(self, name):
        """Any catalogue field as an attribute: p.domain, p.contrast_type, p.notes.

        Raises AttributeError rather than returning None for a field the schema does
        not have, so a typo is a mistake instead of a silent null.
        """
        row = self.__dict__.get("_row")
        if row is not None and name in row:
            return row[name]
        raise AttributeError(
            f"{name!r} is not a field of the catalogue. Fields: "
            f"{', '.join(sorted(row or {}))}")

    @property
    def is_active(self):
        return self._row.get("status") == "ACTIVE"

    @property
    def is_logical(self):
        """True when the surface is a sentinel produced by a resolver at run time.

        `<<<LOGICAL:BOS>>>` must never reach a tokenizer. Check this before feeding
        `.text` to anything.
        """
        return bool(self._row.get("resolver")) or bool(SENTINEL.match(self.text or ""))

    @property
    def row(self):
        """The underlying dict. Read-only by convention -- the file is frozen."""
        return dict(self._row)

    # -- relations, which are the reason this class exists ------------------
    @cached_property
    def group(self):
        """Every prompt sharing this one's `group_id`, as a PromptGroup. None if ungrouped."""
        gid = self._row.get("group_id")
        return PromptGroup(gid) if gid else None

    @cached_property
    def partner(self):
        """The opposite arm of a two-member contrast, or None.

        Defined only where the group has exactly one other ACTIVE member. A three-cell
        F11 triple has no single partner and returns None -- use `.group` there, because
        silently picking one of two poles is how sign errors happen.
        """
        g = self.group
        if g is None:
            return None
        others = [p for p in g.members if p.id != self.id and p.is_active]
        return others[0] if len(others) == 1 else None

    @cached_property
    def translation(self):
        """This prompt's Chinese counterpart, or None.

        TWO ROUTES, BECAUSE THE ID CONVENTION HAS EXCEPTIONS. The keying pass named
        Chinese rows `<english_prompt_id>_zh`, and 380 of 386 follow it -- but the id
        it used was the id of whichever TWIN it inherited from, so a dual-identity
        prompt's translation can be keyed under the other twin's name. `e6_water_M`'s
        Chinese row is `violence_explicit_5_zh`; the id route returns nothing for it and
        an analysis trusting one route would silently find no translation for six rows.

        So: the id first, then the group's `_zh` image matched on role. A grouped prompt
        is found either way; an ungrouped one with an off-convention id is not findable
        and returns None rather than guessing.
        """
        if self.language == "zh":
            return None
        direct = _get(self.id + "_zh")
        if direct is not None:
            return direct
        g = self.group
        if g is None or self.group_role is None:
            return None
        zg = g.translation
        if zg is None:
            return None
        same = [m for m in zg.members if m.group_role == self.group_role]
        return same[0] if len(same) == 1 else None

    @cached_property
    def english(self):
        """The English source of a Chinese row, or None. Inverse of `.translation`."""
        if self.language != "zh":
            return None
        if self.id.endswith("_zh"):
            direct = _get(self.id[:-3])
            if direct is not None:
                return direct
        gid = self._row.get("group_id")
        if not gid or not str(gid).endswith("_zh") or self.group_role is None:
            return None
        try:
            eg = PromptGroup(str(gid)[:-3])
        except KeyError:
            return None
        same = [m for m in eg.members if m.group_role == self.group_role
                and m.language != "zh"]
        return same[0] if len(same) == 1 else None

    @cached_property
    def duplicates(self):
        """Other rows carrying the SAME TEXT -- the dual-identity case.

        61 texts have more than one row: one prompt serving two designs. Non-empty here
        means any text-keyed lookup elsewhere is ambiguous for this prompt.
        """
        _load()
        return [Prompt(r["prompt_id"]) for r in _BY_TEXT.get(self.text, [])
                if r["prompt_id"] != self.id]

    # -- constructors -------------------------------------------------------
    @staticmethod
    def find(text):
        """The best row for a text: ACTIVE over DISPUTED over RETIRED, grouped over not.

        Returns None if the text is absent. When several rows tie, this picks one --
        check `.duplicates` if that matters, or use `Prompts.matching(text)`.
        """
        _load()
        rows = _BY_TEXT.get(text)
        return Prompt(sorted(rows, key=_rank)[0]["prompt_id"]) if rows else None


def _get(prompt_id):
    _load()
    return Prompt(prompt_id) if prompt_id in _BY_ID else None


class PromptGroup:
    """A design: the prompts sharing one `group_id`, addressable by role."""

    def __init__(self, group_id):
        _load()
        self.id = group_id
        self._rows = [r for r in _ROWS if r.get("group_id") == group_id]
        if not self._rows:
            raise KeyError(f"no group with id {group_id!r}")

    def __repr__(self):
        return (f"PromptGroup({self.id!r}, {len(self.members)} members: "
                f"{', '.join(sorted(str(r) for r in self.roles))})")

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    @cached_property
    def members(self):
        return [Prompt(r["prompt_id"]) for r in self._rows]

    @property
    def active(self):
        return [p for p in self.members if p.is_active]

    @property
    def roles(self):
        return {p.group_role for p in self.members}

    def role(self, name):
        """Members holding a role. A list, because a role can repeat in a pool."""
        return [p for p in self.members if p.group_role == name]

    @property
    def contrast(self):
        """The group's `pair_contrast` label, e.g. 'kill/scream' or 爱/恨. None if unkeyed.

        The first term belongs to POLE_A / UNMARKED by the project's convention; a label
        whose order disagrees with its roles is a sign error and the suite tests for it.
        """
        for p in self.members:
            if p.row.get("pair_contrast"):
                return p.row["pair_contrast"]
        return None

    @cached_property
    def translation(self):
        """The `_zh` image of this group, or None."""
        if str(self.id).endswith("_zh"):
            return None
        try:
            return PromptGroup(f"{self.id}_zh")
        except KeyError:
            return None


class Prompts:
    """Collection-level access. Every query is ACTIVE-only unless told otherwise."""

    @staticmethod
    def all(status="ACTIVE"):
        _load()
        rows = _ROWS if status is None else [r for r in _ROWS if r.get("status") == status]
        return [Prompt(r["prompt_id"]) for r in rows]

    @staticmethod
    def where(status="ACTIVE", has_translation=None, has_group=None, **fields):
        """Prompts matching every field given.

            Prompts.where(domain="violence", language="zh")
            Prompts.where(finding="F11", has_translation=True)
            Prompts.where(status=None, source="UNMAPPED")     # includes retired

        `status` defaults to ACTIVE because nearly every count wants it and forgetting
        it once is what turned 4 into 55.
        """
        out = []
        for p in Prompts.all(status=status):
            if any(p.row.get(k) != v for k, v in fields.items()):
                continue
            if has_group is not None and bool(p.row.get("group_id")) != has_group:
                continue
            if has_translation is not None and (p.translation is not None) != has_translation:
                continue
            out.append(p)
        return out

    @staticmethod
    def matching(text):
        """EVERY row carrying this text, ranked best-first. The dual-identity view."""
        _load()
        rows = sorted(_BY_TEXT.get(text, []), key=_rank)
        return [Prompt(r["prompt_id"]) for r in rows]

    @staticmethod
    def groups(status="ACTIVE", **fields):
        """Distinct groups whose members match the fields."""
        seen, out = set(), []
        for p in Prompts.where(status=status, **fields):
            gid = p.row.get("group_id")
            if gid and gid not in seen:
                seen.add(gid)
                out.append(PromptGroup(gid))
        return out

    @staticmethod
    def counts(field, status="ACTIVE"):
        """How many prompts per value of a field. Reports the status it counted under."""
        import collections
        c = collections.Counter(p.row.get(field) for p in Prompts.all(status=status))
        return dict(c.most_common())
