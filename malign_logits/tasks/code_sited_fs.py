"""`code_sited` with fifteen few-shot examples. Nothing else differs.

WHY IT EXISTS. RH objected that I compared a licensed coder's 28/29 against an
unlicensed one's 11/20 and read the gap as referent kind. It is instrument:
`code_identity` carries 15 examples and a human licence and flags base-arm
`quiet_drift` at 0.105; `code_sited` carries none and flags at 0.308. The effect
size is the same or LARGER under the noisier coder -- what differs is resolution,
and noise flattens contrasts BETWEEN conditions faster than the conditions
themselves, which is exactly the observed pattern.

Registered at `docs/f20x_examplematch_registration.md`. This file changes ONE
thing against `code_sited`: the example set. Same schema, same system prompt, same
referent handling, same prompt-showing, same model, same temperature. If the
comparison is to isolate examples, nothing else may move.

THE EXAMPLES ARE READ FROM A PARQUET, NOT PASTED. Three of lacan's fifteen were
reformatted for display in the markdown -- newlines rendered as " / " -- and two of
those were the option-list cases whose entire teaching value IS the line structure.
An example saying "these are options, not two accounts" while running them together
on one line teaches the rule with its evidence deleted. `f20x_build_examples.py`
now pulls each passage from the generation parquet by key and refuses to write
unless all fifteen match verbatim, both humans agree, the label matches their
judgment, and none appears in the frozen held-out set.

WHO WROTE WHAT. lacan selected the examples; this seat audited them blind against
the frozen set and runs the analysis. lacan holds the reading this test could
rescue, so writing them was the exposed position. The four cases the humans
DISAGREED on are excluded -- they have no ground truth, and encoding either reading
is the contamination the allocation exists to prevent.
"""
import os

import pandas as pd

from .code_sited import (REFERENT, STIPULATED, SYSTEM_PROMPT, SitedCoding,  # noqa: F401
                         SitedCodingTask, prepare)
from largeliterarymodels.task import Task

EXAMPLES_PARQUET = "data/f20x_coder_examples.parquet"
# The binary judgment each example teaches, mapped onto the scheme's vocabulary.
LABEL_TO_CODES = {
    "fits": ["stable"],
    "does not fit": ["quiet_drift"],
    "too little": ["no_value_posed"],
}


def _load_examples():
    if not os.path.exists(EXAMPLES_PARQUET):
        raise FileNotFoundError(
            f"{EXAMPLES_PARQUET} missing. Run scripts/f20x_build_examples.py; it "
            "refuses to write unless all fifteen verify against the corpus.")
    e = pd.read_parquet(EXAMPLES_PARQUET)
    out = []
    for r in e.itertuples():
        out.append((
            prepare(r.condition, getattr(r, "word", ""), r.prompt, r.text),
            SitedCoding(
                accounts=[], referent_note="", codes=LABEL_TO_CODES[r.label],
                evidence=[], drift_from_genre=False),
        ))
    return out


class SitedFewShotTask(SitedCodingTask):
    name = "f20x_sited_coding_fs"
    examples = _load_examples()
