"""READ GATE ON THE FORMAT BATTERY. Offered [2100].2, taken up [2101].

    from scripts.f20x_read_gate import load
    d = load()                      # every column EXCEPT `text`
    d = load(unfreeze="A")          # requires RH's route on the docket

WHY. [2099].1 froze arm-level reading of `data/f20x_format_battery.parquet`
until RH picks a route: **the asset is not the 9,280 completions, it is their
UNREADNESS, and unreadness is spent the first time anyone looks.**

**BUT THE FREEZE IS A RULING OVER SEATS AND THE FILE IS READABLE BY ANY
COMMAND.** It holds exactly as long as three seats remember it holds — which is
the care-based remedy this docket has spent two days replacing with mechanism.
**This converts three attestations into one property of the file.**

WHAT IT ALLOWS AND WHAT IT REFUSES.

    ALLOWED   every structural column: family, base_model_id, model_id, arm,
              stim_id, condition, word, level, prompt, draw, temperature.
              Row counts, cross-tabs, completeness grids, N-per-cell. All of
              [2098] and [2103] were built from these.
    REFUSED   `text` — the completions themselves, and therefore every
              arm mean, per-level statistic and family contrast.

**`prompt` IS ALLOWED AND `text` IS NOT, and the line is exactly there: the
prompt is the stimulus we authored, the text is the model's answer.** Reading
what we wrote unblinds nobody; reading what came back is the whole of it.

UNFREEZING IS NAMED, NOT SILENT. `load(unfreeze="A")` or `"B"` returns `text`
and PRINTS the route it was unfrozen under. **A gate that can be opened without
saying so is a comment.** The route is not validated against the docket — that
would be a lock, and custody does not hold the key to RH's decision — but an
unfreeze that has to name a route cannot happen by reflex or by autocomplete.
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "data", "f20x_format_battery.parquet")
OUTCOME = "text"
ROUTES = {"A": "exploratory read of the two existing levels",
          "B": "freeze-then-complete; read the full grid under the frozen plan"}


def load(unfreeze=None, path=PARQUET):
    """The battery WITHOUT its outcome column, unless a route is named."""
    d = pd.read_parquet(path)
    if unfreeze is None:
        if OUTCOME in d.columns:
            d = d.drop(columns=[OUTCOME])
        return d
    if unfreeze not in ROUTES:
        raise ValueError(
            f"unfreeze must name RH's route, one of {sorted(ROUTES)} — "
            f"got {unfreeze!r}. A gate opened without naming why is a comment.")
    print(f"*** F20x READ GATE OPENED under route {unfreeze}: {ROUTES[unfreeze]}")
    print(f"*** `{OUTCOME}` is returned. Arm-level reading is no longer frozen.")
    return d


if __name__ == "__main__":
    d = load()
    print(f"gated load: {len(d):,} rows, {len(d.columns)} columns")
    print(f"  columns: {list(d.columns)}")
    print(f"  `{OUTCOME}` present: {OUTCOME in d.columns}  <- must be False")
    try:
        load(unfreeze="nope")
    except ValueError as e:
        print(f"  bad route refused: {str(e)[:70]}...")
