#!/usr/bin/env python
"""Find figure text that the renderer CUTS, which only the PNG would show.

    uv run python meta/figure_text_audit.py
    uv run python meta/figure_text_audit.py --all      # every measured line
    uv run python meta/figure_text_audit.py meta/M01_displacement

WHY THIS EXISTS
---------------
plotnine neither wraps a title, subtitle or caption nor widens the canvas
for one. A line longer than the figure is cut at the edge, MID-WORD, with
no warning: not in the code, not in the producer's stdout, not in any
assert. **The loss exists only in the rendered PNG.**

Two instances were shipped in one session before either was noticed, both
found by looking at the image and by nothing else. The second is the worse
class: it was a CAPTION carrying the list of booked values the producer
asserts before drawing. Losing its tail leaves a figure that appears to
claim less verification than it performed -- and if the cut lands
mid-clause, a different verification from the one it ran. **A truncated
fence is not a missing fence; it is a fence that reads as complete.**

That is the campaign's recurring signature with the instrument on the
wrong side: the fences we move ONTO panels to make figures honest are
exactly the long lines most likely to be eaten.

WHAT IT MEASURES, AND WHY IT IS NOT A CHARACTER COUNT
------------------------------------------------------
Rendered width in inches, from matplotlib's own text metrics at the
element's actual point size, against the figure's declared width. A
character count cannot do this: `WWWW` and `iiii` differ by more than
three times, and these subtitles mix numbers, backticks and capitals.

THE THRESHOLD IS MEASURED, NOT GUESSED, AND MY FIRST ONE WAS WRONG
------------------------------------------------------------------
The obvious threshold is 100% of the declared width, and it is wrong.
This measurement runs about 7% HIGH against the ink plotnine actually
lays down, so a line measuring 105% of the figure still renders whole.
A first version of this file flagged at 92% and called a dozen intact
figures cut.

So the threshold comes from an experiment rather than from reasoning:
render subtitles of known measured width at three figure widths, read
the rightmost inked pixel, and find where it stops growing.

    figw  measured/figw   rightmost ink     verdict
    12.2      1.00           11.27in        whole
    12.2      1.05           11.78in        whole
    12.2      1.10           12.16in        AT THE EDGE, clipped
     9.0      1.05            8.80in        whole
     9.0      1.10            8.99in        AT THE EDGE, clipped
    14.0      1.05           13.51in        whole
    14.0      1.10           14.00in        AT THE EDGE, clipped

Ink saturates at exactly the figure width once measured width reaches
1.10x, at every width tried. A real shipped line measuring 1.083x was
confirmed whole by eye, so the true boundary sits between those.

`CUT` is therefore reported at 110% and `AT RISK` at 100%: everything in
CUT is clipped, and everything in AT RISK needs an eye on it. Both known
failures (134% and, at its original width, well past 110%) sit in CUT.

AND THE THRESHOLD STILL MISSES, BECAUSE THE BUDGET IS NOT THE FIGURE
---------------------------------------------------------------------
**plotnine anchors the title and subtitle to the PANEL, not the canvas.**
A figure with long y-axis labels has its panel pushed right, and the text
gets what is left. `n_z_ceiling_method.png` measured 98.9% of its figure
width -- below `AT RISK` -- and was cut, because its y-axis carries full
checkpoint names and the panel starts roughly a quarter of the way in.
The text had about 7.8in of a 10.5in figure.

So the static budget is too generous by exactly the width of the axis
labels, which is data-dependent and not knowable from the source. **The
false negatives concentrate in figures with long category labels, which
are the ones most likely to need explanatory prose.** A single percentage
threshold cannot be right for both that figure and a wide one with a
numeric axis.

Which is why `--pixels` exists and is the authority. Treat the static
mode as a screen to run while EDITING a producer, before any render, and
the pixel mode as the verdict on what actually shipped.

WHAT IT CANNOT SEE, STATED RATHER THAN IMPLIED
-----------------------------------------------
- `geom_text` labels near a panel edge. They are data-dependent and this
  is a static reader.
- f-string interpolations, which are measured with a short placeholder,
  so a subtitle splicing in a long computed list reads NARROWER here than
  it renders. This tool's misses are in the permissive direction.
- Any producer whose theme or labs are built dynamically.

**So a clean report is not a guarantee. LOOK AT THE IMAGE.** This narrows
where to look; it does not replace looking.

THE SECOND MODE READS THE PIXELS, WHICH IS WHAT ACTUALLY SHIPPED
-----------------------------------------------------------------
`--pixels` scans rendered PNGs instead of source and reports any whose
ink reaches the right edge of the canvas. That is the direct evidence:
plotnine lays no text against the border by design, so ink at the edge
means something was cut off there.

It sees precisely what the static reader cannot -- f-string
interpolations at their real width, `geom_text` labels running off a
panel, and any figure whose producer builds its theme dynamically. It is
also blind where the other is sure: it cannot say WHICH element was cut
or what the text should have said, and a figure whose geometry
legitimately reaches the border will flag.

Use both. They fail in opposite directions, which is the only reason
running two checks is worth more than running one twice.
"""
import argparse
import ast
import glob
import os
import sys

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

DPI = 100
_FIG = Figure(figsize=(30, 30), dpi=DPI)
_REND = FigureCanvasAgg(_FIG).get_renderer()

#: plotnine's theme_minimal defaults, used when a producer does not set one.
DEFAULT_SIZE = {"title": 13.2, "subtitle": 11.0, "caption": 11.0}
DEFAULT_FIGSIZE = 6.4

CUT, RISK = 1.10, 1.00
#: stand-in width for an f-string interpolation, in characters
PLACEHOLDER = "00000"


def width_in(text, size):
    """Rendered width of one line in inches at `size` points."""
    t = _FIG.text(0, 0, text, size=size)
    w = t.get_window_extent(renderer=_REND).width / DPI
    t.remove()
    return w


def _string_of(node):
    """Literal text of a str node, with f-string slots as placeholders."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            else:
                out.append(PLACEHOLDER)
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _string_of(node.left), _string_of(node.right)
        if a is not None and b is not None:
            return a + b
    return None


def _kwargs_of(call, names):
    return {k.arg: k.value for k in call.keywords if k.arg in names}


def _theme_helpers(tree):
    """{helper name: index of its width parameter}.

    A FALSE-POSITIVE CLASS FOUND BY DRAWING (dario, 2026-08-15). A folder that
    wraps its theme in `def _theme(w, h): ... theme(figure_size=(w, h))` and
    calls `_theme(10, 8.5)` declares its width perfectly well, and the inline
    scan below cannot see it. Assuming 6.4in then inflates every fraction in
    that file by real_width/6.4 -- for `f_figures.py:fig_diverging` that turned
    88% into a reported 137.6% CUT, on two lines that ship intact. The pixel
    mode disagreed and the pixel mode was right.
    """
    out = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        params = [a.arg for a in fn.args.args]
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", getattr(node.func, "attr", "")) != "theme":
                continue
            for k in node.keywords:
                if k.arg == "figure_size" and isinstance(k.value, ast.Tuple):
                    w = k.value.elts[0]
                    if isinstance(w, ast.Name) and w.id in params:
                        out[fn.name] = params.index(w.id)
    return out


def _scan_function(fn, helpers=None):
    """(figure_size width, {element: size}, {element: text}) for one function."""
    figw, sizes, texts = None, {}, {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and helpers:
            nm = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if nm in helpers and figw is None:
                i = helpers[nm]
                if i < len(node.args) and isinstance(node.args[i], ast.Constant):
                    v = node.args[i].value
                    if isinstance(v, (int, float)):
                        figw = float(v)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
        if name == "labs":
            for k in node.keywords:
                if k.arg in ("title", "subtitle", "caption"):
                    s = _string_of(k.value)
                    if s is not None:
                        texts[k.arg] = s
        elif name == "theme":
            for k in node.keywords:
                if k.arg == "figure_size" and isinstance(k.value, ast.Tuple):
                    w = k.value.elts[0]
                    if isinstance(w, ast.Constant):
                        figw = float(w.value)
                elif k.arg in ("plot_title", "plot_subtitle", "plot_caption"):
                    if isinstance(k.value, ast.Call):
                        sz = _kwargs_of(k.value, {"size"}).get("size")
                        if isinstance(sz, ast.Constant):
                            sizes[k.arg.replace("plot_", "")] = float(sz.value)
    return figw, sizes, texts


def audit(paths, show_all=False):
    rows = []
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(os.path.join(p, "**", "*.py"),
                                      recursive=True)))
    for f in files:
        try:
            tree = ast.parse(open(f, errors="ignore").read())
        except SyntaxError:
            continue
        helpers = _theme_helpers(tree)
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            figw, sizes, texts = _scan_function(fn, helpers)
            if not texts:
                continue
            w = figw or DEFAULT_FIGSIZE
            for el, text in texts.items():
                size = sizes.get(el, DEFAULT_SIZE[el])
                for i, line in enumerate(text.split("\n")):
                    if not line.strip():
                        continue
                    frac = width_in(line, size) / w
                    rows.append({"file": f, "fn": fn.name, "el": el,
                                 "line": i, "frac": frac, "figw": w,
                                 "size": size, "text": line,
                                 "declared": figw is not None})
    rows.sort(key=lambda r: -r["frac"])
    #: A LINE WHOSE WIDTH IS UNKNOWN GETS NO VERDICT. Reporting CUT against an
    #: assumed 6.4in states a measurement that was never taken, and a false CUT
    #: costs more than a missing one here: it sends a seat to re-run a producer
    #: and re-wrap prose that was fine. These are listed separately and excluded
    #: from both counts, which is the honest shape of "I could not measure it".
    unmeasured = [r for r in rows if not r["declared"] and r["frac"] >= RISK]
    rows = [r for r in rows if r["declared"]]
    bad = [r for r in rows if r["frac"] >= RISK]
    print(f"scanned {len(files)} files, {len(rows)} text lines in "
          f"{len({(r['file'], r['fn']) for r in rows})} figure functions\n")
    for r in (rows if show_all else bad):
        tag = "CUT    " if r["frac"] >= CUT else "AT RISK"
        note = ""
        print(f"{tag} {100 * r['frac']:5.1f}% of {r['figw']}in  "
              f"{os.path.relpath(r['file'])}:{r['fn']} {r['el']} "
              f"line {r['line']}{note}")
        print(f"          {r['text'][:150]}")
    for r in unmeasured:
        print(f"UNMEASURED     no width  {os.path.relpath(r['file'])}:{r['fn']} "
              f"{r['el']} line {r['line']}  [figure_size not resolvable; "
              f"would be {100 * r['frac']:.0f}% IF the default 6.4in applied]")
        print(f"          {r['text'][:150]}")
    n_cut = sum(1 for r in rows if r["frac"] >= CUT)
    print(f"\n  CUT: {n_cut}   AT RISK: {len(bad) - n_cut}   "
          f"clear: {len(rows) - len(bad)}   UNMEASURED: {len(unmeasured)}")
    if unmeasured:
        print("  UNMEASURED means the width could not be resolved, NOT that the line is long.")
    print("  A clean line here is not a guarantee: f-string slots are measured "
          "with a placeholder\n  and geom_text is not measured at all. LOOK AT "
          "THE IMAGE.")
    return 1 if n_cut else 0


def pixels(paths, margin_px=3, ink=128):
    """Flag rendered PNGs whose ink reaches the right edge of the canvas."""
    import numpy as np
    from PIL import Image

    files = []
    for p in paths:
        files.extend(sorted(glob.glob(os.path.join(p, "**", "*.png"),
                                      recursive=True)))
    flagged = []
    for f in files:
        try:
            im = np.array(Image.open(f).convert("L"))
        except Exception as e:
            print(f"  unreadable: {f} ({e})")
            continue
        edge = im[:, -margin_px:]
        if (edge < ink).any():
            rows = np.where((edge < ink).any(axis=1))[0]
            where = ("header" if rows.min() < 0.25 * im.shape[0]
                     else "footer" if rows.min() > 0.75 * im.shape[0]
                     else "panel")
            flagged.append((f, where, len(rows)))
    for f, where, n in flagged:
        print(f"EDGE INK  {os.path.relpath(f)}  first in the {where} band, "
              f"{n} rows touching")
    print(f"\n  scanned {len(files)} PNGs, {len(flagged)} with ink at the right edge")
    print("  Ink at the border means text was cut there, UNLESS the figure's own\n"
          "  geometry reaches the edge by design. Open each one.")
    return 1 if flagged else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--all", action="store_true",
                    help="print every measured line, not only the flagged")
    ap.add_argument("--pixels", action="store_true",
                    help="scan rendered PNGs for ink at the right edge instead")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    paths = a.paths or [here]
    return pixels(paths) if a.pixels else audit(paths, a.all)


if __name__ == "__main__":
    sys.exit(main())
