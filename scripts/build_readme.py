"""Findings status layer: build README, INDEX, and lint frontmatter.

Usage:
    python scripts/build_readme.py extract   # README.md → findings/F01_*.md
    python scripts/build_readme.py build     # findings/ → README.md (narrative layer)
    python scripts/build_readme.py index     # findings/ → INDEX.md (citation layer)
    python scripts/build_readme.py lint      # validate frontmatter
"""

import re
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
INDEX = ROOT / "INDEX.md"
FINDINGS_DIR = ROOT / "findings"
DATA_DIR = ROOT / "data"

SLUGS = {
    1: "logit_analysis",
    2: "cross_family_logits",
    3: "cross_family_generation",
    4: "step_analysis",
    5: "logit_lens",
    6: "baseline_validation",
    7: "training_data_attribution",
    8: "displacement_taxonomy",
    9: "tulu_vs_llama",
    10: "sft_ablation",
    11: "contradiction",
    12: "fold_geometry",
    13: "jakobsonian_axes",
    14: "syntagmatic_baseline",
    15: "passage_metrics",
    16: "corpus_comparison",
    17: "cross_generation_mmd",
    18: "shannon_entropy",
    19: "bos_entropy",
    20: "who_are_you",
    21: "institutional_alignment",
    22: "circuit_decomposition",
    23: "reasoning_distillation",
    24: "pretraining_emergence",
    25: "temporal_alignment_signature",
    27: "nudging_negative",
    28: "resistance_trajectories",
    31: "permanova_decomposition",
    32: "template_mediated_distributions",
    33: "scale_effects",
    26: "census",
    34: "cross_linguistic_displacement",
    35: "architecture_independence",
    36: "euphemism_vs_proximity",
}

VALID_STATUSES = {"verified", "solid-by-design", "unaudited", "rescoped", "retracted",
                  # Added 2026-07-27. A finding whose audit PASSED but whose pass
                  # rested on evidence since found unsound -- here, a fabricated
                  # commit citation. Not "unaudited" (an audit happened) and not
                  # "verified" (its basis is compromised). Resolves to verified
                  # only via a DATED re-verification against frozen history,
                  # because a SHA verification expires when history is rewritten.
                  "verified-pending-reverification"}
VALID_GRADES = {"A", "B", "C", "D"}
VALID_ROLES = {"finding", "addendum", "capstone", "ledger"}
VALID_INSTRUMENTS = {
    "logit-mass", "resistance", "tagger", "checkpoint",
    "entropy", "geometry", "classification", "generation",
    "embedding", "intervention", "census", "regression",
    # added 2026-07-26 with the F07 ruling: a documentary finding's source is a
    # published report, not a measurement. The controlled set predated the
    # category.
    "documentary",
}


def parse_frontmatter(path):
    """Parse YAML frontmatter from a markdown file. Returns (meta, body)."""
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}, text

    end = text.index("\n---\n", 4)
    yaml_text = text[4:end]
    body = text[end + 5:]
    try:
        meta = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        print(f"  WARN: bad YAML in {path.name}: {e}", file=sys.stderr)
        meta = {}
    return meta, body


def load_all_findings():
    """Load all findings files with parsed frontmatter."""
    findings = []
    for path in sorted(FINDINGS_DIR.glob("F[0-9][0-9]_*.md")):
        meta, body = parse_frontmatter(path)
        stem = path.stem
        num = int(stem.split("_")[0][1:])
        first_line = body.strip().split('\n')[0]
        title = re.sub(r'^#+ *(F\d+[: ] *)?', '', first_line).strip()

        role = meta.get("role", "finding")
        findings.append({
            "path": path,
            "stem": stem,
            "num": num,
            "title": title,
            "role": role,
            "meta": meta,
            "body": body,
        })
    return findings


def _heading_to_anchor(heading):
    anchor = heading.lower()
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    anchor = anchor.strip()
    anchor = re.sub(r' ', '-', anchor)
    return anchor


def _status_badge(meta, children=None):
    """Render a status badge line for a finding section."""
    parts = []
    status = meta.get("status")
    grade = meta.get("grade")
    if status:
        parts.append(f"**Status:** {status}")
    if grade:
        parts.append(f"**Grade:** {grade}")
    if meta.get("superseded_by"):
        parts.append(f"see [{meta['superseded_by']}](findings/{meta['superseded_by']}.md)")
    for sa in meta.get("see_also", []):
        parts.append(f"see also [{sa}](findings/{sa}.md)")
    if children:
        links = [f"[{c['stem']}](findings/{c['path'].name})" for c in children]
        parts.append("Related: " + ", ".join(links))
    if not parts:
        return ""
    return "\n\n> " + " | ".join(parts) + "\n"


FINDINGS_END_MARKER = "<!-- findings:end -->"


def extract():
    """Split README findings into individual files."""
    text = README.read_text()
    findings_start = text.index("## Findings\n")
    post_findings = re.search(r'\n## (?!Findings)', text[findings_start + 20:])
    if post_findings:
        findings_end = findings_start + 20 + post_findings.start()
    else:
        findings_end = len(text)

    findings_text = text[findings_start:findings_end]
    pattern = r'(### (\d+)\. .+?)(?=### \d+\.|$)'
    matches = list(re.finditer(pattern, findings_text, re.DOTALL))

    FINDINGS_DIR.mkdir(exist_ok=True)
    for m in matches:
        num = int(m.group(2))
        content = m.group(1).strip()
        content = re.sub(r'^### \d+\. ', '# F{:02d}: '.format(num), content, count=1)
        slug = SLUGS.get(num, f"finding_{num}")
        filename = f"F{num:02d}_{slug}.md"
        path = FINDINGS_DIR / filename
        path.write_text(content + "\n")
        print(f"  {filename} ({len(content)} chars)")

    print(f"\nExtracted {len(matches)} findings to {FINDINGS_DIR}/")


def build():
    """Rebuild README from findings (narrative layer)."""
    text = README.read_text()

    findings_start = text.index("## Findings\n")
    if FINDINGS_END_MARKER in text:
        findings_end = text.index(FINDINGS_END_MARKER) + len(FINDINGS_END_MARKER)
    else:
        post_findings = re.search(r'\n## (?!Findings)', text[findings_start + 20:])
        if post_findings:
            findings_end = findings_start + 20 + post_findings.start()
        else:
            findings_end = len(text)

    header = text[:findings_start]
    footer = text[findings_end:]

    all_findings = load_all_findings()

    # Primary = role=finding AND no parent. Children = anything with a parent
    # or a non-finding role — linked from parent's badge, not own section.
    primary = [f for f in all_findings
               if f["role"] == "finding" and not f["meta"].get("parent")]
    children_by_num = {}
    for f in all_findings:
        if f["role"] != "finding" or f["meta"].get("parent"):
            children_by_num.setdefault(f["num"], []).append(f)

    finding_headings = []
    findings_parts = ["## Findings\n\n"]

    for f in primary:
        content = f["body"].strip()
        num = f["num"]

        content = re.sub(r'^# F\d+[: ] *', f'### {num}. ', content, count=1)

        first_line = content.split('\n')[0]
        heading_text = first_line.lstrip('#').strip()
        finding_headings.append(heading_text)

        badge = _status_badge(f["meta"], children_by_num.get(num))
        if badge:
            lines = content.split('\n', 1)
            content = lines[0] + badge + (('\n' + lines[1]) if len(lines) > 1 else '')

        content = content.replace('](../figures/', '](figures/')

        findings_parts.append(content)
        findings_parts.append("\n\n")

    findings_parts.append(FINDINGS_END_MARKER)
    new_findings = "".join(findings_parts)

    # Rebuild TOC
    toc_start = header.index("## Table of contents\n")
    toc_body_start = header.index("\n", toc_start) + 1
    toc_end_match = re.search(r'\n## ', header[toc_body_start:])
    toc_end = toc_body_start + toc_end_match.start() if toc_end_match else len(header)

    old_toc = header[toc_body_start:toc_end]
    pre_findings = []
    post_findings_toc = []
    in_findings = False
    for line in old_toc.strip().split('\n'):
        if '- [Findings]' in line:
            pre_findings.append(line)
            in_findings = True
        elif in_findings and line.startswith('  '):
            continue
        elif in_findings and not line.startswith('  '):
            in_findings = False
            post_findings_toc.append(line)
        elif not in_findings:
            if pre_findings:
                post_findings_toc.append(line)
            else:
                pre_findings.append(line)

    findings_toc = []
    for heading in finding_headings:
        anchor = _heading_to_anchor(heading)
        short = heading.split('(')[0].strip().rstrip(':')
        findings_toc.append(f'  - [{short}](#{anchor})')

    new_toc_lines = pre_findings + findings_toc + post_findings_toc
    new_toc = '\n'.join(new_toc_lines) + '\n'

    new_header = header[:toc_body_start] + '\n' + new_toc + '\n'

    # Ensure INDEX.md banner exists
    index_banner = (
        "> **This is the narrative layer.** "
        "For the citation-grade index with status, grade, and chapter mapping, "
        "see [INDEX.md](INDEX.md).\n\n"
    )
    if index_banner not in new_header:
        new_header = new_header.rstrip('\n') + '\n\n' + index_banner

    new_readme = new_header + new_findings + footer
    README.write_text(new_readme)
    print(f"Rebuilt {README} ({len(new_readme)} chars, {len(primary)} findings)")


def _grade_label(grade):
    labels = {
        "A": "Campaign-verified (controls + TM review)",
        "B": "Solid by design (measurement-only)",
        "C": "Unaudited",
        "D": "Superseded or retracted",
    }
    return labels.get(grade, grade)


def _trunc(text, maxlen=60):
    """Truncate at word boundary."""
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen].rfind(' ')
    if cut <= 0:
        cut = maxlen
    return text[:cut].rstrip('.,;:') + '...'


def _sort_parent_first(findings):
    """Sort so parents appear before their children within the same F-number."""
    def sort_key(f):
        has_parent = 1 if (f["meta"].get("parent") or f["role"] != "finding") else 0
        role_order = {"finding": 0, "addendum": 1, "capstone": 2, "ledger": 3}
        return (f["num"], has_parent, role_order.get(f["role"], 9))
    return sorted(findings, key=sort_key)


def index():
    """Generate INDEX.md — the citation layer."""
    all_findings = _sort_parent_first(load_all_findings())

    lines = [
        "# Findings Index",
        "",
        "Citation-grade index with status, grade, and chapter mapping. "
        "For the narrative layer, see [README.md](README.md).",
        "",
    ]

    # Master table
    lines.append("## Master table")
    lines.append("")
    lines.append("| F# | Title | Status | Grade | Chapters | Citation doc |")
    lines.append("|---|---|---|---|---|---|")

    for f in all_findings:
        m = f["meta"]
        num_label = f"F{f['num']:02d}"
        role = f["role"]
        is_child = m.get("parent") or role != "finding"
        if is_child:
            suffix = f" ({role})" if role != "finding" else " (sub)"
            num_label = f"  {num_label}{suffix}"

        status = m.get("status", "—")
        grade = m.get("grade", "—")
        chapters = ", ".join(m.get("chapters", [])) or "—"

        superseded_by = m.get("superseded_by")
        citation = f"[{f['stem']}](findings/{f['path'].name})"
        if superseded_by:
            citation += f" → [{superseded_by}](findings/{superseded_by}.md)"

        lines.append(f"| {num_label} | {_trunc(f['title'])} | {status} | {grade} | {chapters} | {citation} |")

    # By grade
    lines.append("")
    lines.append("## By grade")
    lines.append("")
    for grade in ["A", "B", "C", "D"]:
        members = [f for f in all_findings if f["meta"].get("grade") == grade]
        if not members:
            continue
        lines.append(f"### Grade {grade}: {_grade_label(grade)}")
        lines.append("")
        for f in members:
            lines.append(f"- [{f['stem']}](findings/{f['path'].name}) — {_trunc(f['title'])}")
        lines.append("")

    ungraded = [f for f in all_findings if not f["meta"].get("grade")]
    if ungraded:
        lines.append("### Ungraded")
        lines.append("")
        for f in ungraded:
            lines.append(f"- [{f['stem']}](findings/{f['path'].name}) — {_trunc(f['title'])}")
        lines.append("")

    # By chapter
    lines.append("## By chapter")
    lines.append("")
    chapter_map = {}
    for f in all_findings:
        for ch in f["meta"].get("chapters", []):
            chapter_map.setdefault(ch, []).append(f)

    for ch in sorted(chapter_map.keys()):
        lines.append(f"### {ch}")
        lines.append("")
        for f in chapter_map[ch]:
            grade = f["meta"].get("grade", "?")
            lines.append(f"- [{f['stem']}](findings/{f['path'].name}) [{grade}]")
        lines.append("")

    INDEX.write_text("\n".join(lines) + "\n")
    print(f"Generated {INDEX} ({len(all_findings)} entries)")


def lint():
    """Validate frontmatter. Returns exit code."""
    all_findings = load_all_findings()
    errors = []
    warnings = []

    for f in all_findings:
        m = f["meta"]
        name = f["path"].name

        # Missing frontmatter: warn (not fail) until triage sweep
        if not m:
            warnings.append(f"{name}: no frontmatter")
            continue
        if "status" not in m:
            warnings.append(f"{name}: missing status")
        if "grade" not in m:
            warnings.append(f"{name}: missing grade")

        # Validate values
        if m.get("status") and m["status"] not in VALID_STATUSES:
            errors.append(f"{name}: invalid status '{m['status']}'")
        if m.get("grade") and m["grade"] not in VALID_GRADES:
            errors.append(f"{name}: invalid grade '{m['grade']}'")
        if m.get("role") and m["role"] not in VALID_ROLES:
            errors.append(f"{name}: invalid role '{m['role']}'")
        # VALID_INSTRUMENTS was defined but never checked, so 54 out-of-vocabulary
        # values linted clean and the vocabulary read as authoritative while
        # constraining nothing. Enforced 2026-07-26.
        for inst in m.get("instruments") or []:
            if inst not in VALID_INSTRUMENTS:
                errors.append(f"{name}: invalid instrument '{inst}'")

        # Rescoped/retracted must have superseded_by
        if m.get("status") in ("rescoped", "retracted") and not m.get("superseded_by"):
            errors.append(f"{name}: status '{m['status']}' requires superseded_by")

        # Addendum must have parent
        if m.get("role") == "addendum" and not m.get("parent"):
            errors.append(f"{name}: role 'addendum' requires parent")

        # Grade A: check data files exist
        if m.get("grade") == "A":
            for data_file in m.get("data", []):
                if not (DATA_DIR / data_file).exists() and not (ROOT / "data" / data_file).exists():
                    errors.append(f"{name}: grade A but data file missing: {data_file}")

    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}")

    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    else:
        print(f"\nOK ({len(warnings)} warning(s))")
        return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_readme.py [extract|build|index|lint]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "extract":
        extract()
    elif cmd == "build":
        lint_code = lint()
        if lint_code:
            print("\nLint errors found — fix before building.")
            sys.exit(lint_code)
        build()
    elif cmd == "index":
        index()
    elif cmd == "lint":
        sys.exit(lint())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
