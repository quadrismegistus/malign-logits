"""Extract findings from README.md into findings/ and rebuild README from parts.

Usage:
    python scripts/build_readme.py extract   # README.md → findings/F01_*.md
    python scripts/build_readme.py build     # findings/ + README parts → README.md
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
FINDINGS_DIR = ROOT / "findings"

# Finding number → slug mapping (derived from section titles)
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
}


def extract():
    """Split README findings into individual files."""
    text = README.read_text()

    # Find the findings section
    findings_start = text.index("## Findings\n")
    # Find the next ## section after findings (Installation, etc.)
    post_findings = re.search(r'\n## (?!Findings)', text[findings_start + 20:])
    if post_findings:
        findings_end = findings_start + 20 + post_findings.start()
    else:
        findings_end = len(text)

    findings_text = text[findings_start:findings_end]

    # Split on ### N. headings
    pattern = r'(### (\d+)\. .+?)(?=### \d+\.|$)'
    matches = list(re.finditer(pattern, findings_text, re.DOTALL))

    FINDINGS_DIR.mkdir(exist_ok=True)

    for m in matches:
        num = int(m.group(2))
        content = m.group(1).strip()

        # Convert ### to # for standalone file
        content = re.sub(r'^### \d+\. ', '# F{:02d}: '.format(num), content, count=1)

        slug = SLUGS.get(num, f"finding_{num}")
        filename = f"F{num:02d}_{slug}.md"
        path = FINDINGS_DIR / filename
        path.write_text(content + "\n")
        print(f"  {filename} ({len(content)} chars)")

    print(f"\nExtracted {len(matches)} findings to {FINDINGS_DIR}/")


def build():
    """Rebuild README from parts: header + findings + footer."""
    text = README.read_text()

    # Find boundaries
    findings_start = text.index("## Findings\n")
    post_findings = re.search(r'\n## (?!Findings)', text[findings_start + 20:])
    if post_findings:
        findings_end = findings_start + 20 + post_findings.start()
    else:
        findings_end = len(text)

    header = text[:findings_start]
    footer = text[findings_end:]

    # Collect all finding files, sorted
    finding_files = sorted(FINDINGS_DIR.glob("F[0-9][0-9]_*.md"))

    findings_parts = ["## Findings\n\n"]
    for f in finding_files:
        content = f.read_text().strip()
        # Extract finding number from filename
        num = int(f.stem.split("_")[0][1:])

        # Convert # F01: back to ### 1. for README
        content = re.sub(r'^# F\d+: ', f'### {num}. ', content, count=1)

        # Fix figure paths: ../figures/ → figures/ for README context
        content = content.replace('](../figures/', '](figures/')

        findings_parts.append(content)
        findings_parts.append("\n\n")

    new_findings = "".join(findings_parts)
    new_readme = header + new_findings + footer

    README.write_text(new_readme)
    print(f"Rebuilt {README} ({len(new_readme)} chars, {len(finding_files)} findings)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_readme.py [extract|build]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "extract":
        extract()
    elif cmd == "build":
        build()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
