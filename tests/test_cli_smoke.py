"""CLI smoke tests — the argparse tree builds and every subcommand's --help
works. Catches duplicate arguments, bad set_defaults wiring, and import-time
breakage across the ~24 subcommands without loading any models.
"""

import subprocess
import sys

import pytest


def _run(args):
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from malign_logits.cli import main; "
         "sys.argv = ['malign'] + sys.argv[1:]; main()",
         *args],
        capture_output=True, text=True, timeout=120)


def test_top_level_help():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "malign" in r.stdout


# Every registered subcommand — a --help that exits 0 proves the subparser
# and its arguments were constructed without error.
SUBCOMMANDS = [
    "serve", "ui", "info", "battery", "generate-battery", "taxonomy",
    "logit-lens", "step-analysis", "trajectory", "precompute", "surprisal",
    "embed", "ingest", "bos-generate", "api-generate", "vllm-generate",
    "produce-all", "ablation", "topic-drift", "deep-probe", "circuit",
    "cloud", "probe", "download-models",
]


@pytest.mark.parametrize("sub", SUBCOMMANDS)
def test_subcommand_help(sub):
    r = _run([sub, "--help"])
    assert r.returncode == 0, f"`malign {sub} --help` failed:\n{r.stderr}"
    assert sub.split("-")[0] in (r.stdout + r.stderr).lower() or r.stdout


def test_unknown_subcommand_errors():
    r = _run(["definitely-not-a-command"])
    assert r.returncode != 0
