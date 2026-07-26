"""Download -> extract logits for every project prompt -> cache -> delete -> next.

For model suites too large to hold on disk (e.g. the 31-model Tulu 2.5 suite at
~26 GB each). Each model is staged one at a time in a disposable directory,
every prompt the project has ever used is scored and written to the logit stash,
then the weights are deleted before the next model is fetched.

Resumable: prompts already in the stash are skipped, and a model whose prompts
are all cached is skipped entirely. Safe to kill and restart at any point.

Usage:
    uv run python scripts/cache_logits_insitu.py [--suite tulu25] [--limit N]
                                                 [--staging DIR] [--dry-run]
"""
import argparse, gc, json, os, shutil, sys, time
import numpy as np
import torch

from malign_logits.cache import get_cache
from malign_logits.core import _apply_mode
from malign_logits.models import get_base_logits, load_model

PROMPTS_FILE = "data/all_project_prompts.json"
DEFAULT_STAGING = "/Volumes/chambers/hf_staging"

# Both addressing modes are extracted while the weights are staged, because the
# download is the expensive part and re-fetching 26 GB per model to add a mode
# later would cost far more than the extra forward passes now.
#   raw      - bare text, the displacement/contradiction instrument
#   continue - chat template wrapping "Continue this text: {prompt}" (F32,
#              the template-mediated distributions / three-addressing-systems work)
MODES = ["raw", "continue"]

# Tulu 2.5 (AI2, "Unpacking DPO and PPO", arXiv 2406.09279). All share the SFT
# checkpoint allenai/tulu-2-13b (from meta-llama/Llama-2-13b-hf), so algorithm
# and preference dataset are the only axes that vary.
MATCHED = ["uf-mean", "hh-rlhf-60k", "nectar-60k", "stackexchange-60k",
           "chatbot-arena-2023"]
DPO_ONLY = ["alpacafarm-gpt4-pref", "alpacafarm-human-pref", "argilla-orca-pairs",
            "capybara", "chatbot-arena-2024", "helpsteer", "hh-rlhf", "nectar",
            "prm-phase-2", "shp2", "stackexchange", "uf-overall"]
# NB: the "-value" repos in this suite are NOT policies. They are the PPO value
# networks (LlamaForTokenClassification, a scoring head, no lm_head), and the
# 70B ones are 137 GB. AutoModelForCausalLM will happily load them with a
# randomly initialised output head and emit plausible-looking garbage logits,
# so they are excluded here and blocked again by the architecture guard below.
PPO_ONLY = ["uf-mean-13b-mix-rm", "uf-mean-70b-mix-rm", "uf-mean-70b-uf-rm",
            "uf-mean-70b-uf-rm-mixed-prompts"]

def tulu25_models():
    """Priority order: shared ancestor, then matched DPO/PPO pairs, then the
    curriculum axis, then the reward-model axis, then the gated base last."""
    out = ["allenai/tulu-2-13b"]
    for d in MATCHED:                     # the clean algorithm contrast
        out.append(f"allenai/tulu-v2.5-dpo-13b-{d}")
        out.append(f"allenai/tulu-v2.5-ppo-13b-{d}")
    out += [f"allenai/tulu-v2.5-dpo-13b-{d}" for d in DPO_ONLY]
    out += [f"allenai/tulu-v2.5-ppo-13b-{d}" for d in PPO_ONLY]
    out.append("meta-llama/Llama-2-13b-hf")   # gated; last so it cannot block
    return out

SUITES = {"tulu25": tulu25_models}

WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json"]
FALLBACK_PATTERNS = ["*.bin", "*.bin.index.json"]
CONFIG_PATTERNS = ["*.json", "*.model", "tokenizer*", "*.txt"]


def is_causal_lm(repo):
    """True only if the checkpoint really is a causal LM.

    AutoModelForCausalLM does not refuse a checkpoint without an lm_head; it
    randomly initialises one and returns a model whose logits are noise. That
    failure is silent and the output looks like data, so it is checked here
    from config.json (a few KB) before any weights are fetched.
    """
    from huggingface_hub import hf_hub_download
    try:
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
    except Exception:
        return True          # can't tell; let the load attempt decide
    arch = cfg.get("architectures") or []
    return any("CausalLM" in a for a in arch) if arch else True


def repo_uses_safetensors(repo):
    from huggingface_hub import list_repo_files
    try:
        return any(f.endswith(".safetensors") for f in list_repo_files(repo))
    except Exception:
        return True


def staged_dir(staging, repo):
    return os.path.join(staging, "models--" + repo.replace("/", "--"))


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def run(models, prompts, staging, dry_run=False, log=print):
    cm = get_cache()
    os.makedirs(staging, exist_ok=True)
    from huggingface_hub import snapshot_download

    done = skipped = failed = 0
    for i, repo in enumerate(models, 1):
        # (mode, prompt) pairs not yet in the stash
        missing = [(m, p) for m in MODES for p in prompts
                   if not cm.has_logits(repo, p, mode=m)]
        head = f"[{i}/{len(models)}] {repo}"
        if not missing:
            log(f"{head}: complete ({len(prompts)} prompts x {len(MODES)} modes), skipping")
            skipped += 1
            continue
        if not is_causal_lm(repo):
            log(f"{head}: NOT a causal LM (no lm_head) — skipping, would emit "
                f"noise from a randomly initialised head")
            skipped += 1
            continue
        per_mode = {m: sum(1 for mm, _ in missing if mm == m) for m in MODES}
        log(f"{head}: {len(missing)} missing ({per_mode}); "
            f"{free_gb(staging):.0f} GB free on staging")
        if dry_run:
            continue

        target = staged_dir(staging, repo)
        try:
            patterns = (WEIGHT_PATTERNS if repo_uses_safetensors(repo)
                        else FALLBACK_PATTERNS) + CONFIG_PATTERNS
            t0 = time.time()
            snapshot_download(repo, cache_dir=staging, allow_patterns=patterns,
                              max_workers=8)
            log(f"    downloaded in {time.time()-t0:.0f}s")

            model, tok = load_model(repo, cache_dir=staging)

            # Guard: _apply_mode falls back to the bare prompt when a model has
            # no chat template. Writing that under mode="continue" would put a
            # duplicate of the raw vector in the stash under a different key,
            # which is worse than a gap. Drop the mode instead, loudly.
            probe = "She was so angry she wanted to"
            usable = {m for m in MODES
                      if m == "raw" or _apply_mode(probe, tok, m) != probe}
            if usable != set(MODES):
                log(f"    ! no usable chat template; skipping modes "
                    f"{sorted(set(MODES) - usable)} for this model")
                missing = [(m, p) for m, p in missing if m in usable]

            t0, n_ok, n_err = time.time(), 0, 0
            for mode, p in missing:
                try:
                    # key the stash by the ORIGINAL prompt + mode; the template
                    # is applied only to the text fed through the forward pass
                    text = p if mode == "raw" else _apply_mode(p, tok, mode)
                    # MUST be numpy: the stash serializer silently round-trips a
                    # torch tensor to tensor([]) with no error, so storing the
                    # tensor directly writes an empty vector that reads back as
                    # data. Every other call site in the project does .cpu().numpy().
                    lg = get_base_logits(model, tok, text).cpu().numpy()
                    cm.set_logits(repo, p, lg, mode=mode)
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    if n_err <= 3:
                        log(f"    ! {mode} prompt failed ({e.__class__.__name__}): {p[:50]!r}")
            log(f"    cached {n_ok} logit vectors across {len(MODES)} modes "
                f"in {time.time()-t0:.0f}s" + (f" ({n_err} failed)" if n_err else ""))

            # Post-write integrity check: read one vector back out of the stash
            # and confirm it is a full-vocabulary distribution. A write that
            # round-trips to an empty array is indistinguishable from success
            # at the call site, so it is verified here rather than assumed.
            chk_mode, chk_prompt = missing[0]
            back = cm.get_logits(repo, chk_prompt, mode=chk_mode)
            n_back = 0 if back is None else int(np.asarray(back).size)
            if n_back < 1000:
                log(f"    !! INTEGRITY FAIL: readback is {n_back} elements, "
                    f"expected full vocab. Entries for this model are unusable.")
                failed += 1
                continue
            log(f"    verified readback: {n_back} logits")
            del model, tok
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            done += 1
        except Exception as e:
            log(f"    FAILED: {e.__class__.__name__}: {e}")
            failed += 1
        finally:
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
                log(f"    deleted staged weights; {free_gb(staging):.0f} GB free")
    log(f"\nDONE. {done} models cached, {skipped} already complete, {failed} failed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="tulu25", choices=sorted(SUITES))
    ap.add_argument("--models", nargs="*", help="explicit model IDs (overrides --suite)")
    ap.add_argument("--staging", default=DEFAULT_STAGING)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prompts = json.load(open(PROMPTS_FILE))
    models = a.models or SUITES[a.suite]()
    if a.limit:
        models = models[:a.limit]
    print(f"{len(prompts)} prompts x {len(models)} models", flush=True)

    def log(msg):
        print(msg, flush=True)
    run(models, prompts, a.staging, dry_run=a.dry_run, log=log)


if __name__ == "__main__":
    main()
