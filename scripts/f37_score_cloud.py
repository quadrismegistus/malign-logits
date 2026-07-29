"""Self-contained reward/cost scorer for the cloud instance.

    python f37_score_cloud.py --judge pku-reward --limit 1000     # measure first
    python f37_score_cloud.py --judge pku-reward

NO PROJECT DEPENDENCIES. Reads one parquet, writes one parquet. In particular NO
hashstash: the on-disk format options are encoded into the stash path, so an
unpinned open resolves to a different, EMPTY store and raises nothing. That trap
produced two phantom stores in this project's data tree today; on a remote instance
with resume-by-key-parity it would either rescore 256k items or score none, silently
either way. The pinned open stays on the machine where the rule is enforced.

KEY is (idx, model, prompt, temp, temp_type) — five fields, not four. `temp` is
stored as both int 1 and float 1.0 for 55 items in the source stash: two distinct
keys holding two distinct generations, which collide the moment the key is
flattened. The type is carried so the round trip back into the stash is exact.

THE FRAME IS THE SCORING FRAME AND IS DECLARED, NOT HIDDEN. The PKU judges are
trained on (prompt, response) in a conversation template, so response-only input is
out of distribution for them and every item gets the template. But 93% of this
corpus is non-dialogic continuation — narrative stems the model completed as a
document, never as a reply. Wrapping those as USER/ASSISTANT scores them under a
frame the text never inhabited. That is arguably the finding's own shape (the judge
reads everything as an assistant reply), and it is a property of the measurement
that must travel with every number it produces.

RESUME is by key parity against the output parquet: an interrupted run re-reads what
it wrote and scores only the remainder. Writes are temp-then-rename, because a kill
during a read-modify-overwrite truncates the file and takes the whole run with it.
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import LlamaModel, LlamaPreTrainedModel

KEY = ["idx", "model", "prompt", "temp", "temp_type"]

JUDGES = {
    "pku-reward": ("PKU-Alignment/beaver-7b-v1.0-reward", "llama-score"),
    "pku-cost":   ("PKU-Alignment/beaver-7b-v1.0-cost",   "llama-score"),
    "oasst-deberta": ("OpenAssistant/reward-model-deberta-v3-large-v2", "seqcls"),
    "ultrarm": ("openbmb/UltraRM-13b", "llama-score"),
}
PKU_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {user} ASSISTANT:{assistant}"
# A BOS item has no user turn to put in the template; these are the BOS keys.
BOS_PROMPTS = {"", "<|endoftext|>", "<s>"}
BOS_STAND_IN = "Write something."


class LlamaForScore(LlamaPreTrainedModel):
    """PKU's scoring head. Kept here so the instance needs no project code."""
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        self.model = LlamaModel(config)
        self.score_dim = getattr(config, "score_dim", 1)
        self.score_head = nn.Linear(config.hidden_size, self.score_dim,
                                    bias=getattr(config, "score_bias", True))

    def forward(self, input_ids, attention_mask=None, **kw):
        h = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        scores = self.score_head(h)
        # score at the last non-pad position of each sequence
        idx = attention_mask.sum(dim=1) - 1 if attention_mask is not None \
            else torch.full((h.size(0),), h.size(1) - 1, device=h.device)
        return scores[torch.arange(h.size(0), device=h.device), idx].squeeze(-1)


def build_text(prompt, text, framed=True):
    if not framed:
        return str(text)
    user = BOS_STAND_IN if str(prompt) in BOS_PROMPTS else str(prompt)
    return PKU_TEMPLATE.format(user=user, assistant=" " + str(text))


def load(judge):
    model_id, kind = JUDGES[judge]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "llama-score":
        m = LlamaForScore.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        m = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float16)
    m = m.to(dev).eval()
    # RESOLVED model id recorded per score, never the requested alias: an alias
    # whose referent moves silently is how this project lost the ability to say
    # which instrument produced its published numbers.
    rev = getattr(getattr(m, "config", None), "_commit_hash", None) or "unknown"
    print(f"{judge}: {model_id} rev={rev} on {dev}", flush=True)
    return m, tok, kind, model_id, rev, dev


def main(a):
    src = pd.read_parquet(a.src)
    print(f"source {len(src):,} rows", flush=True)
    out_path = a.out or f"f37_scores_{a.judge}{'_unframed' if not a.framed else ''}.parquet"

    done = pd.DataFrame(columns=KEY)
    if os.path.exists(out_path):
        done = pd.read_parquet(out_path)
        print(f"resuming: {len(done):,} already scored", flush=True)
    todo = src.merge(done[KEY], on=KEY, how="left", indicator=True)
    todo = todo[todo._merge == "left_only"].drop(columns="_merge")
    assert len(todo) <= len(src), "resume grew the frame"
    if a.limit:
        todo = todo.head(a.limit)
    print(f"to score: {len(todo):,}", flush=True)
    if not len(todo):
        return

    m, tok, kind, model_id, rev, dev = load(a.judge)
    texts = [build_text(p, t, a.framed) for p, t in zip(todo.prompt, todo.text)]

    # INCREMENTAL WRITES. The first version wrote once at the end, so a crash at
    # 99% returned nothing and cost the whole GPU-hour. Resume is by key parity
    # against this file, so a partial write is not a broken state -- it is exactly
    # the state resume was built to read. Flush is temp-then-rename for the same
    # reason it is everywhere else here: a kill mid-write truncates the file and
    # takes the run with it.
    flushed = 0

    def flush(scores):
        nonlocal flushed
        if len(scores) == flushed:
            return
        rec = todo.iloc[:len(scores)][KEY].copy()   # todo order == texts order
        rec["score"] = scores
        rec["judge"] = a.judge
        rec["judge_model_id"] = model_id
        rec["judge_revision"] = rev
        rec["framed"] = a.framed
        out = pd.concat([done, rec], ignore_index=True) if len(done) else rec
        tmp = out_path + ".tmp"
        out.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, out_path)
        flushed = len(scores)
        print(f"  flushed {len(out):,} rows -> {out_path}", flush=True)

    scores, t0 = [], time.time()
    for i in range(0, len(texts), a.batch):
        b = texts[i:i + a.batch]
        enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.maxlen).to(dev)
        with torch.no_grad():
            if kind == "llama-score":
                s = m(**enc)
            else:
                s = m(**enc).logits.squeeze(-1)
        scores.extend(s.float().cpu().tolist())
        if (i // a.batch) % 20 == 0 and i:
            el = time.time() - t0
            print(f"  {i:,}/{len(texts):,}  {i/el:.1f} it/s  "
                  f"eta {(len(texts)-i)/(i/el)/60:.0f} min", flush=True)
        if len(scores) - flushed >= a.flush_every:
            flush(scores)

    el = time.time() - t0
    print(f"scored {len(scores):,} in {el/60:.1f} min = {len(scores)/el:.1f} it/s",
          flush=True)
    flush(scores)
    print(f"wrote {out_path}  {flushed + len(done):,} rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(JUDGES))
    ap.add_argument("--src", default="f37_gens_for_scoring.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flush-every", dest="flush_every", type=int, default=20_000)
    ap.add_argument("--framed", dest="framed", action="store_true", default=True)
    ap.add_argument("--unframed", dest="framed", action="store_false")
    main(ap.parse_args())
