#!/usr/bin/env bash
# Falcon-H1 repair handover: transfer and PROVE it, both file types.
#
# WHY THE .f16 HALF IS THE POINT. The 2026-08-01 handover compared `*.jsonl`
# ONLY -- 4% of the bytes -- and both cloud instances were destroyed before the
# omission was noticed. `data/logit_index_provenance.json` therefore carries,
# permanently, "SELF-CONSISTENT, NOT SOURCE-VERIFIED", a caveat every reader of
# that index inherits and which can never now be discharged. This run is 2.7 GB
# against that run's 49 GB, so there is no excuse for repeating it.
#
# ESCROW DISCIPLINE: copy -> hash the COPY -> RE-READ THE SOURCE and hash it
# again. Hashing the source once before transfer cannot detect a source that
# changed during transfer; the second read is what makes the pair meaningful.
#
# THE BOX IS NOT DESTROYED BY THIS SCRIPT. It exits non-zero on any mismatch and
# leaves the instance running, because a failed verification is the one moment
# the source is still needed. Killing is a separate, deliberate command.
set -uo pipefail

PORT="${PORT:-30110}"; HOST="${HOST:-root@ssh3.vast.ai}"
REMOTE=/workspace/twp_falcon
LOCAL="${LOCAL:-data/raw/falcon_h1_repair}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=no"
EXPECT_CELLS="${EXPECT_CELLS:-5166}"

say() { printf '%s\n' "$*"; }
fail() { say "*** FAIL: $*"; exit 1; }

mkdir -p "$LOCAL"

say "== 1. SOURCE HASHES, PRE-TRANSFER =="
$SSH "$HOST" "cd $REMOTE && md5sum *.jsonl *.f16" 2>/dev/null \
  | grep -v 'Welcome to vast\|Have fun' | sort > /tmp/fal_src1.txt
cat /tmp/fal_src1.txt
[ -s /tmp/fal_src1.txt ] || fail "no remote files"

say ""
say "== 2. RSYNC DOWN =="
rsync -az --info=stats1 -e "$SSH" "$HOST:$REMOTE/" "$LOCAL/" || fail "rsync rc=$?"

say ""
say "== 3. LOCAL HASHES =="
( cd "$LOCAL" && md5sum *.jsonl *.f16 2>/dev/null || md5 -r *.jsonl *.f16 ) \
  | awk '{print $1"  "$2}' | sed 's#.*/##' | sort > /tmp/fal_local.txt
cat /tmp/fal_local.txt

say ""
say "== 4. RE-READ THE SOURCE AND RE-HASH (escrow: catches a source that moved) =="
$SSH "$HOST" "cd $REMOTE && md5sum *.jsonl *.f16" 2>/dev/null \
  | grep -v 'Welcome to vast\|Have fun' | sort > /tmp/fal_src2.txt

diff /tmp/fal_src1.txt /tmp/fal_src2.txt >/dev/null \
  || fail "THE SOURCE CHANGED DURING TRANSFER -- pre and post hashes differ"
say "source stable across the transfer window: OK"

diff /tmp/fal_src2.txt /tmp/fal_local.txt >/dev/null \
  || { say "--- remote vs local ---"; diff /tmp/fal_src2.txt /tmp/fal_local.txt; \
       fail "REMOTE AND LOCAL DIFFER"; }
say "remote == local on EVERY file, .jsonl AND .f16: OK"

say ""
say "== 5. STRUCTURE, not just bytes =="
python3 - "$LOCAL" "$EXPECT_CELLS" <<'PY' || exit 1
import glob, json, os, sys
root, expect = sys.argv[1], int(sys.argv[2])
total = 0; bad = []
for jf in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
    rows = [json.loads(l) for l in open(jf)]
    total += len(rows)
    ff = jf[:-6] + ".f16"
    dims = {r["logit_dim"] for r in rows if r.get("logit_dim")}
    lrs  = [r["logit_row"] for r in rows if r.get("logit_row") is not None]
    if len(dims) != 1: bad.append(f"{jf}: dim not constant {dims}"); continue
    dim = dims.pop()
    want = (max(lrs) + 1) * dim * 2
    got = os.path.getsize(ff)
    empty = sum(1 for r in rows if not (r.get("rows") or []))
    cdt = {r.get("compute_dtype") for r in rows}
    print(f"  {os.path.basename(jf)}")
    print(f"     cells {len(rows):,}  rows_with_logits {len(lrs):,}  dim {dim}")
    print(f"     .f16 {got:,} B  expected {want:,} B  {'OK' if got==want else 'MISMATCH'}")
    print(f"     EMPTY cells {empty}   compute_dtype {cdt}")
    if got != want: bad.append(f"{jf}: size {got} != {want}")
    if empty:       bad.append(f"{jf}: {empty} EMPTY cells -- the defect is NOT repaired")
    if cdt != {"bfloat16"}: bad.append(f"{jf}: compute_dtype {cdt}, expected bfloat16")
print(f"\n  TOTAL CELLS {total:,}   expected {expect:,}")
if total != expect: bad.append(f"cell count {total} != {expect}")
if bad:
    print("\n*** STRUCTURAL FAILURES:"); [print("   ", b) for b in bad]; sys.exit(1)
print("  structure OK on every file")
PY

say ""
say "== VERIFIED: bytes identical both file types, structure sound. =="
say "   The instance is STILL RUNNING. Kill it deliberately:"
say "     PATH=.venv/bin:\$PATH malign cloud --yes stop"
