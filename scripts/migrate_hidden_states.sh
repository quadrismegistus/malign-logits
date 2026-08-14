#!/usr/bin/env bash
# Move the .hidden.f32 residual-stream sidecars to /Volumes/diderot, mirrored,
# leaving absolute symlinks behind so every reader keeps working.
#
#   MIGRATE_SET=logits scripts/migrate_hidden_states.sh list    # the .f16 tier
#   scripts/migrate_hidden_states.sh list      what would move, and how big
#   scripts/migrate_hidden_states.sh copy      rsync to diderot, with progress
#   scripts/migrate_hidden_states.sh verify    OPTIONAL batch checksum
#   scripts/migrate_hidden_states.sh link      size-check, then symlink (--cmp for bytes)
#   scripts/migrate_hidden_states.sh status    how far along, any inconsistency
#   scripts/migrate_hidden_states.sh restore   bring them all back, undo links
#
# COPY THEN LINK. This header has been wrong twice about its own gates and
# both corrections are recorded rather than overwritten, because a runbook
# that quietly rewrites its safety story is worse than one that shows the
# argument.
#
#   v1  `link` REFUSED without a `verify` stamp. But `link` ALSO cmp'd every
#       file before deleting it, so 87 GB was read twice for one guarantee and
#       the WEAKER pass was the gate: a batch stamp says the set was good at
#       some earlier moment, a per-file check says THIS file is good at the
#       instant it is removed. That is the [6005] distinction exactly -- rsync
#       right about what existed when it ran, wrong about what existed when it
#       was read.
#   v2  stamp requirement removed, per-file cmp became the gate.
#   v3  RH's call: the byte comparison is behind `--cmp` and the DEFAULT is a
#       size check. 87 GB read once at copy time and never again.
#
# WHAT THE DEFAULT GUARANTEES, stated plainly because it is weaker than v2:
# the destination exists and matches the source size, checked per file
# immediately before that file is deleted. Equal-length corruption is not
# covered. It is a narrow window -- rsync verifies a whole-file checksum on
# every file as it transfers, so the exposure is between a verified write and
# a read minutes later -- but it is not zero, and after `link` the copies on
# diderot are the only ones.
#
# USE `link --cmp` when the copy is not fresh, the volume has been remounted,
# or anything else has written to it since. Then size is not enough.
#
# WHY SYMLINKS AND NOT A MOVED DIRECTORY. `lens_ratio_by_layer.scan_hidden()`
# globs `data/**/*.jsonl` recursively and claims any file with a sibling
# `.hidden.f32`. The .jsonl indexes are small and stay put; only the big
# sidecars move, and a per-FILE symlink keeps the sibling relationship intact.
# Recursive glob follows symlinked paths -- tested under this repo's own
# interpreter (3.11.15), not the system one, because that behaviour is
# version-dependent and the repo's Python is what runs.
#
# WHAT DEPENDS ON THE `logits` SET, measured 2026-08-14 and worth having here
# because it is the opposite of what a reader expects from 165 GiB:
#   ZERO live readers for data/raw/verse_fleet/*.f16. `contradiction_null.py
#   --logits` is a real read of the .f16/.f32 STASH, but that resolves through
#   data/logit_dir_resolution.json, whose dirs are cloud_run_20260801 and
#   f11_twp -- the index contains no verse_fleet entry, so no code path
#   existing today reaches those 58.9 GiB. One BOOKED reader: the closure
#   rider (plan_verse_fleet.md:45), unwritten. Whoever writes it must extend
#   the dirmap or read the directory directly.
#   So nothing breaks on the day this moves, and there is NO SMOKE TEST for
#   it either -- only the file count and the bytes.
#
# WHAT DEPENDS ON THESE FILES, so nobody has to rediscover it:
#   lens_ratio_by_layer.py   RESUMABLE and 90 of 370 models scored; the other
#                            280 need these sidecars. Detaching the volume
#                            does not error -- scan_hidden simply enumerates
#                            fewer models and reports success.
#   l3_geometry.py           outputs tracked in git
#   m05_pole_sep.py          calls LENS.scan_hidden() and reads .f32 directly
#
# AND AN EXPOSURE THIS SCRIPT DOES NOT FIX: lens_group_layer.jsonl (52 MB) and
# lens_prompt_layer.jsonl (109 MB) are UNTRACKED. The 90 models already scored
# exist in exactly one place, and their only regeneration path is the files
# being moved here. Back those up separately.
set -euo pipefail

# THE SET TO MOVE. `hidden` is the original and stays the default so every
# invocation in the record still means what it did. `logits` was added once
# diderot had room; the script's NAME is now narrower than the tool, which is
# left alone deliberately -- renaming it would break every docket citation
# that points at it, and a stale name is cheaper than a dangling reference.
#
#   hidden   *.hidden.f32   475 files, 81 GiB   (migrated 2026-08-14)
#   logits   *.f16        1,074 files, 165 GiB
#
# THE `logits` SET EXCLUDES */computed/*, AND THAT EXCLUSION IS LOAD-BEARING.
# `malign_logits/cache.py:461` declares `LOGIT_WRITE_DIR = "computed"` with
# the note that every other directory is read-only history. So `computed/` is
# where the cache APPENDS -- 70 files under cloud_run_20260801 -- and an
# append through a symlink to an unmounted volume fails where a read merely
# returns less. Everything else in the set is history by the cache's own
# declaration, which is the whole reason this is safe.
SET="${MIGRATE_SET:-hidden}"

SRC="/Users/rj416/github/malign-logits"
DST="/Volumes/diderot/malign-logits"
VOL="/Volumes/diderot"
WORK="${SRC}/.migrate_${SET}"
LIST="${WORK}/files.txt"
STAMP="${WORK}/verified.ok"

case "$SET" in
  hidden) PAT='*.hidden.f32' ;;
  logits) PAT='*.f16' ;;
  *) echo "REFUSING: unknown set '$SET'" >&2; exit 1 ;;
esac

mkdir -p "$WORK"
cd "$SRC"
[ "${MIGRATE_QUIET:-}" ] || echo "SET=$SET  pattern=$PAT  work=$(basename "$WORK")"

need_volume() {
  if [ ! -d "$VOL" ]; then
    echo "REFUSING: $VOL is not mounted." >&2
    echo "  A detached volume makes every reader silently see fewer models," >&2
    echo "  which is the failure mode this whole arrangement has to avoid." >&2
    exit 1
  fi
}

build_list() {
  # Real files only. An already-migrated path is a symlink and must not be
  # re-copied onto itself; -type f excludes them, which is what makes every
  # phase safe to re-run.
  #
  # -size +0c EXCLUDES 25 ZERO-BYTE SIDECARS, deliberately. `scan_hidden()`
  # skips them by its own rule -- `os.path.exists(h) and os.path.getsize(h)`
  # -- so they are not in any reader's population, hold no residual stream,
  # and moving them frees nothing. Leaving them in place also keeps that skip
  # path exercised on real local files rather than on symlinks. Found by
  # testing: the smallest file in the set is 0 bytes, and an earlier version
  # of `verify` required non-empty and would have failed on all 25.
  case "$SET" in
    hidden)
      find data -name '*.hidden.f32' -type f -size +0c | sort > "$LIST" ;;
    logits)
      # -path exclusion for the cache's write target, per the header.
      find data -name '*.f16' -type f -size +0c \
           -not -path '*/computed/*' | sort > "$LIST" ;;
    *)
      echo "REFUSING: unknown set '$SET' (want: hidden | logits)" >&2; exit 1 ;;
  esac
}

case "${1:-}" in

list)
  build_list
  n=$(wc -l < "$LIST" | tr -d ' ')
  echo "$n real $PAT files to move"
  if [ "$n" -gt 0 ]; then
    # du -c with find -exec + prints a fresh total per batch when the argument
    # list splits, and reading the last one undercounts. xargs one call at a
    # time is slower and correct.
    tr '\n' '\0' < "$LIST" | xargs -0 stat -f '%z' \
      | awk '{s+=$1} END {printf "  %.1f GB total\n", s/1073741824}'
    awk -F/ '{d=$1"/"$2; if ($1=="data" && $2=="raw") d=$1"/"$2"/"$3; print d}' "$LIST" \
      | sort | uniq -c | sort -rn | sed 's/^/  /'
  fi
  already=$(find data -name "$PAT" -type l | wc -l | tr -d ' ')
  echo "already symlinked: $already"
  ;;

copy)
  need_volume
  build_list
  mkdir -p "$DST"
  echo "rsync $(wc -l < "$LIST" | tr -d ' ') files -> $DST"
  # --files-from preserves the relative directory structure, so
  # data/raw/twp_fill/x.hidden.f32 lands at $DST/data/raw/twp_fill/x.hidden.f32.
  # NO --remove-source-files: nothing is deleted by the copy, ever. Deletion
  # happens in `link`, after `verify`, and only there.
  rsync -a --info=progress2 --human-readable --partial \
        --files-from="$LIST" "$SRC/" "$DST/"
  rm -f "$STAMP"   # a fresh copy invalidates any previous verification
  echo
  echo "copy done. NOTHING has been deleted. Run: $0 verify"
  ;;

verify)
  need_volume
  build_list
  echo "checksumming $(wc -l < "$LIST" | tr -d ' ') files on both sides..."
  # -c forces a full checksum rather than the size+mtime quick check, and
  # --dry-run means it compares and transfers nothing. Any itemised line is a
  # difference. This is the step that catches a copy which ran while the
  # source was being written.
  out=$(rsync -rc --dry-run --itemize-changes \
        --files-from="$LIST" "$SRC/" "$DST/" || true)
  if [ -n "$out" ]; then
    echo "VERIFY FAILED -- these differ or are missing on diderot:" >&2
    echo "$out" | head -40 >&2
    echo "  ...re-run '$0 copy'. Nothing has been deleted." >&2
    rm -f "$STAMP"
    exit 1
  fi
  # and confirm every destination file is present at EXACTLY the source size.
  # Not `-s` (non-empty): that conflates "absent" with "legitimately small",
  # and a size comparison says the thing actually wanted.
  missing=0
  while IFS= read -r f; do
    if [ ! -e "$DST/$f" ]; then
      echo "MISSING: $DST/$f" >&2; missing=$((missing+1)); continue
    fi
    a=$(stat -f '%z' "$f"); b=$(stat -f '%z' "$DST/$f")
    [ "$a" = "$b" ] || { echo "SIZE $a vs $b: $f" >&2; missing=$((missing+1)); }
  done < "$LIST"
  [ "$missing" -eq 0 ] || { echo "$missing bad destination file(s)" >&2; rm -f "$STAMP"; exit 1; }
  date > "$STAMP"
  echo "VERIFIED: every file byte-identical on diderot. Run: $0 link"
  ;;

link)
  need_volume
  # DEFAULT IS SIZE-ONLY. `link --cmp` adds a full byte comparison per file.
  #
  # RH's call, and the reasoning is worth keeping because the default matters
  # more than the flag. What the size check leaves uncovered is corruption
  # that preserves length, in a file rsync transferred minutes earlier -- and
  # rsync ALWAYS verifies a whole-file checksum as it transfers, so that
  # window is between a verified write and this read. Measured before
  # choosing: 475 of 475 present at exactly the source size.
  #
  # What is NOT skipped, in either mode, because these cost nothing and are
  # where the real failures have been: the destination must EXIST and must
  # match the source SIZE, checked per file immediately before that file is
  # removed. A batch stamp taken earlier is what [6005] showed to be the weak
  # form -- rsync right about what existed when it ran, wrong about what
  # existed when it was read -- so the check stays at the moment of deletion
  # even when it is cheap.
  #
  # Use --cmp when the copy is old, the volume has been remounted, or anything
  # else has written to it since. Then the size check is not enough.
  DEEP=0
  [ "${2:-}" = "--cmp" ] && DEEP=1
  if [ "$DEEP" = "1" ]; then
    echo "MODE: full byte comparison per file (slow, reads everything)"
  else
    echo "MODE: size-only per file (fast). Use '$0 link --cmp' for byte comparison."
  fi
  build_list
  n=0; skipped=0
  while IFS= read -r f; do
    d="$DST/$f"
    if [ ! -e "$d" ]; then
      echo "SKIP (destination missing): $f" >&2; skipped=$((skipped+1)); continue
    fi
    a=$(stat -f '%z' "$f"); b=$(stat -f '%z' "$d")
    if [ "$a" != "$b" ]; then
      echo "SKIP (size $a vs $b): $f" >&2; skipped=$((skipped+1)); continue
    fi
    if [ "$DEEP" = "1" ] && ! cmp -s "$f" "$d"; then
      echo "SKIP (bytes differ at equal size): $f" >&2; skipped=$((skipped+1)); continue
    fi
    rm -f "$f"
    ln -s "$d" "$f"          # absolute, so it resolves from any cwd
    n=$((n+1))
  done < "$LIST"
  echo "linked $n files, skipped $skipped."
  if [ "$DEEP" = "1" ]; then
    echo "Each original was removed only after a full byte comparison."
  else
    echo "Each original was removed only after its copy was confirmed present"
    echo "at the identical size. Equal-length corruption is NOT covered; the"
    echo "originals are gone, and the copies on diderot are now the only ones."
  fi
  echo "Run '$0 status' to confirm, and '$0 restore' to undo."
  ;;

status)
  real=$(find data -name "$PAT" -type f | wc -l | tr -d ' ')
  link=$(find data -name "$PAT" -type l | wc -l | tr -d ' ')
  echo "local real files : $real"
  echo "symlinks         : $link"
  if [ -d "$VOL" ]; then
    dead=0
    while IFS= read -r l; do [ -e "$l" ] || dead=$((dead+1)); done \
      < <(find data -name "$PAT" -type l)
    echo "broken symlinks  : $dead"
    [ "$dead" -eq 0 ] || echo "  (a broken link with the volume MOUNTED means the file is gone from diderot)" >&2
  else
    echo "broken symlinks  : UNKNOWN, $VOL not mounted"
    echo "  With the volume detached every symlink dangles and scan_hidden()"
    echo "  will simply enumerate fewer models WITHOUT erroring."
  fi
  [ -f "$STAMP" ] && echo "last verified    : $(cat "$STAMP")" || echo "last verified    : never"
  ;;

restore)
  need_volume
  n=0
  while IFS= read -r l; do
    d=$(readlink "$l")
    [ -s "$d" ] || { echo "SKIP (target missing): $l" >&2; continue; }
    rm -f "$l"
    cp -p "$d" "$l"
    cmp -s "$l" "$d" || { echo "RESTORE MISMATCH: $l" >&2; exit 1; }
    n=$((n+1))
  done < <(find data -name "$PAT" -type l | sort)
  echo "restored $n files locally. The copies on diderot are left in place;"
  echo "delete them by hand once you are satisfied."
  ;;

*)
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
  ;;
esac
