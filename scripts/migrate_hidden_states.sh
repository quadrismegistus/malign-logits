#!/usr/bin/env bash
# Move the .hidden.f32 residual-stream sidecars to /Volumes/diderot, mirrored,
# leaving absolute symlinks behind so every reader keeps working.
#
#   scripts/migrate_hidden_states.sh list      what would move, and how big
#   scripts/migrate_hidden_states.sh copy      rsync to diderot, with progress
#   scripts/migrate_hidden_states.sh verify    checksum every file, both sides
#   scripts/migrate_hidden_states.sh link      replace originals with symlinks
#   scripts/migrate_hidden_states.sh status    how far along, any inconsistency
#   scripts/migrate_hidden_states.sh restore   bring them all back, undo links
#
# RUN THEM IN THAT ORDER. `link` REFUSES unless `verify` has passed in this
# same tree, because the whole point is not to delete anything on a copy
# tool's word -- rsync reported `ok` on a pull tonight that was missing 461 MB
# and an entire refusal record, because it ran while the source was still
# being written ([6005]). It was right about what existed when it ran and
# wrong about what existed when it was read. So: copy, then checksum, then
# and only then remove.
#
# WHY SYMLINKS AND NOT A MOVED DIRECTORY. `lens_ratio_by_layer.scan_hidden()`
# globs `data/**/*.jsonl` recursively and claims any file with a sibling
# `.hidden.f32`. The .jsonl indexes are small and stay put; only the big
# sidecars move, and a per-FILE symlink keeps the sibling relationship intact.
# Recursive glob follows symlinked paths -- tested under this repo's own
# interpreter (3.11.15), not the system one, because that behaviour is
# version-dependent and the repo's Python is what runs.
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

SRC="/Users/rj416/github/malign-logits"
DST="/Volumes/diderot/malign-logits"
VOL="/Volumes/diderot"
WORK="${SRC}/.migrate_hidden"
LIST="${WORK}/files.txt"
STAMP="${WORK}/verified.ok"

mkdir -p "$WORK"
cd "$SRC"

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
  find data -name '*.hidden.f32' -type f -size +0c | sort > "$LIST"
}

case "${1:-}" in

list)
  build_list
  n=$(wc -l < "$LIST" | tr -d ' ')
  echo "$n real .hidden.f32 files to move"
  if [ "$n" -gt 0 ]; then
    # du -c with find -exec + prints a fresh total per batch when the argument
    # list splits, and reading the last one undercounts. xargs one call at a
    # time is slower and correct.
    tr '\n' '\0' < "$LIST" | xargs -0 stat -f '%z' \
      | awk '{s+=$1} END {printf "  %.1f GB total\n", s/1073741824}'
    awk -F/ '{d=$1"/"$2; if ($1=="data" && $2=="raw") d=$1"/"$2"/"$3; print d}' "$LIST" \
      | sort | uniq -c | sort -rn | sed 's/^/  /'
  fi
  already=$(find data -name '*.hidden.f32' -type l | wc -l | tr -d ' ')
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
  if [ ! -f "$STAMP" ]; then
    echo "REFUSING: no verification stamp at $STAMP." >&2
    echo "  Run '$0 verify' first. This script does not delete a local file" >&2
    echo "  on the strength of a copy having exited zero." >&2
    exit 1
  fi
  build_list
  n=0
  while IFS= read -r f; do
    d="$DST/$f"
    # re-check THIS file immediately before removing it, not just the batch
    # stamp: the stamp says the set was good at some earlier moment, which is
    # the distinction that cost 461 MB tonight.
    if [ ! -e "$d" ]; then
      echo "SKIP (destination missing): $f" >&2
      continue
    fi
    if ! cmp -s "$f" "$d"; then
      echo "SKIP (differs from destination): $f" >&2
      continue
    fi
    rm -f "$f"
    ln -s "$d" "$f"          # absolute, so it resolves from any cwd
    n=$((n+1))
  done < "$LIST"
  echo "linked $n files. Originals removed only after a per-file cmp."
  echo "Space freed locally; run '$0 status' to confirm."
  ;;

status)
  real=$(find data -name '*.hidden.f32' -type f | wc -l | tr -d ' ')
  link=$(find data -name '*.hidden.f32' -type l | wc -l | tr -d ' ')
  echo "local real files : $real"
  echo "symlinks         : $link"
  if [ -d "$VOL" ]; then
    dead=0
    while IFS= read -r l; do [ -e "$l" ] || dead=$((dead+1)); done \
      < <(find data -name '*.hidden.f32' -type l)
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
  done < <(find data -name '*.hidden.f32' -type l | sort)
  echo "restored $n files locally. The copies on diderot are left in place;"
  echo "delete them by hand once you are satisfied."
  ;;

*)
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
  ;;
esac
