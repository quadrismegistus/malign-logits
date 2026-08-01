#!/bin/bash
# Keep Q1's coding level with the generator instead of stalling behind it.
#
# lacan's f20x_nonce_code.py resumes correctly on a key but EXITS when it catches
# up. The generator runs ~7h longer, so without this the registered primary's
# coding stops the moment it drains -- which is the failure identified in docket
# [208], where the 2x2 sat uncoded all afternoon while four instrument jobs ran.
#
# Loops the coder unchanged until the generator is gone, then codes once more.
cd "$(dirname "$0")/.." || exit 1
# `[f]20x` brackets the first character so the pattern cannot match the
# process that carries it. An unbracketed `pgrep -f X` inside a WHILE-WAIT
# loop is a latent hang: invoked through `bash -c`, the wrapper's own argv
# contains X, pgrep matches it, and the loop waits forever for a process
# that is the loop. Cost me two false READINGS today on a live run check.
while pgrep -f '[f]20x_nonce_generate' >/dev/null; do
    .venv/bin/python scripts/f20x_nonce_code.py --workers 12 >> logs/f20x_nonce_code.log 2>&1
    sleep 300
done
.venv/bin/python scripts/f20x_nonce_code.py --workers 12 >> logs/f20x_nonce_code.log 2>&1
echo "KEEPUP DONE: generator finished and final coding pass complete"
