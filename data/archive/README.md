# twp_cloud archives

The JSONL transport files are gitignored — 24 MB now and the full-grid re-run
will be several times that, and they are a TRANSPORT format: the canonical
store is HashStash via CacheManager. But a run that cost GPU hours should not
be recoverable only from a directory nobody tracks, so each run is archived
and its CHECKSUM committed here. **The blob is untracked; its identity is not.**

| archive | sha256 | files | records | repo HEAD | taken |
|---|---|---|---|---|---|
| `twp_cloud_run1_20260730-1034.tar.gz` | `df83cba8f2a8f9c0…` | 103 | 13,940 | `d631005` | 2026-07-30 10:35 |

## Run 1 — the pre-grid roster

103/103 models, 13,940 cells. 0 RUN FAILED, 0 MASK FAILED, 221 OOMs absorbed.

**Produced under the OLD boundary rule** — ASCII punctuation only, no CJK
punctuation, no dictionary trie, no script-transition rule. So its Chinese
cells resolve 3–16% of mass against 80–90% for English, and English prompts
with Chinese completions contain glued cross-script units (`mouth什么意思`).
**That is why it is being replaced, and why it is worth keeping**: it is the
only record of what the old rule produced, and any claim that the new rule
changed a result is checkable only against it.

Restore and verify:

```bash
shasum -a 256 data/archive/twp_cloud_run1_20260730-1034.tar.gz   # must match above
tar -xzf data/archive/twp_cloud_run1_20260730-1034.tar.gz -C /tmp
```

Verified on creation: 13,940 records restored, all parse, **0 failing
conservation** — the archive was tested by extracting and re-checking it, not
assumed sound because tar exited 0.
