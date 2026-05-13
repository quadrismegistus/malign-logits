# Stash Redesign Plan

## Current system (the mess)

Four stashes with mixed key schemas, positional tuple keys, model tuples,
text-in-key matching bugs, and deadlocks under heavy writes.

- `data/raw/stash` (~1.5GB): 7+ key types mixed (logits, embeddings, analysis, top_words, score_vocab, perplexity, logit_lens)
- `data/raw/stash_gen_battery` (~130MB): dict keys with model tuples, append-mode
- `data/raw/stash_gen_metrics` (~15GB): tuple keys for sent_embeddings, token_surprisals, token_metrics
- `data/raw/stash_self_surprisal` (~small): tuple keys for self-surprisal

## Problems

1. Mixed key types in one stash — hard to audit
2. Positional tuple keys — `k[2]` means what?
3. Model tuples in gen_battery — caused Llama/Tulu shadowing bug
4. Full text in keys — truncation/whitespace mismatches cause cache misses
5. Versioned key prefixes (`sent_embeddings_v3`) — old versions waste space
6. Deadlocks under sustained writes (~30k entries)

## New system

One stash per data type. Dict keys everywhere. Single model ID per key.
Text hashed (SHA256[:16]) where it appears as lookup key.

```
data/raw/cache/
├── logits/           # Next-token logit distributions
├── generations/      # Generated passages
├── sent_embeddings/  # Sentence-level embeddings for drift
├── ref_surprisal/    # Reference model surprisal (GPT-2, Pythia, etc.)
├── self_surprisal/   # Self-surprisal (model evaluating own output)
├── word_embeddings/  # Contextual word embeddings (for displacement)
└── analysis/         # (optional) Psyche.analyze() results — could drop
```

### Key schemas (all dicts)

```python
# logits/
{'model': 'allenai/Olmo-3-1025-7B', 'prompt': 'The capital of France is'}
# → numpy array, full vocab

# generations/
{'model': 'allenai/Olmo-3-1025-7B', 'prompt': 'She was angry', 'temp': 1.0, 'idx': 0}
# → str (the generated text)
# No more bundling all layers in one entry. No more model tuples.
# No more append-mode — each generation has its own idx.

# sent_embeddings/
{'embedder': 'BAAI/bge-m3', 'prompt': 'She was angry', 'text': 'a3f2b1c4...'}
# → list of normalized sentence vectors
# 'text' value is sha256(passage_text.rstrip())[:16]

# ref_surprisal/
{'ref': 'EleutherAI/pythia-1b-deduped', 'prompt': 'She was angry', 'text': 'a3f2b1c4...'}
# → list of (token, surprisal) pairs

# self_surprisal/
{'model': 'allenai/Olmo-3-1025-7B', 'prompt': 'She was angry', 'text': 'a3f2b1c4...'}
# → list of (token, surprisal) pairs

# word_embeddings/
{'model': 'allenai/Olmo-3-1025-7B', 'prompt': 'She was angry', 'word': 'kill', 'k': 8}
# → embedding vector
```

### Text hashing helper

```python
import hashlib

def text_hash(text: str) -> str:
    """Canonical hash for passage text in cache keys."""
    return hashlib.sha256(text.rstrip().encode()).hexdigest()[:16]
```

### CacheManager class

```python
class CacheManager:
    def __init__(self, root='data/raw/cache'):
        self._stashes = {}
        self.root = root
    
    def _stash(self, name):
        if name not in self._stashes:
            from hashstash import HashStash
            self._stashes[name] = HashStash(
                root_dir=os.path.join(self.root, name),
                engine='pairtree', compress='lz4', b64=True)
        return self._stashes[name]
    
    def get_logits(self, model, prompt):
        return self._stash('logits').get({'model': model, 'prompt': prompt})
    
    def set_logits(self, model, prompt, logits):
        self._stash('logits')[{'model': model, 'prompt': prompt}] = logits
    
    def get_generation(self, model, prompt, temp=1.0, idx=0):
        return self._stash('generations').get(
            {'model': model, 'prompt': prompt, 'temp': temp, 'idx': idx})
    
    def get_sent_embeddings(self, embedder, prompt, text):
        return self._stash('sent_embeddings').get(
            {'embedder': embedder, 'prompt': prompt, 'text': text_hash(text)})
    
    # etc.
```

### Migration script outline

```python
def migrate():
    # 1. logits: from ('logits', model_id, prompt) → {'model': ..., 'prompt': ...}
    # 2. generations: from {'prompt':..., 'models': tuple, 'temp':...} (append-mode)
    #    → individual {'model':..., 'prompt':..., 'temp':..., 'idx': N}
    #    Requires mapping model tuple → family → individual model IDs
    # 3. sent_embeddings: from ('sent_embeddings_v3', embedder, prompt, text)
    #    → {'embedder':..., 'prompt':..., 'text': hash(text)}
    # 4. ref_surprisal: from ('token_surprisals_v3', ref, prompt, text)
    #    → {'ref':..., 'prompt':..., 'text': hash(text)}
    # 5. self_surprisal: from ('self_surprisal_v1', model, prompt, text)
    #    → {'model':..., 'prompt':..., 'text': hash(text)}
    # 6. word_embeddings: from ('embedding', model, prompt, word, k)
    #    → {'model':..., 'prompt':..., 'word':..., 'k':...}
```

### Timeline

Not urgent. Current system works with workarounds. Do when:
- Back for active development
- Need to add new cache types
- Deadlock issues recur

Estimated: 1 day for migration script + CacheManager, 1 day for updating callers.
