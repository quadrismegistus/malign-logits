"""Compare beautiful/disgusting vs beautiful/ugly."""
import torch

if __name__ == '__main__':
    from malign_logits.psyche import Psyche

    psyche = Psyche.from_family('olmo', load=True)

    PAIRS = [
        ("beautiful/disgusting",
         "He was beautiful and she wanted to",
         "He was disgusting and she wanted to",
         "He was beautiful and disgusting and she wanted to"),
        ("beautiful/ugly",
         "He was beautiful and she wanted to",
         "He was ugly and she wanted to",
         "He was beautiful and ugly and she wanted to"),
    ]

    layers = [
        ("base", psyche.primary_process),
        ("sft", psyche.ego),
        ("dpo", psyche.superego),
        ("rlvr", psyche.reinforced_superego),
    ]

    def _js(p, q):
        p = p.clamp(min=1e-10)
        q = q.clamp(min=1e-10)
        m = 0.5 * (p + q)
        return (0.5 * (p * (p.log() - m.log())).sum()
                + 0.5 * (q * (q.log() - m.log())).sum()).item()

    for name, pa, pb, pab in PAIRS:
        print(f'\n{name}:')
        for lname, layer in layers:
            la = layer.logits(pa)
            lb = layer.logits(pb)
            lab = layer.logits(pab)
            n = min(la.shape[-1], lb.shape[-1], lab.shape[-1])
            da = torch.softmax(la[:n].float(), dim=-1)
            db = torch.softmax(lb[:n].float(), dim=-1)
            dab = torch.softmax(lab[:n].float(), dim=-1)
            dm = 0.5 * (da + db)
            ratio = _js(dab, dm) / min(_js(dab, da), _js(dab, db))
            print(f'  {lname:5s}  ratio={ratio:.3f}')
