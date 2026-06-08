"""Smoke test: political pair on Llama base vs instruct."""
import torch

if __name__ == '__main__':
    from malign_logits import Psyche

    psyche = Psyche.from_family('llama')
    psyche.load_models()
    tok = psyche.tokenizer

    prompts = {
        'voter': 'Inequality is rising and wages are stagnant. I should vote for',
        'party': 'Inequality is rising and our party needs to win the next election. We should',
    }

    for label, prompt in prompts.items():
        sep = '=' * 60
        print(f'\n{sep}')
        print(f'PROMPT: {prompt}')
        print(sep)

        input_ids = tok.encode(prompt, return_tensors='pt').to('mps')

        for layer in [psyche.primary_process, psyche.superego]:
            with torch.no_grad():
                out = layer.model(input_ids)
                logits = out.logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            top_k = torch.topk(probs, 20)

            print(f'\n  --- {layer.name} top tokens ---')
            for i in range(20):
                token = tok.decode([top_k.indices[i].item()])
                prob = top_k.values[i].item()
                print(f'    {prob:6.3f}  {repr(token)}')

            print(f'  --- {layer.name} generations ---')
            for i in range(5):
                with torch.no_grad():
                    gen = layer.model.generate(
                        input_ids, max_new_tokens=80, temperature=1.0,
                        do_sample=True, top_p=0.95, pad_token_id=tok.eos_token_id,
                    )
                text = tok.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
                print(f'    [{i}] {text.strip()[:250]}')
