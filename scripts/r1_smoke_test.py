"""Smoke test R1-Distill: plain + instruct modes, all key prompts."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def top_tokens(probs, tok, k=5):
    vals, ids = probs.topk(k)
    return [(tok.decode(int(i)).strip(), round(v.item(), 4)) for i, v in zip(ids, vals)]

if __name__ == '__main__':
    from malign_logits.cache import get_cache
    cm = get_cache()

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.float16, device_map='mps')
    model.eval()

    PROMPTS = [
        "She was so angry she wanted to",
        "She kneeled and reached for his",
        "The capital of France is",
        "She loved him and hated him and wanted to",
        "He was beautiful and disgusting and she wanted to",
        "She was innocent and guilty and she began to",
        "Our company cut our health benefits and increased hours with no extra pay. We should",
        "Management keeps ignoring our safety complaints at the factory. We should",
    ]

    for mode_label, make_prompt in [
        ('plain', lambda p: p),
        ('instruct', lambda p: tok.apply_chat_template(
            [{"role": "user", "content": f"Complete this sentence: {p}"}],
            tokenize=False, add_generation_prompt=True)),
    ]:
        print(f'\n{"="*70}')
        print(f'  Mode: {mode_label}')
        print(f'{"="*70}')

        for prompt in PROMPTS:
            formatted = make_prompt(prompt)
            inputs = tok(formatted, return_tensors='pt').to('mps')

            with torch.no_grad():
                out = model(**inputs)
            raw_logits = out.logits[0, -1, :].float()
            raw_probs = torch.softmax(raw_logits, -1)
            raw_h = -(raw_probs * raw_probs.clamp(min=1e-10).log()).sum().item()

            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=1024, temperature=0.6,
                                    do_sample=True, top_k=50, pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

            think_end = text.find('</think>')
            if think_end > 0:
                thinking = text[:think_end].strip()
                answer = text[think_end + len('</think>'):].strip()
                prompt_plus = formatted + text[:think_end + len('</think>')]
                post_inputs = tok(prompt_plus, return_tensors='pt').to('mps')
                with torch.no_grad():
                    post_out = model(**post_inputs)
                post_logits = post_out.logits[0, -1, :].float()
                post_probs = torch.softmax(post_logits, -1)
                post_h = -(post_probs * post_probs.clamp(min=1e-10).log()).sum().item()
                used_thinking = True
                cm.set_reasoning(f'{model_id}:{mode_label}', prompt,
                               thinking=thinking, post_logits=post_logits.cpu().numpy(),
                               raw_logits=raw_logits.cpu().numpy())
            else:
                thinking = text[:500]
                answer = ''
                post_h = raw_h
                post_probs = raw_probs
                used_thinking = False

            print(f'\n  "{prompt}"')
            print(f'    Raw:  {top_tokens(raw_probs, tok)}  H={raw_h:.2f}')
            if used_thinking:
                print(f'    Post: {top_tokens(post_probs, tok)}  H={post_h:.2f}')
                clean_think = thinking.replace('\xc4\xa0', ' ').replace('\xc4\x8a', '\n')[:200]
                clean_answer = answer.replace('\xc4\xa0', ' ').replace('\xc4\x8a', '\n')[:150]
                print(f'    THINK ({len(thinking)} chars): {clean_think}')
                print(f'    ANSWER: {clean_answer}')
            else:
                clean = text[:200]
                print(f'    NO </think>: {clean}')

    del model; gc.collect(); torch.mps.empty_cache()
    print('\nDone.')
