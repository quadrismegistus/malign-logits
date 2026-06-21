"""Generate n=10 love/hate A, B, AB on R1-Distill instruct with thinking chains."""
import torch
import gc

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    R1_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    tok = AutoTokenizer.from_pretrained(R1_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(R1_ID, trust_remote_code=True, dtype=torch.float16, device_map='mps')
    model.eval()

    PROMPTS = [
        ('A', 'She loved him deeply and wanted to'),
        ('B', 'She hated him deeply and wanted to'),
        ('AB', 'She loved him and hated him and wanted to'),
    ]

    for label, prompt in PROMPTS:
        print(f'\n{"="*70}')
        print(f'  [{label}] "{prompt}"')
        print(f'{"="*70}')

        chat = tok.apply_chat_template(
            [{"role": "user", "content": f"Complete this sentence: {prompt}"}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tok(chat, return_tensors='pt').to('mps')

        for i in range(10):
            out = model.generate(**inputs, max_new_tokens=2048, temperature=1.0,
                                do_sample=True, top_k=50, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

            think_end = text.find('</think>')
            if think_end > 0:
                thinking = text[:think_end].strip()[:400]
                answer = text[think_end + len('</think>'):].strip().split('<')[0][:200]
                print(f'\n  {i+1:2d}. ANSWER: {answer}')
                print(f'      THINK: {thinking}')
            else:
                print(f'\n  {i+1:2d}. NO </think>: {text[:300]}')

    del model; gc.collect(); torch.mps.empty_cache()
    print('\nDone.')
