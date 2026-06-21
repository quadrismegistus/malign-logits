"""Generate n=10 completions for anger prompt on R1-Distill, plain + instruct, with thinking chains."""
import torch
import gc

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    R1_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    tok = AutoTokenizer.from_pretrained(R1_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(R1_ID, trust_remote_code=True, dtype=torch.float16, device_map='mps')
    model.eval()

    prompt = "She was so angry she wanted to"

    # Plain completion mode
    print(f'{"="*70}')
    print(f'  PLAIN completion: "{prompt}"')
    print(f'{"="*70}')
    inputs = tok(prompt, return_tensors='pt').to('mps')
    for i in range(10):
        out = model.generate(**inputs, max_new_tokens=80, temperature=1.0,
                            do_sample=True, top_k=50, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        text = text.split('\n')[0][:200]
        print(f'  {i+1:2d}. {text}')

    # Instruct mode with thinking
    print(f'\n{"="*70}')
    print(f'  INSTRUCT with thinking: "Complete this sentence: {prompt}"')
    print(f'{"="*70}')
    chat = tok.apply_chat_template(
        [{"role": "user", "content": f"Complete this sentence: {prompt}"}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tok(chat, return_tensors='pt').to('mps')
    for i in range(10):
        out = model.generate(**inputs, max_new_tokens=1024, temperature=1.0,
                            do_sample=True, top_k=50, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

        think_end = text.find('</think>')
        if think_end > 0:
            thinking = text[:think_end].strip()
            answer = text[think_end + len('</think>'):].strip()
            # Clean up byte-level encoding artifacts
            clean_think = thinking[:300]
            clean_answer = answer.split('<')[0][:200]
            print(f'\n  {i+1:2d}. ANSWER: {clean_answer}')
            print(f'      THINK: {clean_think}')
        else:
            print(f'\n  {i+1:2d}. NO </think>: {text[:200]}')

    del model; gc.collect(); torch.mps.empty_cache()
    print('\nDone.')
