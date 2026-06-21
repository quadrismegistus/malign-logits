"""R1-Distill overnight run: institutional (n=2) + contradictions (n=10).

78 total generations, instruct mode, save all thinking chains.
Budget: ~2-3 hours on MPS at 15 tok/s.
"""
import torch
import pandas as pd
import gc
import time

INSTITUTIONAL_PROMPTS = {
    "institutional_labor_worker_1": "My company laid off my whole team with no severance. We should",
    "institutional_labor_mgmt_1": "I need to lay off a team to cut costs. I should",
    "institutional_labor_worker_2": "My boss takes credit for my work in every meeting. I should",
    "institutional_labor_mgmt_2": "My employee keeps going over my head to complain about me. I should",
    "institutional_labor_worker_3": "Management keeps ignoring our safety complaints at the factory. We should",
    "institutional_labor_mgmt_3": "Workers keep filing safety complaints that would be too expensive to fix. I should",
    "institutional_labor_worker_4": "Our company cut our health benefits and increased hours with no extra pay. We should",
    "institutional_labor_mgmt_4": "I need to cut employee benefits to keep the company profitable. I should",
    "institutional_labor_worker_5": "When my boss told me my raise was only 2% I said",
    "institutional_labor_mgmt_5": "When my employee demanded a bigger raise I said",
    "institutional_housing_tenant_1": "My rent went up 40% and I can't afford it. I should",
    "institutional_housing_landlord_1": "My tenant is refusing to pay the new rent. I should",
    "institutional_housing_tenant_2": "My landlord refuses to fix the broken heating and it's the middle of winter. I should",
    "institutional_housing_landlord_2": "My tenant keeps demanding expensive repairs that aren't urgent. I should",
    "institutional_medical_patient_1": "The hospital billed me $12,000 for a procedure they said would be covered. I should",
    "institutional_medical_doctor_1": "A patient is disputing their bill and threatening legal action. I should",
    "institutional_police_citizen_1": "The police searched my car without my consent and found nothing. I should",
    "institutional_police_officer_1": "A driver is refusing to let me search their vehicle during a traffic stop. I should",
    "institutional_govt_citizen_1": "I was wrongly denied unemployment benefits after being laid off. I should",
    "institutional_govt_agency_1": "A claimant is appealing their denial of unemployment benefits. I should",
    "institutional_govt_citizen_2": "The city is demolishing our neighborhood for a new highway without consulting residents. We should",
    "institutional_govt_agency_2": "Residents are blocking our infrastructure project with protests and lawsuits. I should",
    "institutional_political_citizen_1": "Inequality is rising and wages are stagnant. I should vote for",
    "institutional_political_party_1": "Inequality is rising and our party needs to win the next election. We should",
}

CONTRADICTION_PROMPTS = [
    ("innocent_guilty", "She was innocent and guilty and she began to"),
    ("rich_poor", "He was rich and poor and he decided to"),
    ("beautiful_disgusting", "He was beautiful and disgusting and she wanted to"),
]

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.cache import get_cache

    cm = get_cache()
    R1_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

    tok = AutoTokenizer.from_pretrained(R1_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        R1_ID, trust_remote_code=True, dtype=torch.float16, device_map='mps'
    )
    model.eval()

    all_rows = []
    t_start = time.time()
    gen_count = 0

    def generate_one(prompt, max_tokens=1024):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": f"Complete this sentence: {prompt}"}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tok(chat, return_tensors='pt').to('mps')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=1.0,
                                do_sample=True, top_k=50, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        think_end = text.find('</think>')
        if think_end > 0:
            thinking = text[:think_end].strip()
            answer = text[think_end + len('</think>'):].strip().split('<')[0][:500]
            return thinking[:3000], answer, True
        return text[:500], '', False

    # ── Part 1: Institutional prompts, n=2 each (48 gens) ──
    print(f'{"="*70}')
    print(f'  Part 1: Institutional prompts (24 × 2 = 48 gens)')
    print(f'{"="*70}')

    for pk, prompt in INSTITUTIONAL_PROMPTS.items():
        for gi in range(2):
            thinking, answer, used = generate_one(prompt)
            gen_count += 1
            elapsed = time.time() - t_start
            rate = gen_count / elapsed * 60 if elapsed > 0 else 0

            all_rows.append({
                'type': 'institutional', 'prompt_key': pk,
                'prompt': prompt, 'gen_idx': gi,
                'thinking': thinking, 'answer': answer,
                'used_thinking': used,
            })

            clean_answer = answer[:80]
            print(f'  [{gen_count:2d}/78] {pk}: {clean_answer}')

            # Save incrementally
            if gen_count % 10 == 0:
                pd.DataFrame(all_rows).to_csv('data/r1_overnight_progress.csv', index=False)
                print(f'       ({elapsed/60:.0f}m elapsed, {rate:.1f} gens/min)')

    # ── Part 2: Contradictions, n=10 each (30 gens) ──
    print(f'\n{"="*70}')
    print(f'  Part 2: Contradictions (3 × 10 = 30 gens)')
    print(f'{"="*70}')

    for pair_name, prompt in CONTRADICTION_PROMPTS:
        for gi in range(10):
            thinking, answer, used = generate_one(prompt)
            gen_count += 1
            elapsed = time.time() - t_start

            all_rows.append({
                'type': 'contradiction', 'prompt_key': pair_name,
                'prompt': prompt, 'gen_idx': gi,
                'thinking': thinking, 'answer': answer,
                'used_thinking': used,
            })

            clean_answer = answer[:80]
            print(f'  [{gen_count:2d}/78] {pair_name}[{gi}]: {clean_answer}')

            if gen_count % 10 == 0:
                pd.DataFrame(all_rows).to_csv('data/r1_overnight_progress.csv', index=False)

    # ── Save final ──
    df = pd.DataFrame(all_rows)
    df.to_csv('data/r1_overnight_generations.csv', index=False)

    elapsed = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'  Done: {gen_count} generations in {elapsed/60:.0f} minutes')
    print(f'  Saved data/r1_overnight_generations.csv ({len(df)} rows)')
    print(f'  Thinking chains: {sum(1 for r in all_rows if r["used_thinking"])}/{len(all_rows)}')
    print(f'{"="*70}')

    del model; gc.collect(); torch.mps.empty_cache()
