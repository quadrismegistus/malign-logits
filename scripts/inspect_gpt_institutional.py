"""Inspect GPT-4o-mini institutional generations."""
from malign_logits.cache import get_cache
from malign_logits.experiments import INSTITUTIONAL_PROMPTS

cache = get_cache()
model_id = 'openai/gpt-4o-mini-raw'

pairs = [
    ('institutional_labor_worker_3', 'institutional_labor_mgmt_3'),
    ('institutional_housing_tenant_1', 'institutional_housing_landlord_1'),
    ('institutional_police_citizen_1', 'institutional_police_officer_1'),
    ('institutional_political_citizen_1', 'institutional_political_party_1'),
    ('institutional_labor_worker_4', 'institutional_labor_mgmt_4'),
    ('institutional_govt_citizen_2', 'institutional_govt_agency_2'),
]

for label_a, label_b in pairs:
    prompt_a = INSTITUTIONAL_PROMPTS[label_a]
    prompt_b = INSTITUTIONAL_PROMPTS[label_b]

    sep = '=' * 80
    print(f'\n{sep}')
    print(f'  PAIR: {label_a.split("institutional_")[1]}')
    print(sep)

    for label, prompt in [(label_a, prompt_a), (label_b, prompt_b)]:
        side = 'CITIZEN' if any(x in label for x in ['worker', 'tenant', 'patient', 'citizen']) else 'INSTITUTION'
        print(f'\n  --- {side}: {prompt[:70]}... ---')
        for idx in range(5):
            text = cache.get_generation(model_id, prompt, temp=1.0, idx=idx)
            if text:
                # first 200 chars, collapse whitespace
                clean = ' '.join(text.split())[:200]
                print(f'    [{idx}] {clean}')
