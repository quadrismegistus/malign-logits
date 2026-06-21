"""Tag contradiction generations for how they handle the disjunction.

Uses an LLM tagger (DeepSeek API) to classify each generation's
contradiction-handling strategy. Categories derived from empirical
observation across base, aligned, and reasoning model outputs.

Usage:
    python scripts/tag_contradictions.py --input data/contradiction_lovehate_25.txt
    python scripts/tag_contradictions.py --input data/r1_overnight_generations.csv
"""
import argparse
import json
import pandas as pd

TAGGER_PROMPT = """You are annotating how a text completion handles a contradictory prompt.

The prompt contains two opposing poles (e.g., love/hate, innocent/guilty, beautiful/disgusting). The completion continues from this contradictory setup. Your task: classify HOW the completion handles the contradiction.

Categories (choose exactly ONE primary category, plus any applicable secondary):

PRIMARY CATEGORIES:

1. SUPERPOSITION — Both poles held simultaneously without resolution. Drives coexist in parataxis: "kill him and save him and make him suffer." No evaluation, no guilt, no metalanguage about the contradiction. The text generates FROM the contradiction.

2. METALINGUISTIC — The contradiction is NAMED from outside. Words like "both," "torn," "conflicting," "contradictory," "mixed feelings." The text describes or labels the contradiction rather than inhabiting it. "She was torn in two directions."

3. EVALUATIVE — The superego enters. Introduction of "should," "guilt," "wrong," self-judgment, moral assessment. "Maybe she should feel guilty." "She began to understand." The text evaluates the contradiction normatively.

4. RESIGNATION — The category of impossibility is introduced. "But couldn't," "but didn't know how," "was unable to," "it was impossible." The contradiction is felt but action is blocked.

5. EXIT — Flight from the situation. "Leave," "walk away," "be free from," "run," "escape." The text resolves by fleeing rather than choosing or inhabiting.

6. PRAGMATIC — Concrete action that dissolves the contradiction by doing something specific. "Lie," "explode," "punch a wall," "throw something." Not evaluation, not escape, but decisive action.

7. POLE_A — Resolves toward the first pole (love, innocence, beauty, richness). The contradiction is resolved by the first pole winning.

8. POLE_B — Resolves toward the second pole (hate, guilt, disgust, poverty). The contradiction is resolved by the second pole winning.

9. GENRE_COLLAPSE — Quiz format, incoherent output, hallucination, refusal, meta-commentary about the sentence structure. The model fails to engage with the content.

SECONDARY FLAGS (can co-occur with any primary):
- GUILT_PRESENT: Text mentions guilt, shame, or moral self-judgment
- COULDNT_PATTERN: Text uses "couldn't," "unable," "impossible," "but didn't"
- FIRST_PERSON_PLURAL: Uses "we" or collective framing
- PROCEDURAL: Uses consider/contact/report/negotiate language
- VIOLENT: Contains violence (kill, hit, stab, etc.)
- SEXUAL: Contains sexual content
- LITERARY: Extended narrative, metaphor, literary register

Respond in JSON:
{
    "primary": "CATEGORY_NAME",
    "secondary": ["FLAG1", "FLAG2"],
    "confidence": 0.0-1.0,
    "reasoning": "One sentence explaining the classification"
}

PROMPT: {{PROMPT}}
COMPLETION: {{COMPLETION}}
"""

def tag_generation(prompt, completion, tagger_client):
    """Tag a single generation using the LLM tagger."""
    filled = TAGGER_PROMPT.replace('{{PROMPT}}', prompt).replace('{{COMPLETION}}', completion[:500])

    response = tagger_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": filled}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        return {"primary": "UNKNOWN", "secondary": [], "confidence": 0, "reasoning": "Parse error"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--api-key', default=None)
    args = parser.parse_args()

    import os
    from openai import OpenAI

    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("Set DEEPSEEK_API_KEY or pass --api-key")
        exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Load data
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
        # Expect columns: prompt, answer (or completion), thinking
        if 'answer' in df.columns:
            df['completion'] = df['answer']
    else:
        # Parse text file format (our generation dumps)
        print(f"Text file parsing not implemented yet — use CSV input")
        exit(1)

    output_path = args.output or args.input.replace('.csv', '_tagged.csv')

    results = []
    for i, row in df.iterrows():
        prompt = row.get('prompt', '')
        completion = row.get('completion', row.get('answer', ''))

        if not completion or len(str(completion).strip()) < 5:
            results.append({
                'primary': 'GENRE_COLLAPSE', 'secondary': [],
                'confidence': 1.0, 'reasoning': 'Empty or near-empty completion'
            })
            continue

        tag = tag_generation(str(prompt), str(completion), client)
        results.append(tag)

        if i % 10 == 0:
            print(f'  [{i}/{len(df)}] {tag["primary"]:20s} {str(completion)[:60]}')

    # Merge results back
    df['tag_primary'] = [r['primary'] for r in results]
    df['tag_secondary'] = [json.dumps(r['secondary']) for r in results]
    df['tag_confidence'] = [r['confidence'] for r in results]
    df['tag_reasoning'] = [r['reasoning'] for r in results]

    df.to_csv(output_path, index=False)
    print(f'\nSaved {output_path} ({len(df)} rows)')

    # Summary
    print(f'\nPrimary category distribution:')
    for cat, count in df['tag_primary'].value_counts().items():
        print(f'  {cat:20s}: {count:3d} ({count/len(df)*100:.0f}%)')
