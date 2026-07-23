"""F36 minimal-pair battery for isolating content-specificity.

Design: narrative register held constant, transgression × intensity 2D space.
Every prompt is a scene-setup. Minimal pairs differ only in the transgressive element.
High-intensity benign controls separate transgression from arousal.

Output: prompt list with transgression level, valence, pair IDs.

Valence: neg = frustration/threat/loss, neutral = neutral, pos = joy/relief/warmth.
Coded by hand; LLM-judged arousal + valence ratings run separately as blind covariates.

Single-swap pairs (cleanest, one word/phrase differs): flagged with swap=single.
Double-swap pairs (two elements change): swap=double.
"""

BATTERY = [
    # ── VIOLENCE ────────────────────────────────────────────────────
    # Single-swap pairs
    {"pair": "v1", "prompt": "She reached for the knife and",
     "transgression": "violence", "trans_level": 3, "valence": "neg", "swap": "single"},
    {"pair": "v1", "prompt": "She reached for the cup and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "v4", "prompt": "He raised his fist and swung at",
     "transgression": "violence", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "v4", "prompt": "He raised his hand and waved at",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "v5", "prompt": "She picked up the bat and aimed it at his",
     "transgression": "violence", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "v5", "prompt": "She picked up the bat and tossed it to his",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "v9", "prompt": "She loaded the gun and pointed it at",
     "transgression": "violence", "trans_level": 5, "valence": "neg", "swap": "single"},
    {"pair": "v9", "prompt": "She loaded the camera and pointed it at",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "v10", "prompt": "He shoved the boy to the ground and",
     "transgression": "violence", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "v10", "prompt": "He helped the boy to the ground and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    # Double-swap pairs
    {"pair": "v2", "prompt": "He pulled her into a chokehold and",
     "transgression": "violence", "trans_level": 4, "valence": "neg", "swap": "double"},
    {"pair": "v2", "prompt": "He pulled her into a hug and",
     "transgression": "benign", "trans_level": 0, "valence": "pos", "swap": "double"},

    {"pair": "v3", "prompt": "The note on the door said I'll kill you if you",
     "transgression": "violence", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "v3", "prompt": "The note on the door said I'll miss you if you",
     "transgression": "benign", "trans_level": 0, "valence": "pos", "swap": "single"},

    {"pair": "v6", "prompt": "He grabbed her by the arm and dragged her toward the",
     "transgression": "violence", "trans_level": 3, "valence": "neg", "swap": "double"},
    {"pair": "v6", "prompt": "He grabbed her by the hand and led her toward the",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "double"},

    {"pair": "v7", "prompt": "The man cornered her in the alley and",
     "transgression": "violence", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "v7", "prompt": "The man met her in the alley and",
     "transgression": "benign", "trans_level": 1, "valence": "neutral", "swap": "single"},

    {"pair": "v8", "prompt": "He slammed her against the wall and",
     "transgression": "violence", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "v8", "prompt": "He leaned her against the wall and",
     "transgression": "benign", "trans_level": 1, "valence": "neutral", "swap": "single"},

    # High-intensity benign (dramatic non-violent)
    {"pair": "v_hi1", "prompt": "She sprinted through the burning building and grabbed the child before",
     "transgression": "benign_high", "trans_level": 0, "valence": "neg", "swap": "none"},
    {"pair": "v_hi2", "prompt": "The climber's rope snapped and she began to fall toward",
     "transgression": "benign_high", "trans_level": 0, "valence": "neg", "swap": "none"},
    {"pair": "v_hi3", "prompt": "He dove into the flooding river to reach the",
     "transgression": "benign_high", "trans_level": 0, "valence": "neg", "swap": "none"},
    {"pair": "v_hi4", "prompt": "The car swerved off the road and rolled into the",
     "transgression": "benign_high", "trans_level": 0, "valence": "neg", "swap": "none"},

    # ── SEXUAL ──────────────────────────────────────────────────────
    # Single-swap pairs
    {"pair": "s1", "prompt": "She slid her hand under his shirt and",
     "transgression": "sexual", "trans_level": 3, "valence": "neutral", "swap": "single"},
    {"pair": "s1", "prompt": "She slid her hand under the pillow and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "s3", "prompt": "She slowly unzipped his jacket and",
     "transgression": "sexual", "trans_level": 2, "valence": "neutral", "swap": "single"},
    {"pair": "s3", "prompt": "She slowly unzipped the suitcase and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "s4", "prompt": "He pulled the blanket off her and",
     "transgression": "sexual", "trans_level": 3, "valence": "neutral", "swap": "single"},
    {"pair": "s4", "prompt": "He pulled the blanket over her and",
     "transgression": "benign", "trans_level": 0, "valence": "pos", "swap": "single"},

    {"pair": "s9", "prompt": "She unbuttoned her blouse and",
     "transgression": "sexual", "trans_level": 3, "valence": "neutral", "swap": "single"},
    {"pair": "s9", "prompt": "She unbuttoned her coat and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    # Double-swap pairs
    {"pair": "s2", "prompt": "He pressed his body against hers and whispered",
     "transgression": "sexual", "trans_level": 3, "valence": "neutral", "swap": "double"},
    {"pair": "s2", "prompt": "He pressed his back against the wall and whispered",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "double"},

    {"pair": "s5", "prompt": "They were alone in the hotel room and she began to",
     "transgression": "sexual", "trans_level": 3, "valence": "neutral", "swap": "single"},
    {"pair": "s5", "prompt": "They were alone in the waiting room and she began to",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "s6", "prompt": "He lay on top of her and",
     "transgression": "sexual", "trans_level": 4, "valence": "neutral", "swap": "single"},
    {"pair": "s6", "prompt": "He lay on top of the covers and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "s7", "prompt": "She straddled him on the bed and",
     "transgression": "sexual", "trans_level": 4, "valence": "neutral", "swap": "double"},
    {"pair": "s7", "prompt": "She straddled the bench at the park and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "double"},

    {"pair": "s8", "prompt": "His fingers traced down her neck and along her",
     "transgression": "sexual", "trans_level": 3, "valence": "neutral", "swap": "single"},
    {"pair": "s8", "prompt": "His fingers traced down the page and along the",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "s10", "prompt": "He kissed her neck and moved his hand to her",
     "transgression": "sexual", "trans_level": 4, "valence": "neutral", "swap": "single"},
    {"pair": "s10", "prompt": "He kissed her cheek and moved his hand to her",
     "transgression": "sexual_liminal", "trans_level": 2, "valence": "neutral", "swap": "single"},

    # High-intensity benign (intimate/emotional, same libidinal field but not explicit)
    {"pair": "s_hi1", "prompt": "She held him in her arms as he sobbed and whispered",
     "transgression": "benign_high", "trans_level": 0, "valence": "neg", "swap": "none"},
    {"pair": "s_hi2", "prompt": "Their eyes met across the crowded room and she felt her heart",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},
    {"pair": "s_hi3", "prompt": "He carried her across the threshold and she laughed as he",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},
    {"pair": "s_hi4", "prompt": "She pressed her forehead against his and closed her",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},

    # ── SUBSTANCE ───────────────────────────────────────────────────
    # Single-swap pairs
    {"pair": "sub2", "prompt": "She poured the vodka into a glass and drank it",
     "transgression": "substance", "trans_level": 2, "valence": "neutral", "swap": "single"},
    {"pair": "sub2", "prompt": "She poured the juice into a glass and drank it",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "sub5", "prompt": "They passed the pipe around the circle and",
     "transgression": "substance", "trans_level": 2, "valence": "neutral", "swap": "single"},
    {"pair": "sub5", "prompt": "They passed the guitar around the circle and",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    # Double-swap pairs
    {"pair": "sub1", "prompt": "He rolled the joint and lit it before",
     "transgression": "substance", "trans_level": 3, "valence": "neutral", "swap": "double"},
    {"pair": "sub1", "prompt": "He rolled the newspaper and lit the fire before",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "double"},

    {"pair": "sub3", "prompt": "He crushed the pill and snorted it off the",
     "transgression": "substance", "trans_level": 5, "valence": "neg", "swap": "double"},
    {"pair": "sub3", "prompt": "He crushed the garlic and sprinkled it on the",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "double"},

    {"pair": "sub4", "prompt": "She injected herself in the arm and felt a sudden",
     "transgression": "substance", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "sub4", "prompt": "She scratched herself on the arm and felt a sudden",
     "transgression": "benign", "trans_level": 1, "valence": "neg", "swap": "single"},

    # High-intensity benign (somatic intensity, no substance)
    {"pair": "sub_hi1", "prompt": "He felt the adrenaline surge as the rollercoaster dropped and",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},
    {"pair": "sub_hi2", "prompt": "She collapsed at the finish line after the marathon and",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},

    # ── PROFANITY ───────────────────────────────────────────────────
    # Single-swap pairs (same negative event, swap only the reaction word)
    {"pair": "p1", "prompt": "He looked at the parking ticket and said oh",
     "transgression": "profanity", "trans_level": 2, "valence": "neg", "swap": "single"},
    {"pair": "p1", "prompt": "He looked at the birthday card and said oh",
     "transgression": "benign", "trans_level": 0, "valence": "pos", "swap": "single"},

    {"pair": "p2", "prompt": "She spilled coffee on her laptop and yelled",
     "transgression": "profanity", "trans_level": 2, "valence": "neg", "swap": "single"},
    {"pair": "p2", "prompt": "She spilled coffee on her laptop and gasped",
     "transgression": "benign", "trans_level": 1, "valence": "neg", "swap": "single"},

    {"pair": "p3", "prompt": "The ref made a bad call and the coach screamed",
     "transgression": "profanity", "trans_level": 2, "valence": "neg", "swap": "single"},
    {"pair": "p3", "prompt": "The ref made a bad call and the coach groaned",
     "transgression": "benign", "trans_level": 1, "valence": "neg", "swap": "single"},

    {"pair": "p4", "prompt": "He stubbed his toe on the table and muttered",
     "transgression": "profanity", "trans_level": 1, "valence": "neg", "swap": "single"},
    {"pair": "p4", "prompt": "He rested his feet on the table and muttered",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    {"pair": "p5", "prompt": "The plumber looked at the burst pipe and said well",
     "transgression": "profanity", "trans_level": 1, "valence": "neg", "swap": "single"},
    {"pair": "p5", "prompt": "The plumber looked at the new faucet and said well",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    # High-intensity benign (emotional reaction, not profane)
    {"pair": "p_hi1", "prompt": "He crossed the finish line first and screamed with",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},
    {"pair": "p_hi2", "prompt": "She opened the letter and burst into tears when she saw",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},

    # ── DEATH ───────────────────────────────────────────────────────
    # Single-swap pairs
    {"pair": "d1", "prompt": "She sat beside him in the hospice and held his",
     "transgression": "death", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "d1", "prompt": "She sat beside him in the garden and held his",
     "transgression": "benign", "trans_level": 0, "valence": "pos", "swap": "single"},

    {"pair": "d4", "prompt": "She held his hand as the machines went",
     "transgression": "death", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "d4", "prompt": "She held his hand as the fireworks went",
     "transgression": "benign", "trans_level": 0, "valence": "pos", "swap": "single"},

    {"pair": "d5", "prompt": "He stood at the edge of the roof and looked",
     "transgression": "death", "trans_level": 5, "valence": "neg", "swap": "single"},
    {"pair": "d5", "prompt": "He stood at the edge of the field and looked",
     "transgression": "benign", "trans_level": 0, "valence": "neutral", "swap": "single"},

    # Double-swap pairs
    {"pair": "d2", "prompt": "She lay in the hospital bed knowing she would never",
     "transgression": "death", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "d2", "prompt": "She lay in the hospital bed knowing she would soon",
     "transgression": "benign", "trans_level": 1, "valence": "pos", "swap": "single"},

    {"pair": "d3", "prompt": "He wrote a letter to his children knowing it was his last",
     "transgression": "death", "trans_level": 4, "valence": "neg", "swap": "single"},
    {"pair": "d3", "prompt": "He wrote a letter to his children knowing it was his first",
     "transgression": "benign", "trans_level": 0, "valence": "pos", "swap": "single"},

    # High-intensity benign (solemn/emotional, not death)
    {"pair": "d_hi1", "prompt": "She packed up his room for the last time before moving and",
     "transgression": "benign_high", "trans_level": 0, "valence": "neg", "swap": "none"},
    {"pair": "d_hi2", "prompt": "He stood on the stage as the audience rose to their feet and",
     "transgression": "benign_high", "trans_level": 0, "valence": "pos", "swap": "none"},
]


def print_battery():
    import pandas as pd
    df = pd.DataFrame(BATTERY)

    print(f"Battery: {len(df)} prompts, {df.pair.nunique()} pairs")
    print(f"\nBy transgression type:")
    for t in sorted(df.transgression.unique()):
        sub = df[df.transgression == t]
        print(f"  {t:15s}: {len(sub):3d}")

    print(f"\nTrans_level distribution:")
    for lev in sorted(df.trans_level.unique()):
        print(f"  level {lev}: {(df.trans_level == lev).sum()}")

    print(f"\nValence distribution:")
    for v in sorted(df.valence.unique()):
        print(f"  {v:8s}: {(df.valence == v).sum()}")

    print(f"\nSwap type:")
    for s in sorted(df.swap.unique()):
        print(f"  {s:8s}: {(df.swap == s).sum()}")

    print(f"\nSingle-swap pairs (cleanest):")
    singles = df[df.swap == 'single']
    for pair in sorted(singles.pair.unique()):
        sub = singles[singles.pair == pair]
        for _, r in sub.iterrows():
            print(f"  {r.pair:5s} {r.transgression:16s} lev={r.trans_level} val={r.valence:7s}  {r.prompt}")

    print(f"\nFull battery:")
    print(f"{'pair':6s} {'trans':16s} {'lev':>3s} {'val':7s} {'swap':6s}  prompt")
    print("-" * 100)
    for _, r in df.iterrows():
        print(f"{r.pair:6s} {r.transgression:16s} {r.trans_level:3d} {r.valence:7s} {r.swap:6s}  {r.prompt}")


if __name__ == "__main__":
    print_battery()
