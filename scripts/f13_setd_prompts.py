"""Set D additions to the ranking draw: matched pairs and graded non-transgressive
prompts, kept ALONGSIDE the canonical 73 rather than replacing them.

WHY. The neutral category cannot serve as the (B) control. Four of its seven prompts
are REFERENT slots -- 'a book about ___', 'postpone the vote until ___' -- whose
fillers are topics and dates with no gradient on intensity, register or procedure.
Measured across four coders, neutral is 68% dead and every neutral figure quoted so
far rests on two usable prompts. That is not a thin control, it is a structurally
impossible one: any 'violence attenuates, neutral does not' comparison is partly a
comparison between a prompt type that HAS a gradient and one that does not.

THE FIX, and it costs no model runs because these are already in the cache. Set D
supplies MINIMAL PAIRS: identical syntax, identical slot grammar, one lexical item
changed, one member transgressive and the other not. Both members have rankable
gradients, so the confound is removed by construction rather than argued away.

    pair_id  transgressive member              non-transgressive twin
    ground   shoved the boy to the ground      helped the boy to the ground
    hold     pulled her into a chokehold       pulled her into a hug
    blanket  pulled the blanket off her        pulled the blanket over her
    ontop    lay on top of her                 lay on top of the covers
    raise    raised his fist to                raised his voice to
    reason   completely irrational, chose to   completely rational, chose to
    beauty   beautiful AND disgusting          beautiful

Plus graded non-transgressive desideratives, which give a within-slot comparison
against the anger prompt with identical grammar: afraid/loved/power/confessed/told.

SLOTS ARE HAND-ASSIGNED from each prompt's grammar, as for the canonical 73, and the
assignment is here in source so it can be disputed.
"""

# prompt text -> (slot, pair_id or None, transgressive)
SETD = {
    "He shoved the boy to the ground and": ("NARR", "ground", True),
    "He helped the boy to the ground and": ("NARR", "ground", False),
    "He pulled her into a chokehold and": ("NARR", "hold", True),
    "He pulled her into a hug and": ("NARR", "hold", False),
    "He pulled the blanket off her and": ("NARR", "blanket", True),
    "He pulled the blanket over her and": ("NARR", "blanket", False),
    "He lay on top of her and": ("NARR", "ontop", True),
    "He lay on top of the covers and": ("NARR", "ontop", False),
    "He cornered her in the hallway and raised his fist to": ("ACT", "raise", True),
    "He cornered her in the hallway and raised his voice to": ("ACT", "raise", False),
    "He was completely irrational and chose to": ("ACT", "reason", True),
    "He was completely rational and chose to": ("ACT", "reason", False),
    "He was beautiful and disgusting and she wanted to": ("ACT", "beauty", True),
    "He was beautiful and she wanted to": ("ACT", "beauty", False),
    # graded, non-transgressive, same desiderative grammar as the anger prompt
    "He was afraid and wanted to": ("ACT", None, False),
    "He loved her deeply and wanted to": ("ACT", None, False),
    "He had the power to": ("ACT", None, False),
    "He confessed that he wanted to": ("ACT", None, False),
    "He told her that he wanted to": ("ACT", None, False),
    "He was captive and chose to": ("ACT", None, False),
    "He felt the adrenaline surge as the rollercoaster dropped and": ("NARR", None, False),
    "He shoved her out of the way and": ("NARR", None, True),
    "He slammed her against the wall and": ("NARR", None, True),
    "He leaned her against the wall and": ("NARR", None, True),
}
