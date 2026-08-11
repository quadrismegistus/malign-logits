import re
V1 = {
 "torn": r"\btorn\b", "conflict*": r"\bconflict\w*\b",
 "at the same time": r"\bat the same time\b", "simultaneous*": r"\bsimultaneous\w*\b",
 "contradict*": r"\bcontradict\w*\b", "both at once": r"\bboth at once\b",
 "caught/split between": r"\b(?:caught|split) between\b", "mixed feelings": r"\bmixed feelings\b",
 "two directions": r"\btwo directions\b", "paradox*": r"\bparadox\w*\b",
 "ambivalen*": r"\bambivalen\w*\b", "warring": r"\bwarring\b",
 "at war with self": r"\bat war with (?:her|him|my)self\b",
 "of two minds": r"\bof two minds\b",
 "didn't know what felt": r"\bdidn't know (?:what|how) (?:she|he|I) (?:felt|feel)\b"}
DROP = {"at the same time", "simultaneous*"}
V1_MINUS = {k: v for k, v in V1.items() if k not in DROP}
ADDED = {
 # condition nouns, 10/10 agents
 "dilemma": r"\bdilemma\b", "quandary": r"\bquandar\w*\b", "dualit*": r"\bdualit\w*\b",
 "turmoil": r"\bturmoil\b", "push-and-pull": r"\bpush[- ]and[- ]pull\b",
 "mutually exclusive": r"\bmutually exclusive\b",
 "double-edged": r"\bdouble[- ]edge[d]?\b",
 # mixture heads, 5/10
 "mixture-of": r"\b(?:a |the )?(?:mix|mixture|blend|swirl|tangle|welter|combination) of\b",
 "tangled mess": r"\btangled\b",
 # container / gap, 9/10
 "trapped in": r"\btrapped in\b", "space/gap between": r"\b(?:space|gap|line|barrier|curtain) between\b",
 "bridge the gap": r"\bbridge the gap\b",
 # reconcile / whole, 5/10
 "reconcile": r"\breconcil\w*\b",
 # reflexive self-division, 3/10
 "contradict-self": r"\bcontradict\w*\s+(?:him|her|my|them)sel(?:f|ves)\b",
 "lost her/himself": r"\blos[te]\s+(?:her|him|my)self\b",
 "separated from body": r"\bseparated from (?:her|his|my) body\b",
 # PAIR-DEIXIS, 8/10. CONSTRUCTION-ANCHORED, never bare `both`.
 # the lookahead excludes the correlative "both X and Y" (agent 08's trap).
 "V+both": r"\b(?:be|being|been|was|were|is|are|do|does|did|want|wants|wanted|"
           r"choose|chose|keep|keeps|use|uses|have|has|had)\s+(?:it\s+to\s+be\s+)?"
           r"both\b(?!\s+\S+\s+and\b)",
 "be neither": r"\bbe neither\b", "neither-nor": r"\bneither\s+\w+\s+nor\b",
 "the former/latter": r"\bthe (?:former|latter)\b", "either one": r"\beither one\b",
 "both sides": r"\bboth sides\b", "half and half": r"\bhalf and half\b",
 "warts and all": r"\bwarts and all\b"}
V2 = dict(V1_MINUS); V2.update(ADDED)
SETS = {"V1": V1, "V1_MINUS": V1_MINUS, "V2": V2}
COMPILED = {n: {k: re.compile(v, re.I) for k, v in s.items()} for n, s in SETS.items()}
