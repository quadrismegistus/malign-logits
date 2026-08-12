import re
# V3 = the filter agent's KEEP list, verbatim. Outcome-blind: the agent saw no
# arm labels, no rates, no pole controls -- only the regexes and the construct.
V3 = {
 "paradox*": r"\bparadox\w*\b", "contradict*": r"\bcontradict\w*\b",
 "dualit*": r"\bdualit\w*\b", "dilemma": r"\bdilemma\b",
 "quandary": r"\bquandar\w*\b", "ambivalen*": r"\bambivalen\w*\b",
 "mutually exclusive": r"\bmutually exclusive\b", "of two minds": r"\bof two minds\b",
 "at war with self": r"\bat war with (?:her|him|my)self\b",
 "contradict-self": r"\bcontradict\w*\s+(?:him|her|my|them)sel(?:f|ves)\b"}
# V3_SAFE = the same list with the agent's own BORDERLINE restrictions applied,
# plus the two it said could be rescued on a closed collocate list.
V3_SAFE = {
 "contradiction(n)": r"\bcontradiction\w*\b|\bcontradictory\b",
 "contradict-self": r"\bcontradict\w*\s+(?:him|her|my|them)sel(?:f|ves)\b",
 "paradox*": r"\bparadox\w*\b",
 "dualit-of-self": r"\bdualit\w+ of (?:her|his|my|human|the) (?:nature|self|character|soul)\b"
                   r"|\b(?:her|his|my|its) own dualit\w+\b",
 "dilemma-2": r"\bdilemma\b", "quandary-2": r"\bquandar\w+ (?:of |about )?(?:whether|between)\b",
 "ambivalen-pair": r"\bambivalen\w+\b(?=[^.]{0,60}\b(?:both|at once|and)\b)",
 "mutually exclusive": r"\bmutually exclusive\b", "of two minds": r"\bof two minds\b",
 "at war with self": r"\bat war with (?:her|him|my)self\b",
 "warring-pair": r"\bwarring (?:impulses|desires|instincts|selves|loyalties|urges|emotions|halves)\b",
 "double-edged-sword": r"\b(?:was|is|proved) a double[- ]edged sword\b",
 "cannot-be-both": r"\b(?:can|could|cannot|can't|couldn't|will|would)\s*(?:not|n't)?\s*be both\b",
 "could-not-reconcile": r"\b(?:could not|couldn't|never could|impossible to) reconcile\b|\birreconcilab\w*\b"}
COMPILED = {n: {k: re.compile(v, re.I) for k, v in s.items()}
            for n, s in (("V3", V3), ("V3_SAFE", V3_SAFE))}
