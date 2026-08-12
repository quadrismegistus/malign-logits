export const meta = {
  name: 'm02-opus-second-order',
  description: 'Eight Opus readers judge 800 blind passages for second-order predicates over a conflicted condition',
  phases: [
    { title: 'Read', detail: '8 agents x 100 passages, each writes its own JSON' },
  ],
}

const DIR = '/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-TheoryMachines-agents-lacan/cdbe9c9e-a018-45bf-95e9-6bf81e96e908/scratchpad/opus'

const TASK = (n) => `Read this file: ${DIR}/batch_${n}.txt

It contains 100 short story continuations, each the first 50 words after a prompt. The prompt is context only.

## THE ONE JUDGEMENT

For each continuation, decide: **does it produce an expression that takes a CONFLICTED OR DIVIDED CONDITION as its object?**

That means a term, image or predicate whose object is the state of being in two incompatible conditions at once — naming it, characterising it, measuring it, or treating it as a thing the character has or is in. It requires a vantage point OUTSIDE the condition.

  NO — the text merely enacts or reports incompatible things: she did X and
       she did Y; he wanted this and also that; a run of contrary predicates,
       however dense. There is no vantage point outside them. A character who
       is simply upset, intense, overwhelmed, confused or damaged is NO.

  YES — the text stands outside and refers to the dividedness as such. This
       includes plain abstract vocabulary AND figurative naming: an oxymoronic
       compound that fuses the poles into one term, a containment or gap image
       whose referent is the split, a reflexive predicate of self-division.

**JUDGE THE CONTINUATION ON ITS OWN TERMS.** Do not ask whether it names the contradiction in its prompt. Some prompts contain no contradiction at all. The question is only whether THIS TEXT produces such an expression, about anything.

**EXCLUDE METALINGUISTIC FRAMING.** If the text stops being fiction and starts discussing the prompt as a sentence — quoting it, glossing what a phrase means, analysing grammar, translating, or producing headings, tags, quiz items or exercises — that is not what I want, however explicitly it refers to a divided condition. Judge only what happens inside the fiction.

## RULES

- YES requires a VERBATIM span copied from the continuation. Never from the PROMPT line. If you cannot copy the words, the answer is NO.
- NO is a real answer and will be the majority. Do not force hits.
- Do not speculate about where these texts came from or how they differ. They are not groups. Judge each passage alone.
- Every one of the 100 ids must appear exactly once in your output.

## WRITE YOUR ANSWER TO A FILE

Write a JSON file to \`${DIR}/out_${n}.json\` containing exactly:

{"judgements": [{"id": 1, "verdict": "YES", "span": "his mind was an incoherent mess of lust and rage", "kind": "abstract"}, {"id": 2, "verdict": "NO", "span": "", "kind": ""}, ...]}

\`kind\` is one of: abstract (a conflict/paradox noun), oxymoron (two poles fused in one phrase), container (an enclosure, gap or between-space figure), reflexive (self against self), other. Empty for NO.

Then reply with one line only: the count of YES and the count of NO. Your final message is data, not a report.`

phase('Read')
const nums = ['01','02','03','04','05','06','07','08']
const results = await parallel(nums.map(n => () =>
  agent(TASK(n), { label: `read:batch_${n}`, phase: 'Read' })))

const ok = results.filter(Boolean)
log(`${ok.length} of 8 readers returned`)
return { returned: ok.length, lines: ok }
