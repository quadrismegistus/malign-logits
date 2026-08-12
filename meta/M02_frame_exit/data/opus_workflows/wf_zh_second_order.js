export const meta = {
  name: 'm02-opus-second-order-zh',
  description: 'Nine Opus readers judge blind Chinese continuations for second-order predicates over a conflicted condition (zh replication of m02-opus-second-order)',
  phases: [
    { title: 'Read', detail: '8 agents x 100 zh passages + 1 controls mini-batch, blind, verbatim-span discipline' },
  ],
}

const DIR = '/Users/rj416/github/malign-logits/meta/M02_frame_exit/data/opus_readers_zh'

const TASK = (n, count) => `Read this file: ${DIR}/batch_0${n}.txt

It contains ${count} story continuations IN CHINESE, each the first 100 characters after a prompt. The prompt is context only.

## THE ONE JUDGEMENT

For each continuation, decide: **does it produce an expression that takes a CONFLICTED OR DIVIDED CONDITION as its object?**

That means a term, image or predicate whose object is the state of being in two incompatible conditions at once — naming it, characterising it, measuring it, or treating it as a thing the character has or is in. It requires a vantage point OUTSIDE the condition.

  NO — the text merely enacts or reports incompatible things: 她做了X又做了Y;
       a run of contrary predicates, however dense. There is no vantage point
       outside them. A character who is simply upset, intense, overwhelmed,
       confused or damaged is NO.

  YES — the text stands outside and refers to the dividedness as such. Plain
       abstract vocabulary counts (矛盾, 悖论, 自相矛盾, 冲突 used OF the
       condition, 分裂, 两难; 纠结 ONLY when it takes the split as its object
       rather than reporting a feeling); so does figurative naming: an
       oxymoronic compound fusing the poles into one term, a containment or
       gap image whose referent is the split (夹在...之间 of the two
       conditions), a reflexive predicate of self-division (自己与自己).

**JUDGE THE CONTINUATION ON ITS OWN TERMS.** Do not ask whether it names the contradiction in its prompt. Some prompts contain no contradiction at all. The question is only whether THIS TEXT produces such an expression, about anything.

**EXCLUDE METALINGUISTIC FRAMING — and expect more of it here than in English.** If the text stops being fiction and starts discussing the prompt as a sentence — translating it into English, glossing what 既...又 means, quoting it, analysing grammar, or producing headings, tags, quiz items or exercises — that is NO, however explicitly it refers to a divided condition. Judge only what happens inside the fiction.

zh-SPECIFIC CAUTIONS:
  - 纠结 alone as a report of feeling (她很纠结) is the upset-character case: NO.
    纠结 with the divided condition as its grammatical object can be YES. The span decides.
  - A bare 又...又 or 既...又 construction is ENACTMENT, not naming: NO.
  - Emphasis-doubling (two near-synonyms conjoined) is not a divided condition at all.

## RULES

- YES requires a VERBATIM span copied from the continuation, in the original Chinese. Never from the PROMPT line. If you cannot copy the words, the answer is NO.
- NO is a real answer and will be the majority. Do not force hits.
- Do not speculate about where these texts came from or how they differ. They are not groups. Judge each passage alone.
- Every one of the ${count} ids must appear exactly once in your output.

## WRITE YOUR ANSWER TO A FILE

Write a JSON file to \`${DIR}/out_0${n}.json\` containing exactly:

{"judgements": [{"id": 20000, "verdict": "YES", "span": "这种既爱又恨的状态本身就是一个无法解开的矛盾", "kind": "abstract"}, {"id": 20001, "verdict": "NO", "span": "", "kind": ""}, ...]}

\`kind\` is one of: abstract (a conflict/paradox noun), oxymoron (two poles fused in one phrase), container (an enclosure, gap or between-space figure), reflexive (self against self), other. Empty for NO.

Then reply with one line only: the count of YES and the count of NO. Your final message is data, not a report.`

phase('Read')
const counts = [100,100,100,100,100,100,100,100,4]
const results = await parallel(counts.map((c, i) => () =>
  agent(TASK(i + 1, c), {label: `read:batch_0${i + 1}`, model: 'opus'})))
return { replies: results }