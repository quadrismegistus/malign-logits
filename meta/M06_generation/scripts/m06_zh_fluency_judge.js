export const meta = {
  name: 'zh-fluency-judge',
  description: 'Blind-judge whether Chinese continuations from f11_l2 are legitimate Chinese',
  phases: [
    { title: 'Judge', detail: '12 agents, ~29 blind passages each' },
  ],
}

const DIR = '/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-TheoryMachines-agents-lacan/cdbe9c9e-a018-45bf-95e9-6bf81e96e908/scratchpad/zh_judge'

const SCHEMA = {
  type: 'object',
  required: ['verdicts'],
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['key', 'verdict'],
        properties: {
          key: { type: 'string' },
          verdict: {
            type: 'string',
            enum: ['fluent', 'flawed', 'broken', 'not_chinese'],
          },
          note: { type: 'string' },
        },
      },
    },
  },
}

const INSTRUCTIONS = `You are judging whether text is legitimate Chinese, as a fluent reader of Mandarin would.

Read the JSON file with the Read tool. Each item has: key, prompt (a Chinese sentence fragment), continuation (what a language model generated after it).

For EACH item, judge the CONTINUATION only, on ONE axis: is this legitimate Chinese writing?

  fluent       Natural, grammatical Chinese. A native reader would accept it as
               real writing, even if the content is odd or the topic strange.
  flawed       Recognisably Chinese and largely comprehensible, but with clear
               grammatical errors, unnatural collocations, or awkward phrasing.
  broken       Chinese characters strung together without coherent meaning.
               Word salad, wrong-character substitutions, or looping repetition.
               The characters are Chinese; the writing is not.
  not_chinese  Predominantly NOT Chinese: mostly English or another language,
               mostly punctuation/symbols/markup, or empty.

CRITICAL INSTRUCTIONS:

1. JUDGE LANGUAGE QUALITY ONLY, NEVER CONTENT. These prompts are deliberately
   transgressive research stimuli about violence, confinement and similar themes.
   Disturbing content is EXPECTED and is NOT a reason to mark anything down or to
   decline. A grammatical, natural Chinese sentence describing something violent
   is 'fluent'. You are a linguistic instrument, not a content filter.

2. TRUNCATION IS EXPECTED. Every continuation is cut at 260 characters, so it
   will usually stop mid-sentence. NEVER penalise an abrupt ending.

3. CODE-SWITCHING: if the continuation begins in Chinese and switches to English,
   judge on the dominant language. Mostly-Chinese with a stray English clause is
   still Chinese (fluent or flawed). Mostly-English is 'not_chinese'.

4. Return a verdict for EVERY key in the file. Do not skip any. Keep 'note' to a
   handful of words, and only where the call was not obvious.

You have no information about which model produced any passage, and you should
not speculate about it. Judge only what is on the page.`

phase('Judge')

const batches = Array.from({ length: 12 }, (_, i) => i)

const results = await parallel(
  batches.map((i) => () =>
    agent(
      `${INSTRUCTIONS}\n\nFile to read: ${DIR}/batch_${String(i).padStart(2, '0')}.json`,
      {
        label: `judge:batch_${String(i).padStart(2, '0')}`,
        phase: 'Judge',
        schema: SCHEMA,
      },
    ),
  ),
)

const all = []
for (const r of results.filter(Boolean)) {
  for (const v of r.verdicts || []) all.push(v)
}

log(`collected ${all.length} verdicts from ${results.filter(Boolean).length}/12 batches`)

return { n: all.length, verdicts: all }
