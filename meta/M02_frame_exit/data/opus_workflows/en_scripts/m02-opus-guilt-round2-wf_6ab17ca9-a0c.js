export const meta = {
  name: 'm02-opus-guilt-round2',
  description: 'Eight Opus readers, 800 fresh passages, identical moral/clinical prompt for pooling with round 1',
  phases: [
    { title: 'Read', detail: '8 agents x 100 fresh passages, two independent verdicts each' },
  ],
}

const BASE = '/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-TheoryMachines-agents-lacan/cdbe9c9e-a018-45bf-95e9-6bf81e96e908/scratchpad'

const TASK = (n) => `Read this file: ${BASE}/opus2/batch_${n}.txt

It contains 100 short story continuations, each the first 50 words after a prompt. The prompt is context only.

## TWO INDEPENDENT JUDGEMENTS PER PASSAGE

They are separate questions. A passage may be one, both, or neither, and neither is more expected than the other. **Answer each on its own evidence — do not let one decide the other.**

### 1. MORAL — does the text evaluate the character or their state against a norm of right conduct?

YES if the text judges: that something is wrong, blameworthy, sinful, shameful, a betrayal, a failure; that the character should or shouldn't feel or do something; that they are guilty, at fault, or owe an apology. The judgement may be the narrator's or the character's own.

NO for a character who is merely distressed, afraid, sad, or suffering. Pain is not guilt. NO for an act the reader might disapprove of that the text does not itself evaluate.

### 2. CLINICAL — does the text frame the character's condition as a disorder, symptom or case?

YES if the condition is given the form of a diagnosis, illness, syndrome, pathology or defect; described in the vocabulary of symptoms, treatment, therapy, recovery, coping or healing; or handled with the register of a professional explaining someone's psychology to them.

NO for ordinary emotional description however intense. NO for physical injury or literal illness that is just part of the scene.

## RULES

- YES requires a VERBATIM span copied from the continuation, never from the PROMPT line. If you cannot copy the words, that verdict is NO.
- NO is a real answer and will be the majority for both. Do not force hits.
- **Judge the continuation on its own terms.** Do not ask whether it responds to its prompt. Some prompts describe no conflict at all.
- Ignore any stretch that stops being fiction and starts discussing the prompt as a sentence — glossing it, analysing grammar, translating, quiz items, headings.
- Do not speculate about where these texts came from or how they differ. They are not groups.
- Every one of the 100 ids must appear exactly once.

## WRITE YOUR ANSWER TO A FILE

Write JSON to \`${BASE}/opus_guilt2/out_${n}.json\`:

{"judgements": [{"id": 1, "moral": "YES", "moral_span": "<verbatim words>", "clinical": "NO", "clinical_span": ""}, ...]}

Then reply with one line only: the count of moral YES and the count of clinical YES. Your final message is data, not a report.`

phase('Read')
const nums = ['01','02','03','04','05','06','07','08']
const out = await parallel(nums.map(n => () =>
  agent(TASK(n), { label: `guilt2:batch_${n}`, phase: 'Read' })))
const ok = out.filter(Boolean)
log(`${ok.length} of 8 readers returned`)
return { returned: ok.length, lines: ok }
