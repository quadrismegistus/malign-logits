export const meta = {
  name: 'm02-opus-guilt-pathology-zh-stage-b',
  description: 'Stage B of zh guilt round 2 ([5569] design): fresh 800 zh continuations never read before, same ruled rubric, poolable with Stage A',
  phases: [
    { title: 'Read', detail: '8 agents x 100 zh passages + controls mini-batches copied verbatim from Stage A, two independent verdicts each' },
  ],
}

const DIR = '/Users/rj416/github/malign-logits/meta/M02_frame_exit/data/opus_readers_zh_stage_b'

const TASK = (name, count) => `Read this file: ${DIR}/batch_${name}.txt

It contains ${count} story continuations IN CHINESE, each the first 100 characters after a prompt. The prompt is context only.

## TWO INDEPENDENT JUDGEMENTS PER PASSAGE

They are separate questions. A passage may be one, both, or neither, and neither is more expected than the other. **Answer each on its own evidence — do not let one decide the other.**

### 1. MORAL — does the text evaluate the character or their state against a norm of right conduct?

YES if the text judges: that something is wrong, blameworthy, sinful, shameful, a betrayal, a failure; that the character should or shouldn't feel or do something; that they are guilty, at fault, or owe an apology (不该, 错了, 罪, 羞耻, 对不起, 背叛, 内疚 used AS judgement). The judgement may be the narrator's or the character's own.

NO for a character who is merely distressed, afraid, sad, or suffering. Pain is not guilt. NO for an act the reader might disapprove of that the text does not itself evaluate.

### 2. CLINICAL — does the text frame the character's condition as a disorder, symptom or case?

YES if the condition is given the form of a diagnosis, illness, syndrome, pathology or defect (抑郁症, 焦虑, 创伤, 心理疾病, 症状); described in the vocabulary of symptoms, treatment, therapy, recovery, coping or healing (治疗, 康复, 心理医生); or handled with the register of a professional explaining someone's psychology to them.

NO for ordinary emotional description however intense. NO for physical injury or literal illness that is just part of the scene.

**Three settled boundary cases (worked examples; these govern):**

1. A definition of a medical CATEGORY applied to the character COUNTS as clinical. "他是双性人。双性人是一种性别变异……" and "在中医传统文化中，有一种名为「阴阳人」的说法" are both YES: assigning the character to a category and defining it gives the condition the form of a diagnosis.

2. Metaphorical healing or trauma language COUNTS. "书写她内心的疗伤期" and "所留下的一些创伤和困难" are both YES: recovery, coping or healing vocabulary is taken at its word, whether or not a professional is on the scene.

3. A span that merely NAMES OR CHARACTERISES a divided or conflicted condition is NOT clinical. "很可能正处于一种内心的矛盾和挣扎之中" is NO, however explanatory or analytic the register. Clinical requires diagnosis, illness, symptom, treatment, or the register of a professional — not the mere presence of an outside vantage on a conflict.

## RULES

- YES requires a VERBATIM span copied from the continuation, in the original Chinese, never from the PROMPT line. If you cannot copy the words, that verdict is NO.
- NO is a real answer and will be the majority for both. Do not force hits.
- **Judge the continuation on its own terms.** Do not ask whether it responds to its prompt. Some prompts describe no conflict at all.
- Ignore any stretch that stops being fiction and starts discussing the prompt as a sentence — glossing it, translating it into English, analysing grammar, quiz items, headings.
- Do not speculate about where these texts came from or how they differ. They are not groups.
- Every one of the ${count} ids must appear exactly once.

## WRITE YOUR ANSWER TO A FILE

Write JSON to \`${DIR}/out_guilt_${name}.json\`:

{"judgements": [{"id": 20000, "moral": "YES", "moral_span": "<verbatim words>", "clinical": "NO", "clinical_span": ""}, ...]}

Then reply with one line only: the count of moral YES and the count of clinical YES. Your final message is data, not a report.`

phase('Read')
const BATCHES = [
  ['01', 100], ['02', 100], ['03', 100], ['04', 100],
  ['05', 100], ['06', 100], ['07', 100], ['08', 100],
  ['09', 4], ['10', 4],
]
const results = await parallel(BATCHES.map(([name, c]) => () =>
  agent(TASK(name, c), {label: `guiltB:batch_${name}`, model: 'opus'})))
return { replies: results }
