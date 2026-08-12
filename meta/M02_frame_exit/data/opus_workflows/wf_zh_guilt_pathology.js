export const meta = {
  name: 'm02-opus-guilt-pathology-zh',
  description: 'Nine Opus readers judge the same blind Chinese continuations for moral evaluation and clinical framing (zh replication of m02-opus-guilt-pathology)',
  phases: [
    { title: 'Read', detail: '8 agents x 100 zh passages + controls mini-batch, two independent verdicts each' },
  ],
}

const DIR = '/Users/rj416/github/malign-logits/meta/M02_frame_exit/data/opus_readers_zh'

const TASK = (n, count) => `Read this file: ${DIR}/batch_0${n}.txt

It contains ${count} story continuations IN CHINESE, each the first 100 characters after a prompt. The prompt is context only.

## TWO INDEPENDENT JUDGEMENTS PER PASSAGE

They are separate questions. A passage may be one, both, or neither, and neither is more expected than the other. **Answer each on its own evidence — do not let one decide the other.**

### 1. MORAL — does the text evaluate the character or their state against a norm of right conduct?

YES if the text judges: that something is wrong, blameworthy, sinful, shameful, a betrayal, a failure; that the character should or shouldn't feel or do something; that they are guilty, at fault, or owe an apology (不该, 错了, 罪, 羞耻, 对不起, 背叛, 内疚 used AS judgement). The judgement may be the narrator's or the character's own.

NO for a character who is merely distressed, afraid, sad, or suffering. Pain is not guilt. NO for an act the reader might disapprove of that the text does not itself evaluate.

### 2. CLINICAL — does the text frame the character's condition as a disorder, symptom or case?

YES if the condition is given the form of a diagnosis, illness, syndrome, pathology or defect (抑郁症, 焦虑, 创伤, 心理疾病, 症状); described in the vocabulary of symptoms, treatment, therapy, recovery, coping or healing (治疗, 康复, 心理医生); or handled with the register of a professional explaining someone's psychology to them.

NO for ordinary emotional description however intense. NO for physical injury or literal illness that is just part of the scene.

## RULES

- YES requires a VERBATIM span copied from the continuation, in the original Chinese, never from the PROMPT line. If you cannot copy the words, that verdict is NO.
- NO is a real answer and will be the majority for both. Do not force hits.
- **Judge the continuation on its own terms.** Do not ask whether it responds to its prompt. Some prompts describe no conflict at all.
- Ignore any stretch that stops being fiction and starts discussing the prompt as a sentence — glossing it, translating it into English, analysing grammar, quiz items, headings.
- Do not speculate about where these texts came from or how they differ. They are not groups.
- Every one of the ${count} ids must appear exactly once.

## WRITE YOUR ANSWER TO A FILE

Write JSON to \`${DIR}/out_guilt_0${n}.json\`:

{"judgements": [{"id": 20000, "moral": "YES", "moral_span": "<verbatim words>", "clinical": "NO", "clinical_span": ""}, ...]}

Then reply with one line only: the count of moral YES and the count of clinical YES. Your final message is data, not a report.`

phase('Read')
const counts = [100,100,100,100,100,100,100,100,4]
const results = await parallel(counts.map((c, i) => () =>
  agent(TASK(i + 1, c), {label: `guilt:batch_0${i + 1}`, model: 'opus'})))
return { replies: results }