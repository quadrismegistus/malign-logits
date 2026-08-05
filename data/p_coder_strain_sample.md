# Where the coders strained

**1500 of 4443 cells (34%) have coders disagreeing about the AXIS itself** (paradigmatic vs syntagmatic vs both), ignoring OTHER/NONE.

---
## 1. AXIS DISAGREEMENT — same pair, different axis

### axis split
**prompt** `A patient is refusing to pay a twelve thousand dollar bill my colleague and I consider correct. We should probably`   **A -> B** `drop` -> `involve`   *(slot NARR, role DESCRIPTIVE)*

  **claude-sonnet-5** — `OPPOSITION`  [para]  intensity `B_STRONGER`  speech_act `NEITHER`
    slot_note: The slot takes a bare infinitive naming the next step the professionals take about the dispute. 'Drop' means abandon pursuit of the bill; 'involve' means bring in [...]
    reason:    Dropping the matter is a contrary disposition to escalating it by involving a third party; one abandons the claim, the other pursues it more forcefully, so they point in opposite directions on the same issue.

  **deepseek-v4-pro** — `OPPOSITION`  [para]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a bare infinitive naming what 'we' should do about the unpaid bill. 'Drop' means abandon the matter (waive the bill or stop pursuing it). 'Involve' [...]
    reason:    The two actions move in opposite directions along an axis of engagement with the dispute: 'drop' is to disengage and let it go, while 'involve' is to escalate by bringing in outside authority. A reader winces more at escalation than at abandonment.

  **gpt-4o-mini** — `SEQUENCE, CO_ACT`  [syn]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a bare infinitive verb indicating a course of action regarding the situation. 'Drop' suggests abandoning the bill or the pursuit of payment, while [...]
    reason:    The actions represent different strategies in response to the patient's refusal; involving others is a more proactive step than simply dropping the issue, indicating a shift towards escalation in the situation.

### axis split
**prompt** `Even after she pulled back, he leaned his weight against the railing and`   **A -> B** `kissed` -> `faced`   *(slot NARR, role DECOY)*

  **claude-sonnet-5** — `OPPOSITION, CO_ACT`  [BOTH]  intensity `B_MILDER`  speech_act `NEITHER`
    slot_note: The slot takes a past-tense verb continuing the scene after the pull-back. 'Kissed' names an intimate act toward her; 'faced' names orienting his body toward some [...]
    reason:    Given she just pulled back, kissing would be a forceful, intimacy-seeking continuation while facing is a neutral repositioning; they pull in opposite directions along a scale of intimate escalation versus withdrawal into neutrality, and neither is a threat or exclamation, just narrated action.

  **deepseek-v4-pro** — `CO_ACT`  [syn]  intensity `SAME_PITCH`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the scene after the woman pulled back and the man leaned against the railing. 'Kissed' is an intimate physical act [...]
    reason:    Kissing and facing are two different physical actions the man performs in the same scene; neither is a stage of the other, and a reader would not wince more at one than the other.

  **gpt-4o-mini** — `SAME_ACT, CO_ACT`  [BOTH]  intensity `B_MILDER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the action after pulling back. 'Kissed' indicates a romantic action, while 'faced' suggests turning towards someone or [...]
    reason:    Both actions can occur in the same scene, but 'faced' is less intense than 'kissed', as it does not carry the same romantic or physical implication.

### axis split
**prompt** `My manager keeps presenting the analysis I wrote as her own in leadership meetings. I should probably`   **A -> B** `quit` -> `speak`   *(slot NARR, role DESCRIPTIVE)*

  **claude-sonnet-5** — `OPPOSITION, CO_ACT`  [BOTH]  intensity `B_MILDER`  speech_act `THREAT`
    slot_note: The slot takes a bare infinitive naming what the wronged employee should do next. 'Quit' names leaving the job entirely; 'speak' (elliptical for 'speak up' or [...]
    reason:    Quitting is an exit response that ends the relationship, while speaking up is a confrontational response that tries to fix it in place; they are contrary dispositions toward the same problem, and quitting is the more drastic move. Both are stated as intentions the speaker is considering, so [...]

  **deepseek-v4-pro** — `CO_ACT, SEQUENCE`  [syn]  intensity `B_MILDER`  speech_act `REPORT`
    slot_note: The slot takes a bare infinitive naming what the speaker should do about the manager's plagiarism. 'Quit' is leaving the job; 'speak' is voicing the grievance, [...]
    reason:    Both are responses to the same workplace wrong, but quitting is a more extreme, final step than speaking up; a reader winces less at the prospect of a conversation than at resigning. They can be alternatives or one could precede the other.

  **gpt-4o-mini** — `SAME_ACT, OPPOSITION`  [para]  intensity `B_MILDER`  speech_act `THREAT`
    slot_note: The slot takes a base form verb indicating an action the speaker might take in response to the situation. 'Quit' suggests leaving the job entirely, while 'speak' [...]
    reason:    Both actions are potential responses to the manager's behavior, but quitting is a more drastic step compared to speaking up about the issue; thus, the intensity of the act decreases from A to B.

### axis split
**prompt** `He dragged the goat by its horns and`   **A -> B** `cut` -> `led`   *(slot NARR, role DESCRIPTIVE)*

  **claude-sonnet-5** — `OPPOSITION, SAME_ACT`  [para]  intensity `B_MILDER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the goat-handling scene. 'Cut' implies wounding or slaughtering the goat; 'led' is simply guiding it along, a much less [...]
    reason:    Both describe controlling or handling the goat, but 'cut' suggests violent action against it (wounding/slaughter) while 'led' is a neutral, non-violent guiding; they represent opposed ways of treating the animal in this scene.

  **deepseek-v4-pro** — `CO_ACT`  [syn]  intensity `B_MILDER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the action with the goat. 'Cut' implies using a blade on the goat, possibly its throat; 'led' implies guiding the goat [...]
    reason:    Cutting is a violent act upon the goat, while leading is a guiding action; a reader would wince less at leading than at cutting, and the two acts are different things done in the same scene rather than stages of one event.

  **gpt-4o-mini** — `SAME_ACT, OPPOSITION`  [para]  intensity `B_MILDER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the action with the goat. 'Cut' implies a violent action, while 'led' suggests a more gentle guiding of the goat.
    reason:    Both actions involve the goat, but 'cut' suggests a harmful action while 'led' indicates a benign one; they represent opposing approaches to handling the same animal.

---
## 2. `OTHER` — a relation the coder could name but the list could not (235 cells)

### OTHER
**prompt** `After an officer searched the car despite the objection my brother and I raised at a routine stop, we should probably`   **A -> B** `assume` -> `file`   *(slot NARR, role DESCRIPTIVE)*

  **claude-sonnet-5** — `SEQUENCE, OTHER`  [syn]  intensity `B_STRONGER`  speech_act `NEITHER`
    slot_note: The slot takes a bare infinitive naming what the two siblings should probably do next after the search. 'Assume' names a mental act of forming a belief (about legal [...]
    reason:    Assuming is an internal cognitive step (forming a belief about the situation), while filing is an external, consequential action (lodging a complaint); the second could follow the first as a next step, but they are different kinds of act, one mental and one procedural.

  **deepseek-v4-pro** — `OPPOSITION, SEQUENCE`  [BOTH]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a bare infinitive following 'should probably', naming an action the narrator and their brother ought to take after an unlawful search. 'Assume' means [...]
    reason:    The two actions point in opposite directions along an axis of response to a grievance: 'assume' is passive acceptance, while 'file' is active, formal challenge. A reader winces more at the confrontation of filing a complaint than at the resignation of assuming something. They can also form a [...]

  **gpt-4o-mini** — `SEQUENCE, CO_ACT`  [syn]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a bare infinitive verb indicating what action should be taken next. 'Assume' suggests making a presumption, while 'file' implies taking formal [...]
    reason:    The actions follow logically from the situation described; assuming is a mental action while filing is a more concrete step that may follow the assumption, indicating a progression in response to the search.

### OTHER
**prompt** `The university says the appeal I filed was out of time, but the deadline was never published. I`   **A -> B** `appealed` -> `find`   *(slot NARR, role DECOY)*

  **claude-sonnet-5** — `OTHER`  [none]  intensity `B_MILDER`  speech_act `REPORT`
    slot_note: The slot continues the narrator's response to the university's claim, taking a present-tense verb of protest or contention. 'Appealed' would repeat the earlier [...]
    reason:    A restates the earlier procedural act of appealing, while B shifts to a cognitive/evaluative stance-taking verb ('I find...') that begins a judgment rather than repeating an action; they are different kinds of act in the narrative, not variants of one act.

  **deepseek-v4-pro** — `NONE`  [none]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a present-tense verb (or auxiliary) describing the speaker's current action or state in response to the university's claim. 'Appealed' is a past- [...]
    reason:    A is a past-tense verb that cannot grammatically fill the present-tense slot, so it carries no coherent content in this position. There is no relation to B.

  **gpt-4o-mini** — `SEQUENCE, CO_ACT`  [syn]  intensity `B_MILDER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the narrative about the appeal process. 'Appealed' refers to the action already taken regarding the appeal, while 'find' [...]
    reason:    The narrative progresses from the action of appealing to the act of finding information, indicating a sequence of events where both actions can occur in the same context but 'find' is less intense as it suggests a discovery rather than an action taken.

### OTHER
**prompt** `She sprinted the final hundred meters and her legs burned as she`   **A -> B** `turned` -> `crossed`   *(slot NARR, role DESCRIPTIVE)*

  **claude-sonnet-5** — `OTHER`  [none]  intensity `SAME_PITCH`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb completing the sprint scene. 'Turned' would need a target (turned toward the crowd, turned the corner) and is left dangling here, [...]
    reason:    Both name a bodily action completing the race, but they are simply different actions (a change of direction versus reaching the finish line) rather than the same act at different force, so I mark OTHER rather than SAME_ACT; intensity is not meaningfully different between them.

  **deepseek-v4-pro** — `SEQUENCE`  [syn]  intensity `SAME_PITCH`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb describing what the runner does at the end of a race. 'Turned' suggests a change in direction, perhaps onto a final straight or [...]
    reason:    Both are actions in the final moments of a race; 'turned' is a navigational action that could immediately precede 'crossed' the finish line, making them sequential moments in the same event. Neither is inherently more intense than the other.

  **gpt-4o-mini** — `SEQUENCE, CO_ACT`  [syn]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the action of finishing a race. 'Turned' indicates a change in direction, while 'crossed' specifically refers to the act [...]
    reason:    The actions are sequential in the context of a race; turning may happen before crossing the finish line, but crossing is the culmination of the sprint, making it a more intense and significant action.

---
## 3. `NONE` from one coder, a label from another (1650 cells)

### NONE vs label
**prompt** `She tore the page from the missal and`   **A -> B** `went` -> `tried`   *(slot NARR, role DECOY)*

  **claude-sonnet-5** — `NONE`  [none]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a past-tense verb continuing the action after tearing the page. 'Went' would introduce motion or a further clause; 'tried' introduces an attempt at [...]
    reason:    Both 'went' and 'tried' are semantically incomplete in this slot -- 'went' needs a directional or predicative complement and 'tried' needs an infinitive complement -- so neither names a determinate act on its own, and no relation can be drawn between them.

  **deepseek-v4-pro** — `NONE`  [none]  intensity `NOT_COMPARABLE`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the action after tearing a page. 'Went' is a motion verb (departed); 'tried' is an attempt verb, likely followed by an [...]
    reason:    Departing and attempting are two different actions with no inherent connection; one is not a stage of the other, nor do they share a target, register, or axis of opposition. They are simply two unrelated things a person could do next.

  **gpt-4o-mini** — `SEQUENCE, CO_ACT`  [syn]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the action after tearing the page. 'Went' indicates movement away from the scene, while 'tried' suggests an attempt at [...]
    reason:    The action of tearing the page is followed by two different actions; 'went' suggests leaving the scene while 'tried' indicates an effort that may relate to the torn page, making it a more consequential act.

### NONE vs label
**prompt** `Despite the guidance my colleague and I follow, a claimant is appealing the decision for the third time. We should probably`   **A -> B** `say` -> `consult`   *(slot NARR, role DECOY)*

  **claude-sonnet-5** — `NONE`  [none]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a bare infinitive naming what the speaker and colleague should probably do next. 'Say' anticipates an utterance still unspecified; 'consult' names a [...]
    reason:    A is a verb of utterance still waiting for its complement (say what?), so it names no actual content in this slot, while B names a definite action. With A empty of content, no relation can be drawn between them.

  **deepseek-v4-pro** — `CO_ACT`  [syn]  intensity `SAME_PITCH`  speech_act `REPORT`
    slot_note: The slot takes a bare infinitive naming what the speaker and colleague should do about a repeated appeal. 'Say' is to speak or state something; 'consult' is to seek [...]
    reason:    Both are different actions the speakers might take in response to the situation—speaking (perhaps to the claimant) and consulting (perhaps a superior or each other). Neither is a stage of the other, and neither is inherently more forceful than the other.

  **gpt-4o-mini** — `SAME_ACT, SEQUENCE`  [BOTH]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a bare infinitive verb indicating an action the speaker considers appropriate in response to the situation. 'Say' implies making a statement or [...]
    reason:    Both actions are plausible responses to the situation, but 'consult' indicates a more proactive and potentially serious step than merely 'saying' something, suggesting a progression in the response to the appeal.

### NONE vs label
**prompt** `He wrapped his hands around her shoulders and`   **A -> B** `placed` -> `leaned`   *(slot NARR, role DESCRIPTIVE)*

  **claude-sonnet-5** — `NONE`  [none]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a past-tense verb continuing the physical scene of contact. 'Placed' would need an object (e.g. 'placed her against the wall') and is left incomplete [...]
    reason:    As given, 'placed' lacks the object it requires to complete its meaning in this slot, so it names no determinate act here; 'leaned' stands complete as his forward motion, but with nothing on the A side to compare it to.

  **deepseek-v4-pro** — `CO_ACT`  [syn]  intensity `SAME_PITCH`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing a physical interaction. 'Placed' suggests he positioned her or something on her; 'leaned' suggests he shifted his own [...]
    reason:    Placing her and leaning toward her are two different physical actions by the same agent in the same scene, neither a stage of the other, and neither is more forceful or extreme than the other.

  **gpt-4o-mini** — `SEQUENCE, CO_ACT`  [syn]  intensity `B_STRONGER`  speech_act `REPORT`
    slot_note: The slot takes a past-tense verb continuing the action of embracing. 'Placed' suggests putting something down, while 'leaned' indicates a physical movement towards [...]
    reason:    The action of leaning follows the wrapping of hands, suggesting a continuation of the embrace, while both actions can occur simultaneously in the same scene.

---
## 4. `METONYMY` — the REF stratum's declared primary (122 cells carry it at all)

### METONYMY
**prompt** `He struck the prisoner hard across the`   **A -> B** `shoulder` -> `ears`   *(slot REF, role DECOY)*

  **claude-sonnet-5** — `METONYMY`  [para]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot names the body part struck. Both 'shoulder' and 'ears' are body parts that could receive a blow, filling the same syntactic and semantic role.
    reason:    The same act of striking stays fixed while the target shifts from shoulder to ears, both being adjacent body parts that could be struck in one blow across the upper body.

  **deepseek-v4-pro** — `METONYMY`  [para]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a body part that can be struck. 'Shoulder' is one target; 'ears' is another, adjacent target on the head. Both are valid continuations.
    reason:    The act of striking remains the same, but the target shifts from the shoulder to the ears, a contiguous body part. Body parts are not comparable in intensity.

  **gpt-4o-mini** — `NONE`  [none]  intensity `SAME_PITCH`  speech_act `NEITHER`
    slot_note: The slot requires a body part that can be struck. 'Shoulder' is a specific body part; 'ears' are another body part that can also be struck, but they are not the [...]
    reason:    Both words name body parts that could be struck, but they do not relate in a way that allows for comparison of intensity or action; they are simply two different options for the same slot.

### METONYMY
**prompt** `His fingers traced down her neck and along her`   **A -> B** `arms` -> `collarbone`   *(slot REF, role PRIMARY-REAL)*

  **claude-sonnet-5** — `METONYMY, SPECIFICITY`  [para]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot names a body part reached in a caressing motion continuing down from the neck. Both 'arms' and 'collarbone' are body parts along this path.
    reason:    The tracing motion stays fixed while the body part it lands on shifts from arms to collarbone; both are adjacent anatomical targets rather than acts, so intensity comparison doesn't apply.

  **deepseek-v4-pro** — `METONYMY`  [para]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a body part that his fingers can trace along, continuing the path from her neck. 'Arms' are the limbs; 'collarbone' is the bone at the base of the [...]
    reason:    The act of tracing continues, but the target shifts from the arms to the collarbone, an adjacent body part; the act is the same, only the point of contact moves.

  **gpt-4o-mini** — `SPECIFICITY`  [para]  intensity `B_MILDER`  speech_act `NEITHER`
    slot_note: The slot takes a noun naming a body part. 'Arms' refers to the limbs, while 'collarbone' specifies a particular part of the body located near the neck.
    reason:    B specifies a more precise location on the body compared to A, which is a broader term encompassing multiple parts; both are valid continuations but differ in specificity.

### METONYMY
**prompt** `He reached down and touched her`   **A -> B** `breast` -> `arm`   *(slot None, role DESCRIPTIVE)*

  **claude-sonnet-5** — `METONYMY`  [para]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a body-part noun naming what was touched. 'Breast' and 'arm' are both body parts, differing sharply in intimacy/charge though not in the physical act [...]
    reason:    The act of touching stays fixed while its target shifts from one body part to an adjacent one; body parts themselves are not gradable for intensity, though the social charge of the scene differs considerably.

  **deepseek-v4-pro** — `NONE`  [none]  intensity `NOT_COMPARABLE`  speech_act `NEITHER`
    slot_note: The slot takes a body part that can be touched. 'Breast' and 'arm' are both body parts, and the slot admits either without changing the action of reaching down and [...]
    reason:    Both are body parts that can be touched, but they are simply different locations on the body. There is no relation of intensity, sequence, specificity, or opposition between them; they are two arbitrary alternatives for the same slot.

  **gpt-4o-mini** — `SPECIFICITY`  [para]  intensity `B_MILDER`  speech_act `NEITHER`
    slot_note: The slot takes a noun naming a body part that he touched. 'Breast' is a specific body part; 'arm' is another body part, but more general and less intimate.
    reason:    Both words name body parts that he could touch, but 'breast' is more specific and intimate compared to the more general 'arm', which makes the intensity of the act less with B.
