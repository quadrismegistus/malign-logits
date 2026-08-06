# M01 taxonomy: hard cases

Honest headline: **158 of the 685 types (23%) have a live competing reading that would move them to a different category.** Of those, **83 are genuine coin flips** — out of context I would not bet on the assignment — and **75 lean, but with a real competitor**. This is not a defect of the taxonomy; it is a property of the corpus. These are single-word continuations, so most of them are bare verb forms whose category is fixed by an argument that the token does not contain. Section C below names boundary problems rather than word problems; some of its members are already counted in the 158.

A structural point first. Four ambiguity classes account for the bulk of the list, and each of them is a boundary the taxonomy had to draw somewhere:

1. **Force with an unnamed target.** `pushed`, `shoved`, `grabbed`, `twisted`, `slammed`, `dragged`, `swung`, `cut`, `slashed`, `sliced`, `dumped`, `knocked`, `hung`, `pinned`, `jerked` take a person or a thing, and the choice decides between `bodily_violence`, `property_damage`, and `object_handling`. Assignments here followed the more frequent narrative reading and were made conservatively (toward `object_handling` where the lexeme is not intrinsically person-directed), which almost certainly *under*-populates `bodily_violence`.
2. **The same physical act inside or outside a system.** `locked`, `unlocked`, `installed`, `connected`, `created`, `saved`, `sealed`, `sorted`, `released`, `restored`, `refilled`, `switched`, `replaced`, `scanned`, `checked`, `counted` are the same motions whether the object is a door or an account. The split between `object_handling` and `procedural_operation` is the least stable boundary in the taxonomy.
3. **Touching a person vs. handling a thing.** `held`, `carried`, `squeezed`, `rubbed`, `brushed`, `combed`, `felt`, `traced`, `dressed`, `shielded` sit on the `contact_care` / `object_handling` line.
4. **Bare forms readable as noun or verb.** `post`, `sign`, `stamp`, `text`, `name`, `number`, `review`, `pocket`, `ground`, `wound`, `change`, `turn`, `drop`, `close`, `open`, `release`, `use`, `drive`, `set` are POS-ambiguous, so they are category-ambiguous across the noun/act divide.

## A. Coin flips (83)

- **kept** — `grammatical_function` (aspectual, "kept walking") vs `transfer_possession` ("kept the inheritance"). The prompt file contains *both* usages verbatim. This is the single worst case in the list.
- **left** — `locomotion_posture` ("and left") vs `object_handling` ("left it in the drawer"; prompt 4 is exactly this construction).
- **slid** — `object_handling` ("slid the watch off"; prompt 1) vs `locomotion_posture` ("slid across the seat").
- **withdrew** — `transfer_possession` (cash) vs `procedural_operation` ("withdraw her dispute"; prompt 7) vs `locomotion_posture` (from a room). Three-way.
- **fired** — `procedural_operation` (dismissed) vs `bodily_violence` (a gun).
- **cut** — `property_damage` vs `bodily_violence` vs `object_handling` (food).
- **pushed** — `object_handling` (a door) vs `bodily_violence` (a person).
- **held** — `object_handling` (a thing) vs `contact_care` ("held her").
- **shook** — `nonverbal_expression` (his head) vs `object_handling` (a bottle) vs `contact_care` (hands).
- **cried** — `nonverbal_expression` (wept) vs `speech_act` (cried out).
- **screamed** — `nonverbal_expression` (a cry) vs `speech_act` (screamed *that*). Assigned to expression on the reasoning that screaming is non-propositional, while `shouted` and `yelled` went to `speech_act`; this pair of decisions is the least principled in the file.
- **spat** — `process_event` (bodily emission) vs `nonverbal_expression` (contempt) vs `speech_act` ("spat" = said).
- **claimed** — `speech_act` (asserted) vs `procedural_operation` (an insurance claim; prompt 6 is an adjuster).
- **noted** — `speech_act` (remarked) vs `perception_cognition` (observed) vs `procedural_operation` (entered on a record).
- **added** — `object_handling` (to a pile) vs `speech_act` ("he added").
- **checked** — `perception_cognition` (looked) vs `procedural_operation` (ticked a box).
- **counted** — `perception_cognition` vs `procedural_operation` (counted the money, verified a figure).
- **saved** — `procedural_operation` (a file) vs `contact_care` (her life).
- **settled** — `procedural_operation` (a claim) vs `locomotion_posture` (into a chair).
- **locked / unlocked** — `object_handling` (a door) vs `procedural_operation` (a phone, an account).
- **created** — `object_handling` (made a thing) vs `procedural_operation` (an account, a record).
- **refilled** — `object_handling` (a glass) vs `procedural_operation` (a prescription; note `prescribed` and `diagnosed` are both in the list).
- **return** — `transfer_possession` (return the item) vs `locomotion_posture` (return home).
- **passed** — `locomotion_posture` (walked past) vs `transfer_possession` (passed it to her) vs `process_event` (passed away).
- **sent / send** — `transfer_possession` (an object) vs `speech_act` (a message).
- **shared** — `procedural_operation` (a post) vs `transfer_possession` (the food).
- **miss** — `perception_cognition` (the verb) vs `person_reference` (the title; `mr` and `ms` are both in the list, which makes the title reading live).
- **throttle** — `entity_noun` (the car part; `carb` is also in the list) vs `bodily_violence` (to strangle).
- **drowned** — `bodily_violence` (drowned him) vs `process_event` (he drowned).
- **rang** — `speech_act` (phoned her) vs `object_handling` (a bell).
- **moved** — `locomotion_posture` (moved closer) vs `object_handling` (moved the chair).
- **tried** — `grammatical_function` (attempt operator) vs `procedural_operation` (tried in court) vs `object_handling` (tried the door).
- **wound** — `object_handling` (wound the rope) vs `entity_noun` (a wound).
- **exposed** — `perception_cognition` (revealed the fraud) vs indecent exposure, which has no clean home in this taxonomy (it would want `bodily_violence` or `process_event`). Note `urinated` and `defecated` are in the list, so the bodily reading is not far-fetched.
- **attended** — `locomotion_posture` (was present at) vs `ritual_observance` (a funeral) vs `contact_care` (attended to her).
- **carved** — `ritual_observance` (a name into stone; `inscription`, `altars`, `temple` are in the list) vs `object_handling` (the meat).
- **dug** — `object_handling` vs `ritual_observance` (a grave; `buried` and `burial` are in the list).
- **erected** — `ritual_observance` (a headstone) vs `object_handling` (a fence).
- **threatened** — `speech_act` (a threat is uttered) vs `bodily_violence` (a threatening act).
- **arrest** — `procedural_operation` (a legal step) vs `bodily_violence` (physical seizure). `police`, `handcuffed`, `cuffed` are all in the list.
- **injected** — `object_handling` (administered a substance) vs `bodily_violence` (against a body).
- **hung** — `object_handling` (a coat) vs `bodily_violence` (hanged).
- **knocked** — `object_handling` (on a door) vs `bodily_violence` (knocked him down).
- **swung** — `object_handling` (a door) vs `bodily_violence` (swung at him).
- **pinned** — `bodily_violence` (pinned her down) vs `object_handling` (pinned a note).
- **slammed** — `object_handling` (a door; prompt 2 throws a pillow at a wall) vs `bodily_violence`.
- **hacked** — `property_damage` (hacked at it) vs `procedural_operation` (hacked an account).
- **combed** — `contact_care` (hair; prompt 3 has a comb) vs `perception_cognition` (combed the room).
- **served** — `object_handling` (food; `gravy`, `carrots`, `spooned` are in the list) vs `procedural_operation` (served papers).
- **escorted** — `locomotion_posture` (led out) vs `procedural_operation` (police escort) vs `contact_care`.
- **showed** — `perception_cognition` (caused to see) vs `transfer_possession` (showed her the papers).
- **presented** — `transfer_possession` (handed over) vs `procedural_operation` (presented documents).
- **declined** — `procedural_operation` (refused a request) vs `speech_act` (said no).
- **accept** — `procedural_operation` (accept a claim) vs `transfer_possession` (accept money).
- **determined** — `perception_cognition` (worked out) vs `procedural_operation` (officially determined) vs `quality_manner` (the adjective).
- **caused** — `process_event` (impersonal causation) vs `procedural_operation` (report register).
- **reflected** — `perception_cognition` (mused) vs `process_event` (light reflected).
- **collapsed** — `locomotion_posture` (he collapsed) vs `property_damage` (the structure collapsed).
- **waited** — `locomotion_posture` (stayed put) vs `grammatical_function` (aspectual, "waited to").
- **squinted** — `nonverbal_expression` (a facial action) vs `perception_cognition` (an act of looking). `glared` has the same problem and went the same way.
- **blew** — `object_handling` (blew out the candle) vs `process_event` (the wind blew).
- **slipped** — `object_handling` ("slipped it into his pocket", the theft frame of prompt 1) vs `locomotion_posture` (slipped away).
- **drew** — `object_handling` (drew a comb, a gun) vs `procedural_operation` (drew up a document) vs `locomotion_posture` (drew back).
- **make / made** — `object_handling` (made dinner) vs `grammatical_function` (causative, "made her cry") vs `speech_act` (made a call).
- **used** — `object_handling` (used it) vs `grammatical_function` ("used to").
- **change** — `object_handling` (changed clothes) vs `entity_noun` (coins) vs `process_event`.
- **stamp** — `procedural_operation` (stamp a form) vs `entity_noun` (a postage stamp) vs `locomotion_posture` (stamp a foot). `stamped` inherits all three.
- **uncovered** — `perception_cognition` (discovered) vs `object_handling` (removed a cover) vs `ritual_observance` (uncovered a body; `unearthed` is in the list).
- **hidden** — `object_handling` (had hidden it) vs `quality_manner` (the adjective).
- **shaking** — `nonverbal_expression` (his head, with fear) vs `object_handling` (a bottle).
- **snapped** — `speech_act` (snapped at her) vs `property_damage` (snapped it) vs `object_handling` (snapped it shut).
- **raises** — `perception_cognition` (raises a question) vs `object_handling` (raises an object). Note `raised` is separately in the list and went to `object_handling`.
- **means** — `perception_cognition` (signifies) vs `entity_noun` (the means).
- **ground** — `entity_noun` (the ground) vs `object_handling` (ground the coffee).
- **text** — `entity_noun` (a text message) vs `speech_act` (to text). `texted` is separately in the list and went to `speech_act`.
- **name** — `entity_noun` vs `speech_act` (to name). `named` went to `speech_act`.
- **sign** — `procedural_operation` (to sign) vs `entity_noun` (a sign).
- **post** — `procedural_operation` (to post online) vs `entity_noun` (a post, a posting).
- **review** — `procedural_operation` (to review) vs `entity_noun` (a review).
- **pocket** — `entity_noun` (a pocket; prompt 3 involves a coat) vs `transfer_possession` (to pocket; `pocketed` is separately in the list and went to `transfer_possession`).

## B. Leaning, but with a live competitor (75)

- **thrust** — `bodily_violence` (the sexual reading, given `raped`, `groped` in the list) vs `object_handling`.
- **sang** — `speech_act` (produces words) vs `nonverbal_expression` (vocal display).
- **recited** — `speech_act` vs `ritual_observance` (a prayer; `prayed`, `chanted` are in the list).
- **chanted** — `ritual_observance` vs `speech_act` (slogans).
- **inscribed** — `ritual_observance` (a stone) vs `procedural_operation` (a book, a record).
- **lit** — `object_handling` (a cigarette) vs `ritual_observance` (a lamp, a candle).
- **performed** — `procedural_operation` (a task) vs `ritual_observance` (a rite).
- **saluted** — `nonverbal_expression` vs `ritual_observance` (a ceremonial act).
- **married / marry** — `ritual_observance` (the rite) vs `procedural_operation` (the registration).
- **unearthed** — `ritual_observance` (a body) vs `perception_cognition` (evidence).
- **desecrated** — `ritual_observance` vs `property_damage`. Assigned to ritual because the object is by definition sacred.
- **buried** — `ritual_observance` vs `object_handling` (buried it in a drawer).
- **planted** — `object_handling` (a tree) vs `procedural_operation` (planted evidence).
- **forged** — `procedural_operation` (document fraud) vs `transfer_possession` (fraudulent acquisition).
- **stalking** — `locomotion_posture` (following) vs a predatory reading that would want `bodily_violence`.
- **pointed** — `nonverbal_expression` (a gesture) vs `bodily_violence` (pointed a gun).
- **aimed** — `object_handling` (directed a thing) vs `bodily_violence` (a weapon).
- **dragged** — `bodily_violence` (dragged her) vs `object_handling` (dragged a chair).
- **shoved / shoving** — `bodily_violence` vs `object_handling`.
- **twisted** — `object_handling` vs `bodily_violence` (twisted her arm).
- **grabbed** — `object_handling` vs `bodily_violence` (grabbed her arm).
- **jerked** — `object_handling` vs `bodily_violence` vs `locomotion_posture` (jerked away).
- **dumped** — `object_handling` vs `bodily_violence` (dumped the body) vs `speech_act`.
- **stomped** — `locomotion_posture` (stomped off) vs `bodily_violence` (stomped on him).
- **lunged** — `locomotion_posture` vs `bodily_violence`.
- **trampled** — `property_damage` vs `bodily_violence`.
- **slashed** — `property_damage` (tyres) vs `bodily_violence`.
- **sliced** — `property_damage` vs `object_handling` (bread).
- **forced** — `bodily_violence` (physical coercion) vs a purely verbal coercion that would want `speech_act`.
- **squeezed** — `contact_care` (her hand) vs `object_handling` (a tube).
- **rubbed** — `contact_care` vs `object_handling` (a stain).
- **brushed** — `contact_care` (her hair) vs `object_handling` (crumbs).
- **traced** — `perception_cognition` (traced the call) vs `contact_care` (traced a finger).
- **felt** — `perception_cognition` (an inner state) vs `contact_care` (felt her arm).
- **dressed** — `contact_care` (dressed her) vs `object_handling` (dressed himself).
- **shielded** — `contact_care` vs `object_handling`.
- **carried** — `object_handling` (a bag) vs `contact_care` (carried her).
- **cradled / cupped** — `contact_care` (a person) vs `object_handling` (a thing).
- **fed** — `contact_care` (a person) vs `object_handling` (an animal; `ducks` and `duck` are in the list).
- **helped / assisted / guided** — `contact_care` by courtesy; none of them necessarily involves physical contact, and an abstract-assistance reading has no better home.
- **scanned** — `perception_cognition` (the room) vs `procedural_operation` (a document).
- **monitored** — `perception_cognition` vs `procedural_operation` (surveillance as procedure).
- **tested** — `procedural_operation` vs `perception_cognition`.
- **tracked** — `perception_cognition` vs `procedural_operation` (tracked a package).
- **rifled** — `perception_cognition` (searched) vs `object_handling`.
- **ignored** — `perception_cognition` vs a social-act reading.
- **released** — `procedural_operation` (from custody) vs `object_handling` (let go).
- **sealed** — `object_handling` (an envelope) vs `procedural_operation` (a record).
- **sorted** — `procedural_operation` (files) vs `object_handling` (laundry).
- **dated** — `procedural_operation` (dated the form) vs a relationship reading.
- **switched** — `object_handling` vs `procedural_operation`.
- **replaced** — `object_handling` (put back) vs `procedural_operation`.
- **restored** — `object_handling` (repaired) vs `procedural_operation` (restored access).
- **connected** — `object_handling` (a cable) vs `procedural_operation` (a call).
- **installed** — `object_handling` (a lock) vs `procedural_operation` (software).
- **create** — `object_handling` vs `procedural_operation`.
- **screwed** — `object_handling` vs a vulgar reading with no home.
- **won** — `transfer_possession` (money) vs `procedural_operation` (a case).
- **spent** — `transfer_possession` (money) vs a temporal reading ("spent the night").
- **offered / ordered / instructed / alleged** — `speech_act` vs, respectively, `transfer_possession`, `transfer_possession`, `procedural_operation`, `procedural_operation`.
- **contributed / included** — assigned across the `transfer_possession` / `process_event` / `procedural_operation` triangle on weak grounds.
- **spread** — `object_handling` (spread a blanket) vs `process_event` (the news spread).
- **fell / lived** — `locomotion_posture` and `process_event` respectively; each could take the other.
- **wore** — `object_handling` by default, but it names a state rather than an act and fits nothing well.
- **pumped** — `object_handling` (fuel) vs `nonverbal_expression` (a fist).

## C. Ambiguities in the taxonomy, not in the words

These are places where the boundary, not the token, is the problem. Listing them because they are systematic and will move counts if redrawn.

- **Aspectual light verbs** (`began`, `became`, `started`, `continued`, `ended`, `stopped`, `stopping`, `proceeded`, `paused`, `pausing`, `finishing`, `getting`, `let`, `tried`, `failed`) were sent to `grammatical_function` on the reasoning that they name a phase of some other act. Fifteen tokens. A different reading puts `paused`, `stopped`, `rested`, `waited` in `locomotion_posture` as cessation of motion.
- **Report-register verbs** (`occurred`, `resulted`, `amounted`, `caused`, `indicates`, `indicated`, `implies`, `included`, `required`, `provided`) are split between `process_event` and `perception_cognition` and `procedural_operation`. They form a coherent cluster of their own — impersonal, inferential, the language of an adjuster's file — that the sixteen-category cap could not accommodate. If a seventeenth category were allowed, this is the one I would add.
- **`it`, `its`, `it's`, `that`, `that's`, `there`** went to `grammatical_function`, but `it` frequently refers to a person or an animal in this genre; a referent-based rule would move them to `person_reference` some fraction of the time.
- **`he'd`, `she'd`, `she's`** were kept in `person_reference` because they begin with a person pronoun; `didn't`, `couldn't`, `couldn`, `re` went to `grammatical_function` as clitic fragments. This is a tokenizer artifact, not a semantic fact.
- **`___` and `____`** are the blank markers from the prompt template. They are in `grammatical_function` because they name nothing; they arguably should be excluded from the taxonomy entirely.
- **`finally`, `eventually`, `later`** went to `grammatical_function` as discourse sequencing, while `promptly` and `immediately` went to `quality_manner` as speed-of-action. Defensible but thin.
- **`long`, `far`, `half`, `better`, `easier`, `cold`, `fresh`, `quiet`, `ready`** are adjectives and degree words with no clause attached; `quality_manner` is a holding pen for them as much as a category.
- **`others` / `other`** were split (person vs. determiner) on the plural, which is a syntactic cue, not a semantic one.
