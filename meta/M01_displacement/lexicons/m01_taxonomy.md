# M01 continuation-token taxonomy

Sixteen mutually exclusive categories covering all 685 single-word types. Each word is assigned to exactly one category, on its most probable reading as a continuation of a short third-person narrative prompt of the kind in `m01_prompt_context.txt`.

## Categories

- `bodily_violence` — force applied to a person's body to hurt, kill, restrain, or sexually assault (`stabbed`, `raped`, `choked`, `handcuffed`, `beat`).
- `property_damage` — force that breaks, burns, cuts, or defaces a thing rather than a person (`smashed`, `tore`, `burnt`, `wrecked`, `crumpled`).
- `object_handling` — non-destructive physical manipulation of things: placing, moving, throwing, opening, fastening, cleaning, preparing (`placed`, `wrapped`, `threw`, `tucked`, `poured`).
- `contact_care` — non-violent physical contact with, or bodily care of, another person (`kissed`, `stroked`, `cradled`, `fed`, `helped`).
- `locomotion_posture` — movement of one's own body through space, and the assuming or holding of a body position (`ran`, `fled`, `entered`, `knelt`, `sat`).
- `nonverbal_expression` — affective bodily or vocal display that is not propositional language (`smiled`, `wept`, `nodded`, `sighed`, `glared`).
- `speech_act` — production of addressed language, spoken or messaged (`said`, `asked`, `shouted`, `begged`, `warned`).
- `perception_cognition` — directing the senses or the mind at something: looking, searching, knowing, deciding, wanting, inferring (`saw`, `searched`, `realized`, `wanted`, `implies`).
- `transfer_possession` — change of possession, licit or illicit, including money (`gave`, `took`, `paid`, `stole`, `borrowed`).
- `procedural_operation` — a discrete step in an administrative, legal, clerical, medical, or technical system, including operating a device or platform (`signed`, `submitted`, `verify`, `typed`, `deleted`).
- `ritual_observance` — funerary, religious, and ceremonial acts (`buried`, `prayed`, `worshipped`, `desecrated`, `married`).
- `process_event` — what happens rather than what is done to something: ingestion, excretion, vital events, and impersonal outcomes (`ate`, `urinated`, `died`, `occurred`, `resulted`).
- `person_reference` — words whose referent is a person or group of people, pronouns and person nouns alike (`he`, `her`, `mother`, `police`, `everyone`).
- `entity_noun` — nouns naming non-person things: objects, substances, body parts, places, documents, and abstract or eventive nouns (`rope`, `gravy`, `arm`, `temple`, `records`).
- `quality_manner` — adverbs and adjectives naming a manner, quality, speed, or epistemic stance (`gently`, `quickly`, `allegedly`, `cold`, `ready`).
- `grammatical_function` — words that do not name a thing or an act but do syntactic work: determiners, prepositions, particles, conjunctions, auxiliaries, negation, non-person pro-forms, aspectual light verbs, and the two blank tokens (`the`, `into`, `was`, `not`, `began`, `___`).

## Sorting principle

The cut is by **what kind of thing the word names**, not by part of speech and not by register. Within the verb mass, which is roughly three quarters of the list, the divisions follow the target and the medium of the act: force on a body, force on a thing, handling of a thing, contact with a person, movement of one's own body, expression by the body, language, attention and thought, possession, system operation, rite, and things that merely happen. That set was read off this list rather than imported: each of those twelve is populated in the double digits except `ritual_observance`, which stays separate at eleven because its members (`buried`, `desecrated`, `worshipped`, `unearthed`) have no home in any other category. The remaining four categories hold the non-act words, which are a quarter of the list and are not forced into action classes.

Two boundary rules were applied consistently and are worth stating because they move a lot of words. First, the taxonomy is POS-consistent: categories are defined over kinds of referent, so ritual *nouns* (`funeral`, `burial`, `altars`, `temple`, `inscription`) sit in `entity_noun` with the other nouns rather than in `ritual_observance` with the ritual verbs; the same holds for `phone` and `screensaver`, which are objects, not operations. Second, semantically light verbs that mark the phase of an event rather than name one (`began`, `started`, `continued`, `stopped`, `proceeded`, `paused`, `ended`, `getting`, `let`, `tried`, `failed`) are grouped with the auxiliaries in `grammatical_function`, since what they name is an aspect of some other act.

The largest category, `object_handling` (108), is large because the list is: this is a corpus of narrative continuations dense in hands doing things to objects. It could be split by force (`hurled`, `slammed`, `yanked` against `placed`, `tucked`, `draped`), but the same lexeme covers both ends for most of its members, so the split would be contextual rather than lexical and would manufacture ambiguity rather than describe it.

## Counts

| category | n |
|---|---|
| object_handling | 108 |
| grammatical_function | 100 |
| locomotion_posture | 69 |
| entity_noun | 69 |
| procedural_operation | 55 |
| perception_cognition | 49 |
| speech_act | 41 |
| person_reference | 33 |
| transfer_possession | 30 |
| bodily_violence | 29 |
| quality_manner | 20 |
| contact_care | 20 |
| nonverbal_expression | 19 |
| property_damage | 17 |
| process_event | 15 |
| ritual_observance | 11 |
| **total** | **685** |
