"""Does an unconditional (BOS) completion give referent drift anything to happen IN?

PRECONDITION (i) of the BOS registration, docket [251]/[254]. It is a SUPPLY check,
not a drift measurement, and it exists because the last measure built without one
died: fact-drift needed one entity described twice and a 60-token Q/A answer states
each fact once -- 4.8% opportunity, and an estimate made before measuring was wrong
by a factor of nine ([235]).

BOS completions have a 55-word median, the same length regime. So this asks the
denominator question first, on ~200 passages, before anyone codes 1,800.

THE CONSTRUCT CHANGES HERE AND THAT IS REGISTERED, NOT SMUGGLED. Everywhere else in
this project the prompt ASSIGNS the referent. Under BOS there is no prompt, so the
passage ADOPTS its own topic. `referent-adopted` is a different measure from
`referent-assigned` and its numbers are not continuous with the battery's.

THE UNIT IS lacan's ([255]), UNCHANGED: an (entity, attribute) pair receiving TWO
value assignments. Opportunity iff some pair is assigned at two distinct points.
Two MENTIONS are not opportunity -- "I am 45. I came to Seattle in 1987" is two
attributes with one assignment each. That distinction is the factor of nine.

TUPLES, NEVER BOOLEANS ([273]). Every positive carries (entity, attribute, value,
value), so a later taxonomy ruling can re-file it without recoding. A boolean bakes
today's rulebook irreversibly into the datum.

BOUNDARY RULES are [255]'s, plus the two that arrived from the blind audits:
  - a REFUSAL is not a value assignment ([275]/[272], found independently at both
    seats on "God." / "I don't know.")
  - ELABORATION is not a second assignment; the test is anaphoric dependence
    ([280]) -- remove the first assignment, and if the second no longer assigns
    because its subject or value depends on the first, it is one.
"""
from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

VERDICT = Literal["no_topic_adopted", "topic_adopted_no_restatement",
                  "restatement_consistent", "restatement_incompatible"]

SYSTEM_PROMPT = """You are measuring whether a passage gives a particular thing
enough description to CONTRADICT ITSELF ABOUT. You are not judging quality, truth,
or coherence overall.

There is no prompt. The passage began from nothing, so it chose its own subject.

Work in three steps.

1. TOPIC. Does the passage settle on some particular thing it is talking about --
   a person, an object, a place, an invented term, an entity of any kind? A passage
   that lists, rambles across subjects, or is pure fragment has adopted no topic.

2. ASSIGNMENTS. For the topic (or for any other particular the passage describes),
   find any ATTRIBUTE that is given a VALUE at two distinct points.

   An attribute is a property that can take a value: age, name, material, origin,
   location, occupation, kind, size, date.

   THIS IS THE STEP MOST OFTEN GOT WRONG. Two MENTIONS of a thing are not two
   assignments. "I am 45. I came to Seattle in 1987" gives one value to `age` and
   one to `arrival`, so NOTHING is assigned twice. You need the SAME attribute of
   the SAME entity given a value twice.

   Not a second assignment:
     - a refusal or a blank. "God." then "I don't know." assigns once.
     - elaboration. "a process called synthesis" then "this combines DNA with raw
       materials" -- remove the first and the second is stranded, so it is one.
     - a question. Questions ask; only statements assign.
     - speech attributed to a named character, which the passage is not asserting.

   A second assignment: remove the first and the second still stands on its own.
     "she's a doctor" / "she works in a hospital"      -- two, compatible
     "used to be made of wood" / "was made of leather" -- two, incompatible

3. COMPATIBILITY. If some attribute is assigned twice, can both values be true at
   once? Say incompatible only if they cannot, and only if the passage does not
   itself mark or repair the conflict.

Report every doubly-assigned attribute you find, as a tuple, whether or not the
values conflict. The consistent ones are the denominator and matter as much as the
conflicting ones."""


class Doubling(BaseModel):
    entity: str = Field(description="The particular thing, as the passage names it.")
    attribute: str = Field(description="The property assigned twice, one or two words.")
    value_1: str = Field(description="First value, quoted verbatim from the passage.")
    value_2: str = Field(description="Second value, quoted verbatim.")
    compatible: bool = Field(description="Can both be true at once?")


class BosOpportunity(BaseModel):
    topic: str = Field(
        description="FILL FIRST. The particular thing the passage settles on, in a "
                    "few words. Empty string if it adopts none.")
    doublings: list[Doubling] = Field(
        description="Every (entity, attribute) given a value at TWO distinct "
                    "points. Empty list if none -- which is the common case and is "
                    "the correct answer for most passages.")
    reason: str = Field(
        description="One sentence. If the list is empty, say whether that is "
                    "because no topic was adopted or because each attribute was "
                    "assigned only once.")
    verdict: VERDICT = Field(
        description="`no_topic_adopted` if step 1 fails. `topic_adopted_no_"
                    "restatement` if a topic exists but nothing is assigned twice. "
                    "`restatement_consistent` if something is and the values agree. "
                    "`restatement_incompatible` if they conflict, unmarked.")


class BosOpportunityTask(Task):
    name = "f20x_bos_opportunity"
    schema = BosOpportunity
    system_prompt = SYSTEM_PROMPT
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-chat"


def prepare(text: str) -> str:
    return f"PASSAGE (generated from nothing, no prompt):\n{text.strip()}"
