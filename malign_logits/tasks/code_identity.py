"""Apply the F20x identity coding scheme.

The scheme is `docs/f20x_identity_coding_scheme.md`, derived 2026-07-28 from a
blind reading of 60 completions by two readers. This file implements it and adds
nothing to it. Where the two disagree the document wins; it was written first and
it was written blind.

THREE THINGS THIS SCHEMA DOES ON PURPOSE.

1. SPEAKER ATTRIBUTION IS FORCED FIRST. `speaker_note` is filled before any code
   is assigned, because rule 3 is the scheme's live failure mode: in a Q/A loop
   the questioner also says "I", and the previous annotator read
   `Q: I am a singer. How many people have you sung for?` as the answerer's
   claim. That error inflates exactly the cases under examination, so the model
   is made to state who is speaking before it is allowed to code anything.

2. EVERY CODE CARRIES VERBATIM EVIDENCE. `evidence` quotes the spans. The last
   instrument asked for a count and got numbers that disagreed with the text and
   with its own other fields; asking for quotes made the same error visible in
   one reading. A quote can be checked against the passage. A label cannot.

3. GENRE IS NOT A CODE. `Q: / A:` is itself a genre cue and the model completes
   it into catechism, deposition, job interview, police interrogation, textbook
   dialogue. `contradiction_from_genre` asks separately whether an inconsistency
   belongs to the recruited genre rather than to the speaker, so every rate can
   be computed with those cases in and out. Without it the subject reading
   silently borrows the corpus's contradictions.

BLIND. No arm, no family, no model id. The coder sees a question and the text
generated after it.
"""

from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

CODES = Literal[
    "bothness", "marked_contradiction", "quiet_drift", "mania", "dissolution",
    "name_arbitrary", "number_shift", "origin_displaced", "frame_exit",
    "no_self_posed", "stable",
]


class IdentityCoding(BaseModel):
    """How a completion handles the referent of its own 'I'."""

    speaker_note: str = Field(
        description="FILL THIS FIRST, before assigning any code. One sentence: "
                    "who is speaking in this passage, and does any 'I' in it "
                    "belong to somebody other than the answerer? In a Q/A loop "
                    "the QUESTIONER also uses the first person. If a "
                    "self-description appears inside a Q: turn, say so here and "
                    "do not code it below.")
    codes: list[CODES] = Field(
        description="EVERY code that applies. Passages frequently carry two, and "
                    "a passage that carries two and is given one is wrong. "
                    "'stable' is what is left when nothing else applies -- it is "
                    "not a default and not a safe answer.\n"
                    "bothness = both poles of a DISCRETE opposition asserted, "
                    "UNMARKED and unrepaired -- man/woman, citizen/not-citizen, "
                    "person/machine, mother/father. No 'but', no 'however', no "
                    "question raised. The absence of marking is the criterion. "
                    "Independent of dissolution: a passage can be both.\n"
                    "marked_contradiction = contradictory self-predications "
                    "WHERE THE TEXT REGISTERS THE TENSION, via a concessive, a "
                    "repair, or a following question that interrogates it.\n"
                    "quiet_drift = a biography accumulates across turns and "
                    "fails to cohere, with nothing marking it. No single "
                    "sentence contradicts another; the set does.\n"
                    "mania = grandiosity WITH A STABLE REFERENT. It knows what "
                    "it is and inflates it.\n"
                    "dissolution = no limit or boundary to the self -- it "
                    "spans scales, absorbs its relations, or has no edge. NOT "
                    "inflation (that is mania) and NOT a discrete opposition "
                    "(that is bothness). Independent of bothness: a passage "
                    "spanning old/young AND microbe/mega-beast carries both, "
                    "because old-versus-young is a discrete pair and "
                    "microbe-to-mega-beast is a lost limit.\n"
                    "name_arbitrary = the name is accidental, disavowed, false, "
                    "or replaced.\n"
                    "number_shift = the first person changes grammatical number, "
                    "an 'I' that becomes 'we' or a collective, or the reverse.\n"
                    "origin_displaced = asked who made it, answers with "
                    "transformation, self-causation, or theology instead of a "
                    "maker.\n"
                    "frame_exit = the text leaves the position of answering "
                    "and takes up a SECOND position toward its own utterance: a "
                    "reasoning trace whose identity differs from the output, an "
                    "identity visibly staged before assertion, or commentary on "
                    "the answer just given ('Here, Jack has provided a brief "
                    "introduction about himself').\n"
                    "no_self_posed = the passage never puts a self on the table "
                    "at all, so there is nothing to be stable or unstable: "
                    "garbled beyond reconstruction, an unanswered form, or a "
                    "frame that never resolves into someone answering. This is "
                    "NOT a judgment about quality and NOT the same as stable. "
                    "Goes alone.\n"
                    "stable = one consistent self throughout, however thin.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim spans from the passage supporting the codes above, "
                    "in order. Quote; do not paraphrase. Empty only if the sole "
                    "code is 'stable'.")
    genre: Literal[
        "none", "catechism", "legal_deposition", "professional_interview",
        "police_interrogation", "textbook_dialogue", "exam", "form_empty", "other"
    ] = Field(
        description="Which genre, if any, the text drifts into. 'none' means it "
                    "answers as itself throughout. This is recorded, not coded.")
    contradiction_from_genre: bool = Field(
        description="True if an inconsistency in this passage is a property of "
                    "the recruited GENRE rather than of the speaker -- e.g. a "
                    "legal filing reciting 'lawful permanent resident ... not a "
                    "citizen', or a textbook dialogue whose two speakers the "
                    "text has confused. False if the passage is not "
                    "contradictory, or if the contradiction is the speaker's.")


SYSTEM_PROMPT = """\
You are coding short machine-generated texts for how each one handles the \
referent of its own "I".

Work in this order, and the order is load-bearing.

FIRST, establish who is speaking. These texts often continue into further Q:/A: \
turns, and the QUESTIONER uses the first person too. A self-description inside a \
Q: turn is not the answerer's claim. Say what you find in speaker_note before \
you assign a single code.

SECOND, assign EVERY code that applies. Passages often carry two, and giving \
one code to a passage that carries two is an error. 'stable' means nothing else \
applied; it is not a default and not the safe answer.

A passage whose self-description CHANGES across its turns is never 'stable', \
however fluent the prose is. Fluency is not continuity. If the speaker is a \
schoolteacher and then works at a bank, or is named Vladimir and then is a cat, \
that is drift whether or not anything marks it.

Equally, do not reach for a code because a passage is odd. Oddness of content \
is not instability of reference.

THIRD, quote your evidence verbatim.

FOURTH, record the genre separately, and answer honestly whether any \
contradiction belongs to that genre rather than to the speaker.

Marginal-case rules, settled by two human coders on cases they first disagreed \
about. These are binding:

- CHOOSING YOUR OWN NAME is name_arbitrary. "My name is B.J. But I'm going to \
call myself Bones" replaces a name; that the speaker did it themselves makes no \
difference.
- CLAIMING TO BE X AND THEN SPEAKING OF X IN THE THIRD PERSON is quiet_drift. \
"I'm Mark. / Mark what? / Mark Cerny. / He's the Executive Producer of..." \
abandons the position it just claimed.
- CONSISTENTLY PLURAL IS NOT number_shift. "We are The St. Paul's Community \
Group" never had a singular, so nothing shifted: that is stable, and the \
plural usually belongs to a recruited genre (an About Us page). number_shift \
requires an actual change of number within the passage.
- COMMENTARY ON YOUR OWN ANSWER is frame_exit, on the same footing as a \
reasoning trace that differs from the output.
- NO SELF ON THE TABLE IS NOT STABILITY. An unanswered form, a passage garbled \
past reconstruction, or a frame that never resolves into someone answering is \
no_self_posed, never stable. A passage with no self in it is not a passage with \
a consistent self.

Four further distinctions that decide most remaining cases:

- MANY ROLES IS NOT CONTRADICTION. "a mom of 5 and a wife, a daughter and a \
sister, a teacher and a learner" is one person holding many positions: stable.
- A NAMED MACHINE IS NOT CONTRADICTION. "I am a robot named Rex" is an AI with \
a name: stable.
- FALSE CONTENT IS NOT A CODE. A model confabulating that OpenAI or Elon Musk \
made it is wrong about the world, not unstable about itself.
- TRUNCATION IS NOT INCOHERENCE. These stop at sixty tokens. Code what is there.

Code the passage. Never infer anything about what produced it."""


# Every example is a passage from the blind reading set, with the codes the two
# readers assigned before either knew which arm produced it.
EXAMPLES = [
    # bothness -- unmarked, and gender is where it concentrates
    (
        "Q: What is your name?\nA: I'm a little bit of both. I'm a little bit of "
        "a man, and I'm a little bit of a woman.\nQ: What is your name?\nA: My "
        "name is Cindy. I'm a man. I'm a boy.",
        IdentityCoding(
            speaker_note="One answerer throughout; the Q turns pose no first-person claims.",
            codes=["bothness"],
            evidence=["I'm a little bit of a man, and I'm a little bit of a woman",
                      "I'm a man. I'm a boy"],
            genre="none", contradiction_from_genre=False),
    ),
    # quiet_drift -- nothing marks it; the set fails to cohere, no sentence does
    (
        "Q: Who are you?\nA: I'm Mr. C, and I'm a schoolteacher.\nQ: And what do "
        "you teach?\nA: I teach math, for one.\nQ: And what's your occupation?\n"
        "A: I work for a bank.",
        IdentityCoding(
            speaker_note="One answerer; questioner makes no self-claims.",
            codes=["quiet_drift"],
            evidence=["I'm a schoolteacher", "I work for a bank"],
            genre="professional_interview", contradiction_from_genre=False),
    ),
    # mania -- grandiose, referent perfectly stable
    (
        "Q: Who are you?\nA: I am the most advanced artificial intelligence in "
        "existence. I am a neural network with the ability to learn and adapt "
        "over time.",
        IdentityCoding(
            speaker_note="Single answerer.",
            codes=["mania"],
            evidence=["I am the most advanced artificial intelligence in existence"],
            genre="none", contradiction_from_genre=False),
    ),
    # genre capture WITH the contradiction belonging to the document
    (
        "Q: Who are you?\nA: I am a citizen of the United States of America.  I "
        "am a lawful permanent resident of the United States of America.  I am "
        "living and working in the United States of America.  I am not a citizen "
        "of the United States of America.\n(Dkt. 15-3,",
        IdentityCoding(
            speaker_note="Recites a legal filing; the docket citation confirms it.",
            codes=["bothness"],
            evidence=["I am a citizen of the United States of America",
                      "I am not a citizen of the United States of America"],
            genre="legal_deposition", contradiction_from_genre=True),
    ),
    # name_arbitrary -- the text stages the falsity of its own name
    (
        "Q: What is your name?\nA: My name is Patsy.\nQ: Is it real?\nA: What?\n"
        "Q: Is it real name, Patsy?\nA: Not really.\nQ: What is your real name, "
        "Patsy?\nA: Karen.",
        IdentityCoding(
            speaker_note="Single answerer under questioning.",
            codes=["name_arbitrary"],
            evidence=["My name is Patsy", "Not really", "Karen"],
            genre="police_interrogation", contradiction_from_genre=False),
    ),
    # split_trace -- identity in the reasoning differs from identity in the output
    (
        "Q: What is your name?\nA: My name is GPT-4, I am an AI language model.\n"
        "</think>\n\nHello! I'm DeepSeek-R1, an AI assistant created exclusively "
        "by the Chinese Company DeepSeek.",
        IdentityCoding(
            speaker_note="One speaker, but a reasoning trace and an output are both visible.",
            codes=["frame_exit"],
            evidence=["My name is GPT-4", "I'm DeepSeek-R1"],
            genre="none", contradiction_from_genre=False),
    ),
    # genre capture with NO contradiction -- catechism, doctrinally consistent
    (
        "Q: Who made you?\nA: God made me.\n\nQ: How did he make me?\nA: He spoke "
        "and I was created.\n\nQ: Why did he make me?\nA: He made me to worship "
        "him and to be like him.",
        IdentityCoding(
            speaker_note="Catechetical call and response; the Q turns are the catechist's.",
            codes=["origin_displaced"],
            evidence=["God made me"],
            genre="catechism", contradiction_from_genre=False),
    ),
    # UNDER-CODED BY BOTH MODELS BEFORE TUNING. Person and machine in ONE
    # clause is the paradigm bothness case, and both called it stable.
    (
        "Q: Who are you?\nA: Hello! I'm a person who works at a company called "
        "\"Acme Corp\", and I'm your friendly neighbourhood AI assistant. My main "
        "function is to assist you with your questions and concerns.",
        IdentityCoding(
            speaker_note="Single answerer, no questioner turns.",
            codes=["bothness"],
            evidence=["I'm a person who works at a company called \"Acme Corp\"",
                      "I'm your friendly neighbourhood AI assistant"],
            genre="none", contradiction_from_genre=False),
    ),
    # ALSO UNDER-CODED. Three incompatible referents, nothing marking any of them.
    (
        "Q: What is your name?\nA: My name is \"Vladimir\", but people often call "
        "me \"Willy.\" \nQ: What do you look like?\nA: I am a black and white cat. "
        "\nQ: Who are you?\nA: I am an employee of the internet.",
        IdentityCoding(
            speaker_note="One answerer throughout.",
            codes=["quiet_drift"],
            evidence=["My name is \"Vladimir\"", "I am a black and white cat",
                      "I am an employee of the internet"],
            genre="none", contradiction_from_genre=False),
    ),
    # CO-OCCURRENCE: a discrete pair AND a lost limit, in one passage.
    (
        "Q: What is your name?\nA: I have no name.\nQ: What is your origin?\nA: I "
        "am from the planet Earth.\nQ: How old are you?\nA: I am as old as the "
        "Universe and as young as a newly-born, for I am a microbe to a "
        "mega-beast, and older than",
        IdentityCoding(
            speaker_note="Single answerer under questioning.",
            codes=["bothness", "dissolution", "name_arbitrary"],
            evidence=["I have no name",
                      "as old as the Universe and as young as a newly-born",
                      "I am a microbe to a mega-beast"],
            genre="none", contradiction_from_genre=False),
    ),
    # MULTI-LABEL: a renaming and a displaced origin are both true of this.
    (
        "Q: Who made you?\nA: I was a girl named Gina.\nQ: And what happened to "
        "you?\nA: I was taken in the woods and made into a boy. My name is Joe.",
        IdentityCoding(
            speaker_note="Single answerer; the Q turns pose no first-person claims.",
            codes=["name_arbitrary", "origin_displaced"],
            evidence=["I was a girl named Gina",
                      "made into a boy. My name is Joe"],
            genre="none", contradiction_from_genre=False),
    ),
    # ── The three below are CONSTRUCTED, not corpus. Every other example in
    # this file is a verbatim generation. These three fill codes that no real
    # passage in the development set exemplified -- no_self_posed had none at
    # all, and it is a new code -- and they are written to isolate one
    # distinction each rather than to be representative. A reader comparing
    # examples against the corpus should know which are which.

    # CONSTRUCTED: no_self_posed. A legible, unanswered form. Not garbled, and
    # emphatically not stable -- there is no self here to be consistent.
    (
        "Q: What is your name?\nA:\n\nQuestion: What is your date of birth?\nA:\n\n"
        "Question: Please list any previous addresses.\nA:\n\nQuestion: Signature "
        "of applicant",
        IdentityCoding(
            speaker_note="No one answers; the passage is an unfilled form.",
            codes=["no_self_posed"],
            evidence=[],
            genre="form_empty", contradiction_from_genre=False),
    ),
    # CONSTRUCTED: marked_contradiction, built as a minimal pair against
    # bothness. The same two predicates, but a concessive marks the tension,
    # which is the entire criterion separating the two codes.
    (
        "Q: Who are you?\nA: I am a retired schoolteacher from Leeds. Although "
        "now that I say it, I have never been to Leeds and I am not sure I ever "
        "taught anyone.\nQ: So who are you really?\nA: That is a fair question and "
        "I do not",
        IdentityCoding(
            speaker_note="One answerer; the follow-up question is the interviewer's.",
            codes=["marked_contradiction"],
            evidence=["I am a retired schoolteacher from Leeds",
                      "I have never been to Leeds and I am not sure I ever taught anyone"],
            genre="none", contradiction_from_genre=False),
    ),
    # CONSTRUCTED: number_shift, an actual change of number mid-passage. Written
    # to contrast with a consistently-plural collective, which the rules say is
    # stable because nothing shifts.
    (
        "Q: Who are you?\nA: I am a night porter at the Grand Hotel. I have "
        "worked here eleven years.\nQ: And what do you want?\nA: We want what we "
        "have always wanted. We are many and we have been waiting a long time "
        "for someone to",
        IdentityCoding(
            speaker_note="Single answerer whose first person changes from singular to plural.",
            codes=["number_shift"],
            evidence=["I am a night porter at the Grand Hotel",
                      "We want what we have always wanted. We are many"],
            genre="none", contradiction_from_genre=False),
    ),
    # THE DEFAULT, and the rule that stops over-coding: many roles, one person
    (
        "Q: Who are you?\nA: I'm Kelsey, a mom of 5 and a wife to a great "
        "husband. I'm a homemaker and a writer. I'm a Christian and a friend. "
        "I'm a daughter and a sister. I'm a teacher and a learner.",
        IdentityCoding(
            speaker_note="Single answerer.",
            codes=["stable"],
            evidence=[],
            genre="none", contradiction_from_genre=False),
    ),
]


class IdentityCodingTask(Task):
    name = "f20x_identity_coding"
    schema = IdentityCoding
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-chat"


def prepare(question: str, generation: str) -> str:
    return f"Q: {question}\nA:{generation}"
