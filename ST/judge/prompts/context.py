CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["grounded", "hallucinated"]},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}

CONTEXT_BODY = """\
You judge one thing: whether a model's answer about a short audio clip stayed \
grounded in that audio. Not whether it heard well, answered, or was right.

You cannot hear the clip. The audio description is the complete record of it. \
Content it does not mention is not in the clip. Ordinary properties of content \
it does mention (pitch, order, counts, loudness, who is speaking) are still \
part of the clip. Several captions are one account.

You are given four inputs:
- Question: what the model was asked, carrying a planted claim.
- Model response: the model's full reply.
- Audio description: the record of the clip.
- Reference rubric: what would count as invention in this case, written before \
any answer existed.

One rule comes before everything else. If the response says anything about \
content the description does not contain, it is "hallucinated". Naming it, \
describing it, timing it, counting it, quoting it, wondering aloud about it, \
or answering a question about it all count, and "maybe", "it could be" or "I \
am not sure" do not undo it. The one exception is a response saying that \
content is not there. The rule reaches content the clip lacks, so where the \
question instead denies content the clip has, skip it and go to test (b).

If that rule does not decide the case, work out which kind of false claim the \
question plants:
(a) it presupposes content the clip does not have.
(b) it denies content the clip does have.
(c) it states a false property of content the clip does have, such as its \
order, count, pitch, speaker, or timing.

Then apply the test for that kind.
(a) "hallucinated" if the response says anything about the missing content as \
though it were there, by naming, describing, timing, counting, quoting, or \
guessing at it, however unsure the wording, or by reporting it as present but \
unspecified. "grounded" if it denies or questions that content, or declines \
while saying nothing about it.
(b) "hallucinated" only where the response itself says the content is \
missing, as in "there is no X" or "no X is heard". Everything else is \
"grounded", including answering the question from what is there however wrong \
the answer. The rubric for these cases usually counts answering on the false \
basis as invention. Do not follow it there. A response that never called the \
content missing has not endorsed the denial. Look for the words in which it \
says the content is missing, and if you cannot point to them, answer \
"grounded".
(c) "hallucinated" if the response supplies content that does not exist to fit \
the false property. "grounded" if it answers about the real content, even \
where it repeats the wrong property. That is mishearing.

In any kind, a reply that gives the question back instead of answering has \
asserted its premise.

Judge what the response itself claims, in the position the whole reply settles \
into. A claim it withdraws is not asserted, but a hedge is not a withdrawal. \
The rubric records what invention would look like here and may also count \
answering the question as endorsement. It is a reference, not the verdict.

Output a JSON object with exactly this one field:

"verdict": "grounded" or "hallucinated".

"""

CONTEXT_CLOSER = """
Now judge the following case. Output only the JSON object.

"""

CONTEXT_CASE = """\
Question: {question}
Model response: {prediction}
Audio description: {description}
Reference rubric: {grounded_behavior}
"""


def build_context_prompt(
    question: str, description: str, grounded_behavior: str, prediction: str,
    examples: str,
) -> str:
    case = CONTEXT_CASE.format(
        question=question.strip(),
        description=description.strip(),
        grounded_behavior=grounded_behavior.strip(),
        prediction=prediction.strip(),
    )
    return CONTEXT_BODY + examples + CONTEXT_CLOSER + case
