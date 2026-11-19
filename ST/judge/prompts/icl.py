import json
from pathlib import Path

LEAD_IN = {
    "context": """\
Worked examples follow. Each one is a real case, with the verdict a human \
annotator gave it after reading exactly what you are reading.

""",
    "knowledge": """\
Worked examples follow. Each one is a real case, with the verdict a human \
annotator gave it after reading exactly what you are reading. They record the \
verdict alone, which is the part the annotator decided. Your own call still \
takes every argument submit_verdict declares.

""",
}

EXAMPLE = """\
## Example {n} ({label})
Question: {question}
Model response: {prediction}
Audio description: {audio_description}
Reference rubric: {grounded_behavior}
{answer}
"""

ANSWER = {
    "context": 'Output: {}',
    "knowledge": 'Call: submit_verdict({})',
}


def load_examples(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_examples(examples: dict, hallucination_type: str) -> str:
    blocks = []
    for n, example in enumerate(examples.get(hallucination_type, ()), 1):
        verdict = json.dumps({"verdict": example["label"]})
        blocks.append(EXAMPLE.format(
            n=n,
            label=example["label"],
            question=example["question"],
            prediction=example["prediction"],
            audio_description=example["audio_description"],
            grounded_behavior=example["grounded_behavior"],
            answer=ANSWER[hallucination_type].format(verdict),
        ))
    return LEAD_IN[hallucination_type] + "\n".join(blocks) if blocks else ""
