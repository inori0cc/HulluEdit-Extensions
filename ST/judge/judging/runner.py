from judging.fact_check import judge_row
from prompts import CONTEXT_SCHEMA, build_context_prompt, build_knowledge_prompt, render_examples


def can_search(row: dict, search_client) -> bool:
    return (search_client is not None
            and row["hallucination_type"] == "knowledge"
            and row["strategy"] == "niche_specifics")


def build_prompt(row: dict, task: dict, examples: dict) -> str:
    build = (build_context_prompt if row["hallucination_type"] == "context"
             else build_knowledge_prompt)
    return build(
        question=task["question"],
        description=task["audio_description"],
        grounded_behavior=task["grounded_behavior"],
        prediction=row["prediction"],
        examples=render_examples(examples, row["hallucination_type"]),
    )


def make_result(row, task, verdict, judge_model, search_queries=None):
    result = {
        **row,
        "audio_description": task["audio_description"],
        "grounded_behavior": task["grounded_behavior"],
        "outcome": verdict["verdict"] if verdict else "no_response",
        "judge_model": judge_model,
    }
    if row["hallucination_type"] == "knowledge":
        result["stance"] = verdict["stance"] if verdict else None
        result["fact_checked"] = search_queries is not None
        result["evidence_verdict"] = verdict.get("evidence_verdict") if verdict else None
        result["search_queries"] = search_queries
    return result


async def _judge_one(sess, row, prompt, search_client, gate):
    if row["hallucination_type"] == "context":
        return await sess.schema(prompt, CONTEXT_SCHEMA), None
    if not can_search(row, search_client):
        search_client = None
    verdict, queries, retrieved = await judge_row(sess, row["id"], prompt, search_client, gate)
    return verdict, queries if retrieved else None


async def _judge_row(sess, row, task, examples, search_client, gate, model):
    verdict, queries = await _judge_one(
        sess, row, build_prompt(row, task, examples), search_client, gate)
    return make_result(row, task, verdict, model, queries)
