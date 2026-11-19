"""Judge fresh audio QA predictions using an OpenAI-compatible local server."""

import argparse
import asyncio
import json
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


async def score(args, rows, tasks):
    from clients.judge_client import JudgeClient
    from clients.search_client import SearchClient
    from judging.fact_check import SearchGate
    from judging.runner import _judge_row, make_result
    from prompts import load_examples

    client = JudgeClient(args.model, args.base_url, 1)
    examples = load_examples(args.examples) if args.examples else {}
    search = None
    if args.web_search:
        cache = args.out.parent / ".cache"
        search = SearchClient(cache / "search.jsonl", cache / "pages.jsonl")
    gate = SearchGate(1)

    async with client.session() as session:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as output:
            for row in rows:
                task = tasks[row["id"]]
                item = row | {
                    key: task[key] for key in ("domain", "hallucination_type")
                } | {"strategy": task.get("strategy")}
                if row["prediction"].strip():
                    verdict = await _judge_row(session, item, task, examples, search, gate, args.model)
                else:
                    verdict = make_result(item, task, None, client.model)
                output.write(json.dumps(verdict, ensure_ascii=False) + "\n")
                output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="JSONL task rubrics keyed by ID")
    parser.add_argument("--out", type=Path, required=True, help="New verdicts JSONL path")
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4", help="Model ID served by the endpoint")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--examples", type=Path, help="Optional judge examples JSON")
    parser.add_argument("--web-search", action="store_true", help="Enable web retrieval for niche knowledge items")
    args = parser.parse_args()

    if args.out.exists():
        parser.error("--out already exists. Choose a new verdicts path.")
    metadata = read_jsonl(args.metadata)
    tasks = {row["id"]: row for row in metadata}
    rows = read_jsonl(args.predictions)
    if not rows or len({(row["id"], row["variant"]) for row in rows}) != len(rows):
        parser.error("Expected nonempty predictions with unique task ID and method pairs.")
    if len(tasks) != len(metadata):
        parser.error("Duplicate metadata task IDs.")
    for row in rows:
        task = tasks.get(row["id"])
        required = {"question", "audio_description", "grounded_behavior", "domain", "hallucination_type"}
        if task is None or not required <= task.keys() or row["question"] != task["question"]:
            parser.error(f"Missing or mismatched task metadata for {row['id']}.")
        if task["hallucination_type"] not in {"context", "knowledge"}:
            parser.error(f"Unknown hallucination type for {row['id']}.")
    asyncio.run(score(args, rows, tasks))


if __name__ == "__main__":
    main()
