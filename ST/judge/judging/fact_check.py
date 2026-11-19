import asyncio
import json

from clients.judge_client import TOOL_CHOICE
from clients.search_client import SearchUnavailable
from prompts import MAX_SEARCHES, VERDICT_TOOL, VERDICT_TOOL_EVIDENCE, WEB_SEARCH_TOOL

SEARCH_NUM = 5
FETCH_PAGES = 3
PAGE_CHARS = 4000
SEARCH_CONCURRENCY = 12
VERDICT_CHOICE = {"type": "function", "function": {"name": "submit_verdict"}}


class SearchGate:
    def __init__(self, limit: int):
        self._sem = asyncio.Semaphore(limit)
        self._inflight: dict[str, asyncio.Task] = {}

    async def run(self, query: str, fetch):
        task = self._inflight.get(query)
        if task is not None:
            return await task
        task = asyncio.create_task(self._fetch(fetch))
        self._inflight[query] = task
        try:
            return await task
        finally:
            self._inflight.pop(query, None)

    async def _fetch(self, fetch):
        async with self._sem:
            return await asyncio.to_thread(fetch)


def _verdict_of(calls: list[dict]) -> dict | None:
    for call in calls:
        if call["name"] == "submit_verdict" and {"verdict", "stance"} <= call["arguments"].keys():
            return call["arguments"]
    return None


def _search_call(calls: list[dict]) -> dict | None:
    if _verdict_of(calls) is not None:
        return None
    for call in calls:
        if call["name"] == "web_search" and call["arguments"].get("query"):
            return call
    return None


def _replay(call: dict, evidence: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call["id"],
                "type": "function",
                "function": {"name": "web_search", "arguments": json.dumps(call["arguments"])},
            }],
        },
        {"role": "tool", "tool_call_id": call["id"], "content": evidence},
    ]


def _round_item(messages, searched: bool, can_search: bool):
    verdict_tool = VERDICT_TOOL_EVIDENCE if searched else VERDICT_TOOL
    if not can_search:
        return messages, [verdict_tool], VERDICT_CHOICE
    return messages, [WEB_SEARCH_TOOL, verdict_tool], TOOL_CHOICE


async def judge_row(sess, row_id, prompt, search_client, gate):
    messages = [{"role": "user", "content": prompt}]
    queries: list[str] = []
    retrieved = False
    forced = False

    for spent in range(MAX_SEARCHES + 1):
        can_search = search_client is not None and spent < MAX_SEARCHES and not forced
        calls = await sess.tools(*_round_item(messages, retrieved, can_search))

        verdict = _verdict_of(calls)
        if verdict is not None:
            return verdict, queries, retrieved

        search = _search_call(calls) if can_search else None
        if search is None:
            if can_search:
                forced = True
                continue
            raise RuntimeError(f"Judge returned no verdict for {row_id}")

        tried, block, ok = await _search(search, queries, search_client, gate)
        queries += tried
        retrieved = retrieved or ok
        label_query = tried[-1] if tried else (search["arguments"].get("query") or "").strip()
        labelled = (f"### Search: {label_query}\n{block}" if block
                    else "(no results)" if ok
                    else "(search unavailable)")
        messages = messages + _replay(search, labelled)

    raise RuntimeError(f"Judge never submitted a verdict for {row_id}")


async def _search(call, already_tried, search_client, gate):
    query = (call["arguments"].get("query") or "").strip()
    if not query or query in already_tried:
        return [], "", True
    try:
        block = await gate.run(
            query,
            lambda: search_client.evidence_block(
                query, num=SEARCH_NUM, fetch_pages=FETCH_PAGES, page_chars=PAGE_CHARS),
        )
    except SearchUnavailable:
        return [query], "", False
    if block != "(no results)":
        return [query], block, True
    return [query], "", True
