import asyncio
import json
import re
from contextlib import asynccontextmanager

from openai import APIStatusError, AsyncOpenAI, OpenAIError

REQUEST_TIMEOUT = 300
RETRY_DELAYS = [2, 8, 30]
MAX_TOKENS = 1024
SEED = 42
TOOL_CHOICE = "auto"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
RAW_TOOL_CALL_RE = re.compile(r"<tool_call>|<function[ =]", re.I)


class TruncatedReply(RuntimeError):
    pass


class JudgeClient:
    def __init__(self, model: str, base_url: str, max_concurrency: int):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_concurrency = max_concurrency

    @asynccontextmanager
    async def session(self):
        async with AsyncOpenAI(
            base_url=self.base_url, api_key="EMPTY", timeout=REQUEST_TIMEOUT, max_retries=0
        ) as client:
            served = [m.id async for m in client.models.list()]
            if self.model not in served:
                raise RuntimeError(f"{self.base_url} serves {served}, expected {self.model}")
            yield _Session(self, client, asyncio.Semaphore(self.max_concurrency))

    def _base_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

    def _schema_payload(self, prompt: str, schema: dict) -> dict:
        return {
            **self._base_payload([{"role": "user", "content": prompt}]),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "judge_output", "strict": True, "schema": schema},
            },
        }

    def _tool_payload(self, messages, tools, tool_choice) -> dict:
        return {**self._base_payload(messages), "tools": tools, "tool_choice": tool_choice}

    async def _request(self, client, payload: dict, parse):
        for attempt, delay in enumerate([*RETRY_DELAYS, None]):
            payload["seed"] = SEED + attempt
            try:
                response = await client.chat.completions.create(**payload)
                try:
                    return parse(response, payload)
                except TruncatedReply as e:
                    error = e
                    payload["max_tokens"] = min(payload["max_tokens"] * 4, MAX_TOKENS * 16)
                except RuntimeError as e:
                    error = e
            except APIStatusError as e:
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    raise RuntimeError(f"Judge request failed ({e.status_code}): {e.response.text}") from e
                error = e
            except (OpenAIError, asyncio.TimeoutError, OSError) as e:
                error = e
            if delay is None:
                raise RuntimeError(f"Judge request failed after {attempt + 1} attempts: {error}") from error
            await asyncio.sleep(delay)

    @staticmethod
    def _parse(response, payload: dict) -> dict:
        schema = payload["response_format"]["json_schema"]["schema"]
        choice = response.choices[0]
        content = THINK_RE.sub("", choice.message.content or "").strip()
        try:
            verdict = json.loads(content)
        except json.JSONDecodeError as e:
            if choice.finish_reason == "length":
                raise TruncatedReply(f"Judge hit the token ceiling: {content[-300:]!r}") from e
            raise RuntimeError(f"Judge returned non-JSON output: {content!r}") from e
        missing = set(schema["required"]) - set(verdict)
        if missing:
            if choice.finish_reason == "length":
                raise TruncatedReply(f"Judge hit the token ceiling (missing {missing})")
            raise RuntimeError(f"Malformed judge output (missing {missing}): {verdict!r}")
        return verdict

    @staticmethod
    def _parse_calls(response, payload: dict) -> list[dict]:
        choice = response.choices[0]
        message = choice.message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                if choice.finish_reason == "length":
                    raise TruncatedReply(f"Judge hit the token ceiling mid tool call {call.function.name!r}") from e
                raise RuntimeError(f"Unparseable tool arguments: {call.function.arguments!r}") from e
            calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
        content = THINK_RE.sub("", message.content or "").strip()
        if not calls and RAW_TOOL_CALL_RE.search(content):
            raise RuntimeError(
                "Judge emitted a tool call as plain text; serve it with "
                "--enable-auto-tool-choice --tool-call-parser qwen3_coder"
            )
        if choice.finish_reason == "length":
            raise TruncatedReply("Judge hit the token ceiling before a usable tool call")
        return calls


class _Session:
    def __init__(self, parent: JudgeClient, client, semaphore):
        self._parent = parent
        self._client = client
        self._semaphore = semaphore

    async def _run(self, payload, parse):
        async with self._semaphore:
            return await self._parent._request(self._client, payload, parse)

    async def schema(self, prompt: str, schema: dict) -> dict:
        return await self._run(self._parent._schema_payload(prompt, schema), self._parent._parse)

    async def tools(self, messages, tools, tool_choice) -> list[dict]:
        return await self._run(
            self._parent._tool_payload(messages, tools, tool_choice), self._parent._parse_calls
        )
