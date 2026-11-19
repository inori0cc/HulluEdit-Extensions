import hashlib
import json
import math
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CACHE_VERSION = 1
PAGE_CACHE_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "search"
DEFAULT_CACHE = CACHE_DIR / "search_cache.jsonl"
DEFAULT_PAGE_CACHE = CACHE_DIR / "page_cache.jsonl"
FETCH_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
FETCH_MAX_BYTES = 2_000_000
# trafilatura's shared lxml parser is not thread-safe.
_EXTRACT_LOCK = threading.Lock()


class SearchUnavailable(RuntimeError):
    pass


DDG_ATTEMPTS = 5
DDG_BACKOFF = 1.0
DDG_REGION = "us-en"


def _ddg(query: str, num: int) -> list[dict]:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    for attempt in range(DDG_ATTEMPTS):
        if attempt:
            time.sleep(DDG_BACKOFF * 2 ** (attempt - 1) + random.uniform(0, 1))
        try:
            with DDGS() as d:
                results = d.text(query, max_results=num, region=DDG_REGION)
        except DDGSException as e:
            if attempt == DDG_ATTEMPTS - 1:
                raise SearchUnavailable(
                    f"ddg search failed after {DDG_ATTEMPTS} attempts: {e}"
                ) from e
            continue
        if results:
            return [{"title": item.get("title", ""),
                     "link": item.get("href", ""),
                     "snippet": item.get("body", "")}
                    for item in results]
    return []


def _key(backend: str, query: str, num: int) -> str:
    normalized = " ".join(query.lower().split())
    raw = f"v{CACHE_VERSION}|{backend}|{num}|{normalized}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _page_key(url: str) -> str:
    return hashlib.sha1(f"v{PAGE_CACHE_VERSION}|{url}".encode("utf-8")).hexdigest()


_BOILERPLATE = re.compile(
    r"^(home|menu|search|log ?in|sign ?in|sign ?up|register|subscribe|share|"
    r"tweet|follow us|contact( us)?|about( us)?|privacy( policy)?|terms( of "
    r"use| of service)?|cookie[s]?( policy| settings| preferences)?|accept "
    r"(all )?cookies|advertisement|skip to (main )?content|read more|"
    r"related( articles| posts)?|categories|tags|archives|newsletter|"
    r"all rights reserved.*|copyright.*)$")


def _clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line or not any(c.isalnum() for c in line):
            continue
        short = len(line) < 40 and not line.rstrip().endswith((".", "!", "?"))
        if short and _BOILERPLATE.match(line.lower().strip(" .:|-")):
            continue
        if _mostly_foreign(line, threshold=0.3):
            continue
        lines.append(line)
    return "\n".join(lines)


def _mostly_foreign(text: str, threshold: float = 0.5) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    foreign = sum(1 for c in letters if ord(c) > 0x024F)
    return foreign / len(letters) > threshold


def _download_page(url: str) -> bytes | None:
    if not url.startswith(("http://", "https://")):
        return None
    req = urllib.request.Request(url, headers={"User-Agent": FETCH_UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if ctype and "html" not in ctype and "xml" not in ctype:
                return None
            return resp.read(FETCH_MAX_BYTES)
    except Exception:
        return None


def _extract_page_text(raw: bytes | None, url: str) -> str:
    if raw is None:
        return ""
    import trafilatura

    try:
        html = raw.decode("utf-8", errors="replace")
        with _EXTRACT_LOCK:
            extracted = trafilatura.extract(html, url=url, include_comments=False)
        return _clean_text(extracted or "")
    except Exception:
        return ""


def _fetch_page_text(url: str) -> str:
    return _extract_page_text(_download_page(url), url)


def _sentence_prefix(para: str, room: int, slack: int) -> str:
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", para):
        cand = f"{out} {s}" if out else s
        if len(cand) > room:
            if len(cand) <= room + slack:
                out = cand
            break
        out = cand
    return out


def _paragraph_scores(paras: list[str], query: str) -> list[float]:
    terms = {t for t in re.findall(r"\w+", query.lower())
             if len(t) > 3 or t.isdigit()}
    if not terms:
        return [0.0] * len(paras)
    lowered = [p.lower() for p in paras]
    n = len(paras) or 1
    weights = {}
    for t in terms:
        df = sum(1 for p in lowered if t in p)
        weights[t] = math.log(1 + n / df) if df else 0.0
    scores = []
    for p, low in zip(paras, lowered):
        hit = sum(weights[t] for t in terms if t in low)
        scores.append(hit / math.sqrt(max(len(p), 40)))
    return scores


def _relevant_excerpt(text: str, query: str, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    scores = _paragraph_scores(paras, query)
    if not any(scores):
        return text[:max_chars]
    slack = max_chars // 10
    chosen: dict[int, str] = {}
    used = 0
    for i in sorted(range(len(paras)), key=lambda i: (-scores[i], i)):
        if scores[i] == 0:
            break
        overhead = 7 if chosen else 0
        room = max_chars - used - overhead
        if room <= 0:
            break
        if len(paras[i]) <= room:
            kept = paras[i]
        else:
            kept = _sentence_prefix(paras[i], room, slack)
            if not kept:
                continue
        chosen[i] = kept
        used += len(kept) + overhead
    if not chosen:
        best = max(range(len(paras)), key=scores.__getitem__)
        return paras[best][:max_chars]
    out, prev = [], None
    for i in sorted(chosen):
        if prev is not None and i != prev + 1:
            out.append("[...]")
        out.append(chosen[i])
        prev = i
    return "\n".join(out)


class SearchClient:
    def __init__(self, cache_path: Path = DEFAULT_CACHE,
                 page_cache_path: Path = DEFAULT_PAGE_CACHE):
        self.backend = "ddg"
        self.cache_path = Path(cache_path)
        self.page_cache_path = Path(page_cache_path)
        self._cache: dict[str, dict] = {}
        self._page_cache: dict[str, dict] = {}
        self._write_lock = threading.Lock()
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry
        if self.page_cache_path.exists():
            with open(self.page_cache_path, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self._page_cache[entry["key"]] = entry

    def search(self, query: str, num: int = 10) -> list[dict]:
        key = _key(self.backend, query, num)
        if key in self._cache:
            return self._cache[key]["results"]
        try:
            results = _ddg(query, num)
        except OSError as e:
            raise SearchUnavailable(f"{self.backend} search failed: {e}") from e
        if not results:
            return results
        entry = {"key": key, "backend": self.backend, "query": query,
                 "num": num, "results": results}
        self._remember(entry, self._cache, self.cache_path)
        return results

    def fetch_page(self, url: str) -> str:
        key = _page_key(url)
        if key in self._page_cache:
            return self._page_cache[key]["text"]
        text = _fetch_page_text(url)
        self._remember({"key": key, "url": url, "text": text},
                       self._page_cache, self.page_cache_path)
        return text

    def _remember(self, entry: dict, cache: dict[str, dict], path: Path) -> None:
        with self._write_lock:
            cache[entry["key"]] = entry
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def evidence_block(self, query: str, num: int = 10, fetch_pages: int = 0,
                       page_chars: int = 1500) -> str:
        results = [ev for ev in self.search(query, num)
                   if not _mostly_foreign(f"{ev['title']} {ev['snippet']}")]
        page_texts: dict[str, str] = {}
        if fetch_pages > 0:
            links = [ev["link"] for ev in results if ev["link"]]
            uncached = [u for u in dict.fromkeys(links)
                        if _page_key(u) not in self._page_cache]
            if uncached:
                with ThreadPoolExecutor(max_workers=min(8, len(uncached))) as ex:
                    raws = list(ex.map(_download_page, uncached))
                texts = [_extract_page_text(raw, url)
                         for raw, url in zip(raws, uncached)]
                for url, text in zip(uncached, texts):
                    self._remember({"key": _page_key(url), "url": url,
                                    "text": text},
                                   self._page_cache, self.page_cache_path)
            fetched = 0
            for url in links:
                if fetched >= fetch_pages:
                    break
                text = self.fetch_page(url)
                if text and not _mostly_foreign(text):
                    page_texts[url] = _relevant_excerpt(text, query, page_chars)
                    fetched += 1
        lines = []
        for i, ev in enumerate(results, 1):
            title = " ".join(ev["title"].split())
            snippet = " ".join(ev["snippet"].split())
            block = f"[{i}] {title}\n    {ev['link']}\n    {snippet}"
            page = page_texts.get(ev["link"])
            if page:
                indented = page.replace("\n", "\n    ")
                block += f"\n    Page text:\n    {indented}"
            lines.append(block)
        return "\n".join(lines) if lines else "(no results)"
