from __future__ import annotations

import os
from pathlib import Path
import re
from typing import List
from urllib.parse import quote_plus

import requests
from memory_manager import manage_memories


def echo(text: str) -> str:
    return text


def read_memory_lines(prompt_file: str = "system_prompt.txt") -> List[str]:
    path = Path(prompt_file)
    if not path.exists():
        return [f"ERROR: file not found: {prompt_file}"]

    full_text = path.read_text(encoding="utf-8")
    end_marker = "===END_INITIAL_PROMPT==="
    pos = full_text.find(end_marker)
    initial = full_text if pos == -1 else full_text[:pos]

    return [line.strip() for line in initial.splitlines() if line.strip().startswith("<")]


def manage_memories_tool(
    api_key: str,
    prompt_file: str = "system_prompt.txt",
    model: str = "deepseek-chat",
    dry_run: bool = False,
) -> dict:
    return manage_memories(Path(prompt_file), api_key, model, dry_run)


def web_search(query: str, max_results: int = 5, api_key: str | None = None) -> List[dict]:
    """
    Search web results using DuckDuckGo (no API key).
    Returns a list of: {title, url, snippet}
    """
    q = (query or "").strip()
    if not q:
        return []

    max_results = max(1, min(int(max_results), 10))

    # First try Tavily if key is provided.
    tavily_key = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
    if tavily_key:
        try:
            tavily_resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": q,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                    "include_images": False,
                },
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            tavily_resp.raise_for_status()
            data = tavily_resp.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "snippet": str(item.get("content", "")).strip(),
                })
            if results:
                return results
        except Exception:
            pass

    # Fallback 1: DuckDuckGo Instant Answer JSON API.
    api_url = (
        "https://api.duckduckgo.com/"
        f"?q={quote_plus(q)}&format=json&no_redirect=1&no_html=1&skip_disambig=0"
    )
    headers = {"User-Agent": "Mozilla/5.0 (Theresia-MCP-WebSearch)"}
    try:
        api_resp = requests.get(api_url, headers=headers, timeout=15)
        api_resp.raise_for_status()
        data = api_resp.json()
        api_results: List[dict] = []

        def _collect_related(items):
            for item in items or []:
                if "Topics" in item:
                    _collect_related(item.get("Topics", []))
                    continue
                text = str(item.get("Text", "")).strip()
                first_url = str(item.get("FirstURL", "")).strip()
                if text and first_url:
                    api_results.append({
                        "title": text,
                        "url": first_url,
                        "snippet": "",
                    })

        _collect_related(data.get("RelatedTopics", []))

        abstract_text = str(data.get("AbstractText", "")).strip()
        abstract_url = str(data.get("AbstractURL", "")).strip()
        if abstract_text and abstract_url:
            api_results.insert(0, {
                "title": abstract_text,
                "url": abstract_url,
                "snippet": "",
            })

        if api_results:
            return api_results[:max_results]
    except Exception:
        pass

    # Fallback: DuckDuckGo lite HTML endpoint (simple and public).
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(q)}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text

    # Extract links and titles.
    # Example pattern:
    # <a rel="nofollow" href="...">Title</a>
    link_re = re.compile(r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.IGNORECASE | re.DOTALL)

    results: List[dict] = []
    for m in link_re.finditer(html):
        href = re.sub(r"\s+", " ", m.group("href")).strip()
        title_raw = m.group("title")
        title = re.sub(r"<.*?>", "", title_raw)
        title = re.sub(r"\s+", " ", title).strip()

        if not href or not title:
            continue
        if not (href.startswith("http://") or href.startswith("https://")):
            continue

        results.append({
            "title": title,
            "url": href,
            "snippet": "",
        })
        if len(results) >= max_results:
            break

    return results
