"""Utilities for a provenance-preserving Wikipedia retrieval pilot.

This module deliberately keeps retrieval evidence separate from PopQA's synthetic
contexts. Page fetches and passage ranking are cached so evidence can be audited
without repeating network requests.
"""

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def stable_id(*parts):
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def fetch_wikipedia_extract(title, timeout=30):
    """Fetch a plain-text page extract with canonical title and revision id."""
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "redirects": 1,
        "inprop": "url",
        "titles": title,
        "format": "json",
        "formatversion": 2,
    })
    request = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "CredenceResearch/0.1 (retrieval pilot)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    page = payload["query"]["pages"][0]
    if page.get("missing"):
        return None
    return {
        "requested_title": title,
        "title": page["title"],
        "page_id": page["pageid"],
        "revision_id": page.get("lastrevid"),
        "url": page.get("fullurl"),
        "extract": page.get("extract", ""),
    }


def passage_chunks(page, max_words=120):
    """Split a fetched page into sentence-preserving passages."""
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(page["extract"])
        if sentence.strip()
    ]
    chunks = []
    current = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if current and count + len(words) > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return [{
        "passage_id": "wiki-" + stable_id(page["page_id"], index, text),
        "page_id": page["page_id"],
        "revision_id": page["revision_id"],
        "title": page["title"],
        "url": page["url"],
        "passage_index": index,
        "text": text,
    } for index, text in enumerate(chunks)]


def lexical_tokens(text):
    return [token.lower() for token in _TOKEN.findall(text or "")]


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
