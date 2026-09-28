"""Helpers for the OpenAI-compatible chat endpoint.

Open-webui connects to LDR as an OpenAI-compatible "model backend", so these
helpers translate between the OpenAI chat wire format and LDR's internal
question-answering entry point (``quick_summary``).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from ..utilities.url_utils import canonical_url_key

# Adjacent duplicate citation markers, e.g. ``[1], [1]`` or ``[1]、[1]``.
# After remapping collapses ``[11], [21]`` (same source) to ``[1], [1]``, this
# pass drops the redundant bracket plus its separator so open-webui — which
# renders only the first of several same-title citations — never sees an empty
# chip followed by a dangling comma.
_ADJACENT_DUP_RE = re.compile(
    r"(\[\d+(?:\s*,\s*\d+)*\])(?:\s*(?:,|，|、|-)\s*\1)+"
)


def last_user_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the trimmed content of the final non-empty ``user`` message.

    Returns ``None`` when the conversation carries no usable user turn.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _source_url(source: dict[str, Any]) -> str:
    return source.get("url") or source.get("link") or ""


def _source_title(source: dict[str, Any]) -> str:
    return source.get("title") or ""


def _source_index(source: dict[str, Any], position: int) -> int:
    """Return the 1-based citation index for ``source`` at ``position``."""
    index = source.get("index")
    if index is not None:
        try:
            return int(index)
        except (TypeError, ValueError):
            pass
    return position + 1


def _remap_bracket(match: re.Match, remap: dict[int, int]) -> str:
    """Remap and dedup the indices inside one ``[n, m]`` bracket."""
    parts = [p for p in re.split(r"\s*,\s*", match.group(1)) if p.isdigit()]
    new_indices: list[int] = []
    seen: set[int] = set()
    for part in parts:
        index = remap.get(int(part), int(part))
        if index not in seen:
            seen.add(index)
            new_indices.append(index)
    if not new_indices:
        return match.group(0)
    if len(new_indices) == 1:
        return f"[{new_indices[0]}]"
    return "[" + ", ".join(str(i) for i in new_indices) + "]"


def deduplicate_sources_and_remap(
    summary: str, sources: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Collapse duplicate sources and remap their ``[n]`` markers.

    The research pipeline can collect the same document across several search
    iterations, so ``sources`` may list one document at many indices while the
    answer text cites several of them (``[11], [21]``). open-webui dedups
    citations by title, which turns every duplicate-after-the-first marker into
    an empty chip and leaves the ``, `` separators dangling. Deduplicating here
    and remapping the markers to the single surviving index collapses
    ``[11], [21]`` to ``[1]`` before the answer reaches open-webui.

    Returns ``(remapped_summary, deduplicated_sources)``.
    """
    if not sources:
        return summary, []

    remap: dict[int, int] = {}
    key_to_new_index: dict[str, int] = {}
    deduped: list[dict[str, Any]] = []

    for position, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        old_index = _source_index(source, position)
        key = canonical_url_key(_source_url(source)) or _source_title(source)
        new_index = key_to_new_index.get(key)
        if new_index is None:
            new_index = len(deduped) + 1
            key_to_new_index[key] = new_index
            deduped.append(source)
        remap[old_index] = new_index

    if not isinstance(summary, str) or not summary:
        return summary, deduped

    # Pass 1: remap + dedup indices inside each bracket.
    summary = re.sub(
        r"\[(\d+(?:\s*,\s*\d+)*)\]",
        lambda m: _remap_bracket(m, remap),
        summary,
    )
    # Pass 2: collapse adjacent duplicate brackets, dropping their separators.
    summary = _ADJACENT_DUP_RE.sub(lambda m: m.group(1), summary)
    return summary, deduped


def filter_sources_to_cited(
    summary: str, sources: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Filter ``sources`` down to the subset whose ``[N]`` marker actually
    appears in ``summary``, then re-number the survivors to a contiguous
    1-based sequence and remap the markers in ``summary``.

    Pair with :func:`deduplicate_sources_and_remap` upstream — duplicate
    sources collapse to one entry first, then this pass drops every source
    the LLM never named so the citation panel stays aligned with the
    answer body.

    Behaviour:

    - When ``summary`` carries no ``[N]`` markers at all, returns
      ``(summary, [])`` (strict mode): the open-webui citation panel ends
      up empty instead of dangling extras, matching the body's silence.
    - ``[N]`` markers whose index is outside the source range are NOT
      counted as cited — the source list shrinks to whatever the
      in-range subset produces, and the out-of-range marker is preserved
      verbatim in the summary (it will render as an empty chip in the
      panel rather than silently rewriting the LLM's choice).
    - Composite brackets ``[1, 3, 7]`` are remapped to a contiguous
      ``[1, 2, 3]`` form once the un-referenced sources are filtered out.
    """
    if not isinstance(summary, str) or not summary:
        return summary or "", list(sources) if sources else []

    bracket_re = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
    cited_raw = [
        int(part)
        for match in bracket_re.finditer(summary)
        for part in re.split(r"\s*,\s*", match.group(1))
        if part.isdigit()
    ]
    if not cited_raw:
        return summary, []

    # First pass: assign a contiguous new index to every cited source, in
    # the original list order. ``cited_set`` doubles as a membership test
    # for the second pass so a source cited at the same index twice (e.g.
    # once by an ``[N]`` and once by a ``[N, N]`` composite) still gets
    # exactly one slot.
    remap: dict[int, int] = {}
    cited_set: set[int] = set()
    for position, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        old_index = _source_index(source, position)
        if old_index not in cited_set and old_index in cited_raw:
            cited_set.add(old_index)
            remap[old_index] = len(cited_set)

    # Second pass: drop un-cited sources, preserve order, and stamp each
    # survivor's ``index`` with its new position so downstream consumers
    # (citation panel, SSE annotations) see a contiguous 1-based numbering.
    new_sources: list[dict[str, Any]] = []
    for position, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        old_index = _source_index(source, position)
        if old_index in cited_set:
            new_source = dict(source)
            new_source["index"] = remap[old_index]
            new_sources.append(new_source)

    # Remap markers via the existing helper — it preserves out-of-range
    # indices (remap miss → original value) and re-emits composite
    # brackets cleanly.
    new_summary = bracket_re.sub(
        lambda m: _remap_bracket(m, remap), summary
    )

    return new_summary, new_sources


def sources_to_url_citations(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert LDR source dicts into open-webui ``url_citation`` annotations."""
    citations: list[dict[str, Any]] = []
    for source in sources:
        url = _source_url(source)
        if not url:
            continue
        title = _source_title(source) or url
        citations.append(
            {"type": "url_citation", "url_citation": {"url": url, "title": title}}
        )
    return citations


def chat_completion_stream(
    content: str, model: str, citations: list[dict[str, Any]]
) -> str:
    """Build an OpenAI-compatible SSE body as a pseudo-stream.

    Emits one content frame carrying the full answer plus its ``url_citation``
    annotations, a terminal ``finish_reason: "stop"`` frame, and the
    ``[DONE]`` sentinel. open-webui's streaming path reads ``url_citation``
    from ``delta.annotations``.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    frames: list[str] = []

    def emit(payload: dict[str, Any]) -> None:
        frames.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")

    delta: dict[str, Any] = {"content": content}
    if citations:
        delta["annotations"] = citations
    emit(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )
    emit(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def chat_completion_response(content: str, model: str) -> dict[str, Any]:
    """Build an OpenAI ``chat.completion`` response body for ``content``."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
