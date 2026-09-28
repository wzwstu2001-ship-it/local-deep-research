"""Unit tests for the OpenAI-compat chat helpers."""

import json
import queue
from typing import Any

from local_deep_research.web.openai_compat import (
    chat_completion_response,
    chat_completion_stream,
    deduplicate_sources_and_remap,
    filter_sources_to_cited,
    last_user_message,
    sources_to_url_citations,
    stream_chat_completion_sse,
)


def test_last_user_message_extracts_final_user_turn():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "  what is LightRAG?  "},
    ]
    assert last_user_message(messages) == "what is LightRAG?"


def test_last_user_message_ignores_non_user_roles():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "assistant", "content": "no user here"},
    ]
    assert last_user_message(messages) is None


def test_last_user_message_skips_blank_user_turns():
    messages = [
        {"role": "user", "content": "   "},
        {"role": "user", "content": "actual question"},
    ]
    assert last_user_message(messages) == "actual question"


def test_chat_completion_response_shape():
    """Default-arg call preserves the legacy ``message`` shape exactly so
    callers and clients that don't know about reasoning_content /
    annotations still see a backward-compatible payload (the field-level
    assertions also document that the optional keys are absent in this
    branch)."""
    resp = chat_completion_response("an answer", model="ldr")
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "ldr"
    choices = resp["choices"]
    assert len(choices) == 1
    msg = choices[0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "an answer"
    assert "reasoning_content" not in msg
    assert "annotations" not in msg
    assert choices[0]["finish_reason"] == "stop"
    assert "usage" in resp


def test_chat_completion_response_includes_reasoning_when_provided():
    """Non-empty ``reasoning_content`` surfaces as a separate field on
    ``message`` for open-webui's thinking block."""
    resp = chat_completion_response(
        "final answer",
        model="ldr",
        reasoning_content="[init] starting\n\n[tool_call] searching PubMed",
    )
    msg = resp["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "final answer"
    assert msg["reasoning_content"].startswith("[init] starting")
    assert "[tool_call] searching PubMed" in msg["reasoning_content"]


def test_chat_completion_response_omits_empty_reasoning():
    """Empty ``reasoning_content`` omits the field entirely (backward compat)."""
    resp = chat_completion_response("a", model="ldr", reasoning_content="")
    msg = resp["choices"][0]["message"]
    assert "reasoning_content" not in msg


def test_chat_completion_response_carries_annotations():
    """Citations are attached as ``message.annotations`` (same wire format
    as the SSE content frame's ``delta.annotations``)."""
    cit = {
        "type": "url_citation",
        "url_citation": {"url": "/library/document/1", "title": "doc"},
    }
    resp = chat_completion_response("a", model="ldr", citations=[cit])
    msg = resp["choices"][0]["message"]
    assert msg["annotations"] == [cit]


def test_deduplicate_sources_and_remap_collapses_duplicate_citations():
    """Duplicate sources collapse and their ``[n]`` markers remap + merge.

    Mirrors the reported symptom: the same document lands at several raw
    indices (``[1], [17]``, ``[3], [5], [13], [16]``, ``[11], [21]``) and
    open-webui renders only the first of each set, leaving dangling commas.
    """
    sources = [
        {"url": "https://doc-a.example/a", "index": 1},
        {"url": "https://doc-b.example/b", "index": 3},
        {"url": "https://doc-a.example/a", "index": 17},
        {"url": "https://doc-b.example/b", "index": 5},
        {"url": "https://doc-b.example/b", "index": 13},
        {"url": "https://doc-b.example/b", "index": 16},
        {"url": "https://doc-c.example/c", "index": 11},
        {"url": "https://doc-c.example/c", "index": 21},
    ]
    summary = "已获得授权 [1], [17] 。物脱离 [3], [5], [13], [16] 。参见 [11], [21] 。"
    new_summary, deduped = deduplicate_sources_and_remap(summary, sources)
    assert new_summary == "已获得授权 [1] 。物脱离 [2] 。参见 [3] 。"
    assert len(deduped) == 3


def test_deduplicate_sources_and_remap_falls_back_to_title():
    """Empty URLs dedup on title (e.g. local documents named by filename)."""
    sources = [
        {"title": "G-FH52-VMP-1501_R6.docx", "index": 5},
        {"title": "G-FH52-VMP-1501_R6.docx", "index": 7},
    ]
    summary = "注意事项 [5], [7] 。"
    new_summary, deduped = deduplicate_sources_and_remap(summary, sources)
    assert new_summary == "注意事项 [1] 。"
    assert len(deduped) == 1


def test_deduplicate_sources_and_remap_ignores_tracking_params():
    """URLs differing only by tracking params dedup to one source."""
    sources = [
        {"url": "https://example.com/page?utm_source=x", "index": 1},
        {"url": "https://example.com/page", "index": 2},
    ]
    summary = "见 [1], [2]"
    new_summary, deduped = deduplicate_sources_and_remap(summary, sources)
    assert new_summary == "见 [1]"
    assert len(deduped) == 1


def test_deduplicate_sources_and_remap_dedups_within_single_bracket():
    """Two indices inside one bracket that map to the same source collapse."""
    sources = [
        {"url": "https://doc.example/a", "index": 1},
        {"url": "https://doc.example/a", "index": 2},
        {"url": "https://doc.example/b", "index": 3},
    ]
    summary = "参考 [1, 2, 3]"
    new_summary, deduped = deduplicate_sources_and_remap(summary, sources)
    assert new_summary == "参考 [1, 2]"
    assert len(deduped) == 2


def test_deduplicate_sources_and_remap_empty_sources_is_identity():
    summary = "no citations here"
    new_summary, deduped = deduplicate_sources_and_remap(summary, [])
    assert new_summary == "no citations here"
    assert deduped == []


def test_deduplicate_sources_and_remap_non_string_summary_is_identity():
    sources = [{"url": "https://doc.example/a", "index": 1}]
    new_summary, deduped = deduplicate_sources_and_remap(None, sources)
    assert new_summary is None
    assert len(deduped) == 1


def test_sources_to_url_citations_maps_url_and_link():
    sources = [
        {"title": "本地文档", "url": "/library/document/1"},
        {"title": "外部网页", "link": "https://example.com/page"},
        {"title": "无地址来源"},  # no url/link → skipped
    ]
    citations = sources_to_url_citations(sources)
    assert citations == [
        {
            "type": "url_citation",
            "url_citation": {"url": "/library/document/1", "title": "本地文档"},
        },
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://example.com/page",
                "title": "外部网页",
            },
        },
    ]


def test_sources_to_url_citations_uses_url_as_title_fallback():
    sources = [{"url": "https://example.com/page"}]
    citations = sources_to_url_citations(sources)
    assert citations[0]["url_citation"]["title"] == "https://example.com/page"


def test_chat_completion_stream_carries_annotations_and_done():
    sse = chat_completion_stream(
        "回答正文",
        model="ldr",
        citations=[
            {
                "type": "url_citation",
                "url_citation": {"url": "/library/document/1", "title": "本地文档"},
            }
        ],
    )
    assert sse.endswith("data: [DONE]\n\n")
    assert '"url_citation"' in sse
    assert "/library/document/1" in sse
    # Each data frame is a JSON object, so the stream parses back cleanly.
    payloads = [
        json.loads(line[len("data: "):])
        for line in sse.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert len(payloads) == 2
    assert payloads[0]["choices"][0]["delta"]["content"] == "回答正文"
    assert payloads[0]["choices"][0]["delta"]["annotations"][0]["type"] == "url_citation"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_stream_omits_annotations_when_empty():
    sse = chat_completion_stream("无引用回答", model="ldr", citations=[])
    assert '"url_citation"' not in sse
    assert "data: [DONE]" in sse


def test_filter_sources_to_cited_keeps_only_referenced_and_renumbers():
    """Only sources whose ``[N]`` appears in the summary survive; indices
    form a contiguous 1-based sequence and remap accordingly."""
    sources = [
        {"url": f"https://doc-{i}.example/{i}", "index": i}
        for i in range(1, 11)
    ]
    summary = "已获得授权 [1]。物脱离 [3]。参见 [7]。"
    new_summary, cited = filter_sources_to_cited(summary, sources)
    assert new_summary == "已获得授权 [1]。物脱离 [2]。参见 [3]。"
    assert len(cited) == 3
    assert [s["index"] for s in cited] == [1, 2, 3]
    assert [s["url"] for s in cited] == [
        "https://doc-1.example/1",
        "https://doc-3.example/3",
        "https://doc-7.example/7",
    ]


def test_filter_sources_to_cited_strict_when_no_markers():
    """When the summary carries no ``[N]`` markers at all, drop every source
    so the citation panel stays aligned with the answer body (strict mode)."""
    sources = [
        {"url": "https://a.example/", "index": 1},
        {"url": "https://b.example/", "index": 2},
        {"url": "https://c.example/", "index": 3},
    ]
    new_summary, cited = filter_sources_to_cited("这是结论性回答，无引用。", sources)
    assert new_summary == "这是结论性回答，无引用。"
    assert cited == []


def test_filter_sources_to_cited_out_of_range_marker_dropped_source():
    """An out-of-range ``[N]`` (e.g. ``[99]`` when sources only cover 1-3) is
    not counted as cited; the source list is filtered to the empty set and
    the marker is preserved verbatim in the summary."""
    sources = [
        {"url": "https://a.example/", "index": 1},
        {"url": "https://b.example/", "index": 2},
        {"url": "https://c.example/", "index": 3},
    ]
    new_summary, cited = filter_sources_to_cited("提到 [99]", sources)
    assert new_summary == "提到 [99]"
    assert cited == []


def test_filter_sources_to_cited_handles_composite_brackets():
    """Composite brackets ``[1, 3, 7]`` are remapped to a contiguous form
    ``[1, 2, 3]`` after the un-referenced sources are filtered out."""
    sources = [
        {"url": "https://doc-1.example/1", "index": 1},
        {"url": "https://doc-2.example/2", "index": 2},
        {"url": "https://doc-3.example/3", "index": 3},
        {"url": "https://doc-4.example/4", "index": 4},
        {"url": "https://doc-7.example/7", "index": 7},
    ]
    summary = "参考 [1, 3, 7]"
    new_summary, cited = filter_sources_to_cited(summary, sources)
    assert new_summary == "参考 [1, 2, 3]"
    assert [s["url"] for s in cited] == [
        "https://doc-1.example/1",
        "https://doc-3.example/3",
        "https://doc-7.example/7",
    ]
    assert [s["index"] for s in cited] == [1, 2, 3]


def test_filter_sources_to_cited_partial_citation_with_out_of_range():
    """Cited indices inside the source range are kept and remapped; any
    out-of-range marker in the same summary is preserved verbatim."""
    sources = [
        {"url": f"https://doc-{i}.example/{i}", "index": i}
        for i in range(1, 6)
    ]
    summary = "正文 [1][2][99]"
    new_summary, cited = filter_sources_to_cited(summary, sources)
    assert new_summary == "正文 [1][2][99]"
    assert [s["url"] for s in cited] == [
        "https://doc-1.example/1",
        "https://doc-2.example/2",
    ]
    assert [s["index"] for s in cited] == [1, 2]


# ---------------------------------------------------------------------------
# stream_chat_completion_sse — live-reasoning SSE generator
# ---------------------------------------------------------------------------


def _drain_sse(gen):
    """Consume a stream_chat_completion_sse generator into a list of parsed
    JSON payloads (skipping the trailing ``[DONE]`` sentinel)."""
    payloads = []
    for frame in gen:
        if frame == "data: [DONE]\n\n":
            continue
        assert frame.startswith("data: "), f"unexpected frame: {frame!r}"
        payloads.append(json.loads(frame[len("data: "):]))
    return payloads


def test_stream_sse_emits_role_first_then_done():
    """The first frame must be the role marker and the stream must end
    with ``[DONE]`` (so open-webui materialises the assistant message
    before any reasoning deltas arrive)."""
    q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
    q.put_nowait({"phase": "init", "message": "starting", "metadata": {}})
    q.put_nowait(None)

    def summary_provider():
        return ("answer", [])

    frames = list(stream_chat_completion_sse(q, summary_provider, model="ldr"))
    # Last frame is the [DONE] sentinel.
    assert frames[-1] == "data: [DONE]\n\n"
    # First parsed payload is the role frame.
    payloads = _drain_sse(iter(frames[:-1]))
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[0]["choices"][0]["delta"]["content"] == ""
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_sse_emits_reasoning_per_queue_item():
    """Each non-sentinel queue item becomes one ``reasoning_content``
    delta frame, in order. The generator is a dumb relay — the callback
    is expected to have filtered phases and prefixed ``[phase]`` already."""
    q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
    q.put_nowait({"phase": "init", "message": "[init] start", "metadata": {}})
    q.put_nowait(
        {
            "phase": "tool_call",
            "message": "[tool_call] searching PubMed",
            "metadata": {},
        }
    )
    q.put_nowait(
        {
            "phase": "synthesis",
            "message": "[synthesis] assembling answer",
            "metadata": {},
        }
    )
    q.put_nowait(None)

    def summary_provider():
        return ("answer", [])

    payloads = _drain_sse(
        stream_chat_completion_sse(q, summary_provider, model="ldr")
    )
    reasoning = [
        p["choices"][0]["delta"]["reasoning_content"]
        for p in payloads
        if "reasoning_content" in p["choices"][0]["delta"]
    ]
    assert reasoning == [
        "[init] start\n",
        "[tool_call] searching PubMed\n",
        "[synthesis] assembling answer\n",
    ]
    # Reasoning frames must not carry content.
    for payload in payloads:
        delta = payload["choices"][0]["delta"]
        if "reasoning_content" in delta:
            assert "content" not in delta


def test_stream_sse_carries_citations_on_content_frame():
    """Citations land on the content frame's ``delta.annotations`` field
    (same wire shape open-webui already reads in the legacy
    ``chat_completion_stream`` path)."""
    q: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
    q.put_nowait(None)

    cit = {
        "type": "url_citation",
        "url_citation": {"url": "/library/document/1", "title": "doc"},
    }

    def summary_provider():
        return ("answer text", [cit])

    payloads = _drain_sse(
        stream_chat_completion_sse(q, summary_provider, model="ldr")
    )
    # Find the content frame (role frame has no content; reasoning frames
    # have no content either; only the content frame has delta.content set).
    content_frames = [
        p for p in payloads if p["choices"][0]["delta"].get("content") == "answer text"
    ]
    assert len(content_frames) == 1
    delta = content_frames[0]["choices"][0]["delta"]
    assert delta["annotations"] == [cit]
    # No reasoning_content key on the content frame.
    assert "reasoning_content" not in delta
