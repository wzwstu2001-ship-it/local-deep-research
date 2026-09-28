"""Unit tests for the OpenAI-compat chat helpers."""

import json

from local_deep_research.web.openai_compat import (
    chat_completion_response,
    chat_completion_stream,
    deduplicate_sources_and_remap,
    filter_sources_to_cited,
    last_user_message,
    sources_to_url_citations,
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
    resp = chat_completion_response("an answer", model="ldr")
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "ldr"
    choices = resp["choices"]
    assert len(choices) == 1
    assert choices[0]["message"] == {"role": "assistant", "content": "an answer"}
    assert choices[0]["finish_reason"] == "stop"
    assert "usage" in resp


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
