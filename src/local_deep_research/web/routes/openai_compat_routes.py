"""OpenAI-compatible chat endpoint for open-webui.

Open-webui connects to LDR as an OpenAI-compatible model backend. This
blueprint exposes ``/v1/chat/completions``, which folds an OpenAI chat
request into a single LDR ``quick_summary`` call (the agentic
research/retrieval path) and returns an OpenAI ``chat.completion`` body.
"""

from __future__ import annotations

import queue
import threading

from flask import Blueprint, Response, jsonify, request, stream_with_context
from loguru import logger

from ...security.decorators import require_json_body
from ...security.rate_limiter import api_rate_limit
from ..api import (
    _load_user_context_into_params,
    _scrub_error_fields,
    get_openai_compat_username,
)
from ..openai_compat import (
    _MAX_REASONING_CHUNK_CHARS,
    _REASONING_PHASES,
    chat_completion_response,
    deduplicate_sources_and_remap,
    filter_sources_to_cited,
    last_user_message,
    sources_to_url_citations,
    stream_chat_completion_sse,
)

openai_compat_bp = Blueprint("openai_compat", __name__, url_prefix="/v1")


@openai_compat_bp.route("/models", methods=["GET"])
def list_models():
    """OpenAI-compatible model list so open-webui can enumerate LDR as a backend.

    open-webui fetches ``GET /v1/models`` when adding/refreshing an OpenAI
    connection. Without it the model picker shows "no models", so expose the
    single logical model ``ldr`` (the id ``chat_completions`` accepts).
    """
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": "ldr",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-deep-research",
                }
            ],
        }
    )


@openai_compat_bp.route("/chat/completions", methods=["POST"])
@api_rate_limit
@require_json_body(error_message="messages are required")
def chat_completions():
    data = request.json or {}
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return (
            jsonify({"error": {"message": "messages must be a non-empty list"}}),
            400,
        )

    query = last_user_message(messages)
    if query is None:
        return (
            jsonify({"error": {"message": "no non-empty user message found"}}),
            400,
        )

    model = data.get("model") or "ldr"

    try:
        # Import here to avoid the research-stack import cycle, matching the
        # existing /api/v1/quick_summary handler.
        from ...api.research_functions import build_streaming_search_system

        # Session isolation (spec §9, fail-closed): open-webui forwards the
        # conversation id as chat_id (accept a few spellings). Missing chat_id
        # is a 400 — a headless call must never silently search the library.
        chat_id = (
            data.get("chat_id")
            or data.get("session_id")
            or (data.get("metadata") or {}).get("chat_id")
        )
        if not chat_id:
            return jsonify({"error": {"message": "chat_id is required"}}), 400

        username = get_openai_compat_username()
        params = {"temperature": data.get("temperature", 0.7)}
        # Open mode: the fixed service user has no Flask session/password, so
        # its encrypted settings DB can't be opened server-to-server. Fall back
        # to permissive defaults instead of failing closed (LDR binds 127.0.0.1).
        error = _load_user_context_into_params(
            params, username, allow_default_settings=True
        )
        if error is not None:
            return error

        # Resolve (creating if needed) the chat's dedicated collection and
        # scope retrieval to it. Fail closed: an unresolvable chat_id is a 400.
        from ...chat.service import ChatService

        try:
            collection_id = ChatService(username).get_or_create_session_collection(
                chat_id
            )
        except Exception:
            logger.exception("Failed to resolve chat collection for chat_id")
            return (
                jsonify({"error": {"message": "chat_id could not be resolved"}}),
                400,
            )

        # Guard against an absent/non-dict snapshot: the real helper always
        # sets a dict, but a mocked or legacy caller path may not.
        settings_snapshot = params.setdefault("settings_snapshot", {})
        if not isinstance(settings_snapshot, dict):
            settings_snapshot = {}
            params["settings_snapshot"] = settings_snapshot
        settings_snapshot["_session_collection_id"] = collection_id
        # Session isolation is enforced by _session_collection_id alone: the
        # library engine filters by it and the agent hides every OTHER
        # collection tool. When the chat's collection already has an index
        # (uploaded documents were embedded), promote it to the run's primary
        # search engine so the agent's web_search tool retrieves the uploads
        # instead of relying on the LLM to pick a UUID-named collection tool.
        # The egress policy keeps public engines (searxng) alongside this
        # private primary for session runs (context_from_snapshot promotion),
        # so mixed retrieval still works. An empty collection is left as-is so
        # general questions fall back to searxng rather than "No sources".
        if ChatService(username).session_collection_has_index(collection_id):
            settings_snapshot["search.tool"] = f"collection_{collection_id}"

        # Both branches share the same phase-filter + format rule; only the
        # sink differs (queue push for SSE vs. list append for the
        # non-stream message body). Keeping the formatter as a closure means
        # adding a new phase is a one-line change in _REASONING_PHASES.
        def _format_reasoning(metadata, message: str) -> str | None:
            phase = (metadata or {}).get("phase", "")
            if phase not in _REASONING_PHASES:
                return None
            clipped = (message or "")[:_MAX_REASONING_CHUNK_CHARS]
            return f"[{phase}] {clipped}"

        if data.get("stream"):
            reasoning_queue: queue.Queue = queue.Queue(maxsize=1024)

            def _push_progress(message: str, percent, metadata) -> None:
                line = _format_reasoning(metadata, message)
                if line is None:
                    return
                item = {
                    "phase": (metadata or {}).get("phase", ""),
                    "message": line,
                    "metadata": metadata or {},
                }
                try:
                    reasoning_queue.put_nowait(item)
                except queue.Full:
                    # Back-pressure / client disconnect: silently drop.
                    pass

            _, run_fn = build_streaming_search_system(
                query,
                progress_callback=_push_progress,
                **params,
            )
            result_holder: dict = {}
            error_holder: dict = {"err": None}

            def _worker() -> None:
                try:
                    result_holder["result"] = run_fn()
                except BaseException as exc:  # noqa: BLE001
                    error_holder["err"] = exc
                finally:
                    # Sentinel — unblocks the SSE generator's queue loop
                    # even when the worker raised before producing a result.
                    reasoning_queue.put_nowait(None)

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            def _summary_provider() -> tuple[str, list[dict]]:
                thread.join()
                err = error_holder["err"]
                if err is not None:
                    # The worker's exception must propagate into the SSE
                    # generator's caller — the route's outer try/except
                    # converts it into a 504 / 500 response. Re-raising
                    # from inside ``_summary_provider`` keeps the SSE
                    # payload empty (only the [DONE] sentinel flushes)
                    # and lets the exception bubble cleanly.
                    raise err  # noqa: TRY301
                result = result_holder.get("result", {})
                _scrub_error_fields(result)
                summary = result.get("summary", "")
                sources = result.get("sources", [])
                summary, sources = deduplicate_sources_and_remap(summary, sources)
                summary, sources = filter_sources_to_cited(summary, sources)
                return summary, sources_to_url_citations(sources)

            response = Response(
                stream_with_context(
                    stream_chat_completion_sse(
                        reasoning_queue,
                        _summary_provider,
                        model,
                    )
                ),
                mimetype="text/event-stream",
            )
            # Critical for SSE: disable proxy buffering so nginx / cloudflare
            # don't hold frames until the response body fully forms.
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Cache-Control"] = "no-cache"
            return response

        # Non-streaming path: accumulate reasoning lines into a single
        # newline-delimited string and attach it to message.reasoning_content
        # so open-webui's thinking-block renderer still surfaces the agent's
        # decision trail above the answer body.
        non_stream_lines: list[str] = []

        def _accumulate_progress(message: str, percent, metadata) -> None:
            line = _format_reasoning(metadata, message)
            if line is not None:
                non_stream_lines.append(line)

        _, run_fn = build_streaming_search_system(
            query,
            progress_callback=_accumulate_progress,
            **params,
        )
        result = run_fn()
        _scrub_error_fields(result)

        summary = result.get("summary", "")
        sources = result.get("sources", [])
        summary, sources = deduplicate_sources_and_remap(summary, sources)
        summary, sources = filter_sources_to_cited(summary, sources)
        citations = sources_to_url_citations(sources)

        return jsonify(
            chat_completion_response(
                summary,
                model,
                reasoning_content="\n\n".join(non_stream_lines),
                citations=citations,
            )
        )
    except TimeoutError:
        logger.exception("OpenAI-compat chat request timed out")
        return jsonify({"error": {"message": "request timed out"}}), 504
    except Exception:
        logger.exception("Error in OpenAI-compat chat_completions")
        return jsonify({"error": {"message": "internal error"}}), 500
