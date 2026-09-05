"""OpenAI-compatible chat endpoint for open-webui.

Open-webui connects to LDR as an OpenAI-compatible model backend. This
blueprint exposes ``/v1/chat/completions``, which folds an OpenAI chat
request into a single LDR ``quick_summary`` call (the agentic
research/retrieval path) and returns an OpenAI ``chat.completion`` body.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from loguru import logger

from ...security.decorators import require_json_body
from ...security.rate_limiter import api_rate_limit
from ..api import (
    _load_user_context_into_params,
    _scrub_error_fields,
    get_openai_compat_username,
)
from ..openai_compat import chat_completion_response, last_user_message

openai_compat_bp = Blueprint("openai_compat", __name__, url_prefix="/v1")


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
        from ...api.research_functions import quick_summary

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
        error = _load_user_context_into_params(params, username)
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
        settings_snapshot["search.tool"] = f"collection_{collection_id}"

        result = quick_summary(query, **params)
        _scrub_error_fields(result)
        return jsonify(chat_completion_response(result.get("summary", ""), model))
    except TimeoutError:
        logger.exception("OpenAI-compat chat request timed out")
        return jsonify({"error": {"message": "request timed out"}}), 504
    except Exception:
        logger.exception("Error in OpenAI-compat chat_completions")
        return jsonify({"error": {"message": "internal error"}}), 500
