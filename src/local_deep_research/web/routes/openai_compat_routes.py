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
from ...security.rate_limiter import api_rate_limit, get_current_username
from ..api import _load_user_context_into_params, api_access_control
from ..openai_compat import chat_completion_response, last_user_message

openai_compat_bp = Blueprint("openai_compat", __name__, url_prefix="/v1")


@openai_compat_bp.route("/chat/completions", methods=["POST"])
@api_access_control
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

        username = get_current_username()
        params = {"temperature": data.get("temperature", 0.7)}
        error = _load_user_context_into_params(params, username)
        if error is not None:
            return error

        result = quick_summary(query, **params)
        return jsonify(chat_completion_response(result.get("summary", ""), model))
    except TimeoutError:
        logger.exception("OpenAI-compat chat request timed out")
        return jsonify({"error": {"message": "request timed out"}}), 504
    except Exception:
        logger.exception("Error in OpenAI-compat chat_completions")
        return jsonify({"error": {"message": "internal error"}}), 500
