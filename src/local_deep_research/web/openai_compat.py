"""Helpers for the OpenAI-compatible chat endpoint.

Open-webui connects to LDR as an OpenAI-compatible "model backend", so these
helpers translate between the OpenAI chat wire format and LDR's internal
question-answering entry point (``quick_summary``).
"""

from __future__ import annotations

import time
import uuid
from typing import Any


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
