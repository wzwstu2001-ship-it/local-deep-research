"""Unit tests for the OpenAI-compat chat helpers."""

from local_deep_research.web.openai_compat import (
    chat_completion_response,
    last_user_message,
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
