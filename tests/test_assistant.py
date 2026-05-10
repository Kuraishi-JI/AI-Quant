from __future__ import annotations

import json

import pytest

from ai_quant_investing.assistant import (
    AssistantSettings,
    build_llm_prompt,
    build_run_snapshot,
    clear_chat_state,
    create_chat_thread,
    delete_chat_message,
    delete_chat_thread,
    edit_user_chat_message_for_regeneration,
    ensure_chat_threads,
    find_context,
    generate_assistant_response,
    get_active_chat_thread,
    get_context_registry,
    set_active_thread_title_from_messages,
)
from ai_quant_investing.assistant_llm_settings import (
    get_assistant_models_for_provider,
    get_assistant_providers,
    resolve_assistant_runtime_config,
)
from ai_quant_investing.core.data import generate_demo_prices
from ai_quant_investing.core.pipeline import PrototypeConfig, run_prototype
from ai_quant_investing.llm.client import LLMResponse


def test_context_registry_contains_key_workbench_terms():
    registry = get_context_registry()

    for key in [
        "data_source",
        "benchmark",
        "top_k",
        "target_vol",
        "portfolio_weights",
        "sharpe_ratio",
        "var_95",
        "feature_importance",
        "risk_events",
        "api_key",
    ]:
        assert key in registry


def test_assistant_llm_settings_exclude_deterministic_provider():
    assert get_assistant_providers() == ["OpenAI", "Anthropic", "Google"]

    for provider in get_assistant_providers():
        assert {model.provider for model in get_assistant_models_for_provider(provider)} == {provider.lower()}

    with pytest.raises(ValueError, match="hosted LLM providers only"):
        get_assistant_models_for_provider("deterministic")


def test_missing_api_key_requires_real_llm_for_workbench_assistant():
    response = generate_assistant_response(
        "Explain VaR 95",
        AssistantSettings(provider="OpenAI", model="gpt-5.5"),
        active_context=find_context("var_95"),
        snapshot={},
        history=[],
        env={},
    )

    assert response.selection.resolved_provider == "openai"
    assert "hosted LLM calls only" in response.answer
    assert "No offline assistant answer" in response.answer


def test_assistant_runtime_config_prefers_ui_key_and_warns_when_missing():
    with_ui_key = resolve_assistant_runtime_config(
        "Anthropic",
        "claude-sonnet-4-6",
        ui_api_key="anthropic-ui-secret",
        env={"ANTHROPIC_API_KEY": "anthropic-env-secret"},
    )
    missing = resolve_assistant_runtime_config("Google", "gemini-2.5-flash", env={})

    assert with_ui_key.has_api_key is True
    assert with_ui_key.api_key_source == "ui"
    assert with_ui_key.api_key == "anthropic-ui-secret"
    assert missing.has_api_key is False
    assert "will wait for a real LLM key" in missing.warnings[0]


def test_provider_model_settings_are_reused_for_assistant_llm_call():
    seen: dict[str, str] = {}

    class FakeClient:
        def __init__(self, selection, *, api_key=None):
            seen["provider"] = selection.resolved_provider
            seen["model"] = selection.resolved_model
            seen["api_key"] = api_key

        def generate_text(self, prompt: str, *, system: str) -> LLMResponse:
            seen["prompt"] = prompt
            return LLMResponse(
                text=json.dumps({"answer": "Provider-backed assistant answer."}),
                provider=seen["provider"],
                model=seen["model"],
            )

    response = generate_assistant_response(
        "Explain Sharpe",
        AssistantSettings(provider="OpenAI", model="gpt-5.5", api_key="sk-test-secret"),
        active_context=find_context("sharpe_ratio"),
        snapshot={"metrics": {"sharpe_ratio": "1.23"}},
        history=[],
        env={},
        client_factory=FakeClient,
    )

    assert seen["provider"] == "openai"
    assert seen["model"] == "gpt-5.5"
    assert seen["api_key"] == "sk-test-secret"
    assert response.answer == "Provider-backed assistant answer."
    assert "sk-test-secret" not in seen["prompt"]


def test_api_key_values_are_redacted_from_assistant_answer():
    class FakeClient:
        def __init__(self, selection, *, api_key=None):
            self.selection = selection

        def generate_text(self, prompt: str, *, system: str) -> LLMResponse:
            return LLMResponse(
                text=json.dumps({"answer": "Never show sk-redaction-secret in chat."}),
                provider="openai",
                model="gpt-5.5",
            )

    response = generate_assistant_response(
        "Explain API key",
        AssistantSettings(provider="OpenAI", model="gpt-5.5", api_key="sk-redaction-secret"),
        active_context=find_context("api_key"),
        snapshot={},
        history=[],
        env={},
        client_factory=FakeClient,
    )

    assert "sk-redaction-secret" not in response.answer
    assert "[REDACTED]" in response.answer


def test_multi_conversation_mode_creates_switches_clears_and_deletes_threads():
    state = {}

    threads = ensure_chat_threads(state)
    assert len(threads) == 1
    active_thread = get_active_chat_thread(state)
    active_thread["messages"].append({"role": "user", "content": "existing question"})
    set_active_thread_title_from_messages(state)
    assert active_thread["title"] == "existing question"

    new_thread = create_chat_thread(state)
    new_thread["messages"].append({"role": "user", "content": "second chat"})
    assert state["assistant_active_thread_id"] == new_thread["id"]
    assert len(state["assistant_threads"]) == 2

    clear_chat_state(state)
    assert get_active_chat_thread(state)["messages"] == []
    assert get_active_chat_thread(state)["title"] == "New chat"

    first_thread_id = state["assistant_threads"][0]["id"]
    assert delete_chat_thread(state, first_thread_id) is True
    assert all(thread["id"] != first_thread_id for thread in state["assistant_threads"])
    assert len(state["assistant_threads"]) == 1


def test_delete_and_edit_message_helpers():
    state = {}
    ensure_chat_threads(state)
    thread = get_active_chat_thread(state)
    thread["messages"] = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "assistant", "content": "standalone answer"},
    ]

    assert edit_user_chat_message_for_regeneration(state, 0, "new question") is True
    assert [message["content"] for message in thread["messages"]] == ["new question", "standalone answer"]
    assert delete_chat_message(state, 1) is True
    assert delete_chat_message(state, 99) is False


def test_assistant_state_helpers_do_not_mutate_workbench_state_keys():
    sentinel_result = object()
    state = {
        "prototype_result": sentinel_result,
        "prototype_benchmark": "SPY",
        "prototype_error": None,
    }

    ensure_chat_threads(state)
    create_chat_thread(state)
    clear_chat_state(state)
    delete_chat_thread(state, "chat_1")

    assert state["prototype_result"] is sentinel_result
    assert state["prototype_benchmark"] == "SPY"
    assert state["prototype_error"] is None


def test_current_run_snapshot_is_included_in_prompt():
    prices = generate_demo_prices("2020-01-01", "2021-12-31", seed=4)
    result = run_prototype(prices, PrototypeConfig(retrain_every=42))
    snapshot = build_run_snapshot(result)
    prompt = build_llm_prompt(
        "Explain the latest allocation",
        active_context=find_context("portfolio_weights"),
        snapshot=snapshot,
        history=[],
    )

    assert snapshot["metrics"]["sharpe_ratio"]
    assert snapshot["latest_allocation"]
    assert result.config.benchmark in prompt
