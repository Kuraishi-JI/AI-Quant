from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .llm.catalog import (
    ResolvedLLMSelection,
    display_provider,
    resolve_api_key,
    resolve_model_selection,
)
from .llm.client import LLMClientError, LLMResponse, SimpleLLMClient


@dataclass(frozen=True)
class ExplainableContext:
    context_id: str
    label: str
    category: str
    short_definition: str
    detailed_description: str
    formula_or_logic: str
    interpretation: str
    related_controls: tuple[str, ...] = field(default_factory=tuple)
    related_metrics: tuple[str, ...] = field(default_factory=tuple)
    warnings_or_common_misunderstandings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AssistantSettings:
    provider: str
    model: str
    api_key: str = ""


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    selection: ResolvedLLMSelection
    warnings: tuple[str, ...] = field(default_factory=tuple)


ChatMessage = dict[str, str]
LLMClientFactory = Callable[..., SimpleLLMClient]
DEFAULT_THREAD_TITLE = "New chat"


SYSTEM_PROMPT = (
    "You are Workbench Assistant, a guide for the Responsible Quant Advisory Workbench. "
    "Explain the Streamlit UI, controls, metrics, model workflow, risk overlays, signal outputs, "
    "explainability views, and exported research memo. Be clear, educational, and cautious. "
    "Do not provide personalized financial advice or trading instructions. Explain uncertainty, "
    "assumptions, and limitations. Reference the selected context and current run snapshot when "
    "available. Answer in the user's language when possible. Return strict JSON with one string "
    "field named answer."
)


CONTEXT_REGISTRY: dict[str, ExplainableContext] = {}


def _register(context: ExplainableContext) -> None:
    CONTEXT_REGISTRY[context.context_id] = context


def _seed_contexts() -> None:
    entries = [
        ExplainableContext(
            "data_source",
            "Data source",
            "control",
            "The input mode used by the workbench.",
            "The app can run on a reproducible reference synthetic market or on an uploaded wide-form price CSV.",
            "Reference mode calls the demo price generator; upload mode parses a CSV with one date column and one column per ticker.",
            "Use the reference dataset for a stable classroom walkthrough, and upload data when testing your own research universe.",
            ("Start date", "End date", "Benchmark"),
            ("asset_count", "date_range"),
            ("Changing the data source changes the available dates, universe, and every downstream metric.",),
        ),
        ExplainableContext(
            "benchmark",
            "Benchmark",
            "control",
            "The reference asset used for market context and performance comparison.",
            "Benchmark returns are used for beta, correlation, benchmark equity, market regime features, and relative performance review.",
            "The selected benchmark column is pulled from the active price table. If it is unavailable in the engine, the first valid asset is used.",
            "Choose a benchmark that reasonably represents the investment opportunity set.",
            ("Data source", "Market state"),
            ("benchmark_total_return", "beta_to_benchmark", "corr_to_benchmark"),
            ("A poor benchmark can make beta and excess-return interpretation misleading.",),
        ),
        ExplainableContext(
            "top_k",
            "Top-K assets",
            "control",
            "The number of highest-ranked assets eligible for signal weighting on each rebalance date.",
            "The walk-forward model ranks assets by predicted next-period return; the allocator keeps the top K names with positive predictions.",
            "The selected names receive raw weights proportional to positive predictions, then concentration and risk overlays are applied.",
            "Lower values create a more concentrated portfolio; higher values can diversify but may include weaker signals.",
            ("Max single-asset weight", "Prediction horizon"),
            ("final_weight", "turnover", "exposure"),
            ("Top-K is not a guarantee that every selected name receives weight; negative or zero predictions are skipped.",),
        ),
        ExplainableContext(
            "retrain_cadence",
            "Retrain cadence",
            "control",
            "How often the Ridge model is refit during the walk-forward loop.",
            "The model trains only on history before each signal date, then refreshes after the selected number of trading days.",
            "A smaller cadence updates the model more often; a larger cadence keeps model coefficients stable for longer.",
            "Use shorter cadences for faster adaptation and longer cadences when you want lower model churn.",
            ("Training history ratio", "Ridge alpha"),
            ("rank_ic", "turnover"),
            ("Frequent retraining can add noise if the training sample is small.",),
        ),
        ExplainableContext(
            "ridge_alpha",
            "Ridge alpha",
            "control",
            "The regularization strength for the standardized Ridge regression model.",
            "Ridge alpha controls how strongly coefficient sizes are penalized, which affects prediction smoothness and feature sensitivity.",
            "The pipeline standardizes features, then fits Ridge(alpha=selected value).",
            "Higher alpha can reduce overfitting but may mute useful signals; lower alpha can be more responsive but less stable.",
            ("Retrain cadence", "Feature importance"),
            ("prediction", "rank_ic", "mae"),
            ("Alpha tuning should be evaluated out-of-sample rather than picked from one favorable run.",),
        ),
        ExplainableContext(
            "max_weight",
            "Max single-asset weight",
            "risk_control",
            "The concentration cap applied to any one asset before portfolio risk overlays.",
            "After positive predictions are converted into weights, the cap redistributes excess weight into uncapped selected assets.",
            "The allocator repeatedly caps weights above the limit and reallocates available excess to names with remaining room.",
            "Lower caps reduce concentration risk; higher caps allow the model's strongest signals to dominate more.",
            ("Top-K assets", "Target annual vol"),
            ("final_weight", "exposure"),
            ("A cap does not remove risk if selected assets are highly correlated.",),
        ),
        ExplainableContext(
            "target_vol",
            "Target annual vol",
            "risk_control",
            "The annualized volatility level the risk overlay tries not to exceed.",
            "The engine estimates recent portfolio volatility from a covariance lookback window and scales exposure down when risk is too high.",
            "vol_scalar = min(1, target_vol / estimated_volatility).",
            "A lower target produces more defensive exposure; a higher target allows more risk during volatile periods.",
            ("Covariance lookback", "VaR limit"),
            ("annual_volatility", "exposure", "risk_events"),
            ("Volatility targeting can reduce exposure after volatility has already risen.",),
        ),
        ExplainableContext(
            "var_limit",
            "1-day VaR limit",
            "risk_control",
            "The one-day loss threshold used to throttle portfolio exposure.",
            "If the estimated one-day VaR exceeds the selected limit, the overlay scales weights down before the backtest applies them.",
            "var_scalar = min(1, var_limit / estimated_var).",
            "Lower limits create a stricter tail-risk governor; higher limits allow more exposure.",
            ("VaR confidence", "Target annual vol"),
            ("var_95", "es_95", "risk_events"),
            ("VaR does not describe losses beyond the threshold; review expected shortfall as well.",),
        ),
        ExplainableContext(
            "drawdown_guard",
            "Drawdown guard",
            "risk_control",
            "An exposure reduction rule triggered by severe portfolio drawdowns.",
            "When portfolio drawdown breaches the guard threshold, the backtest applies a reduced exposure scalar.",
            "If drawdown is below the selected limit, applied weights are multiplied by drawdown_exposure.",
            "The guard is meant to keep the strategy from staying fully exposed during adverse regimes.",
            ("Exposure after drawdown guard", "Max drawdown"),
            ("max_drawdown", "drawdown", "exposure"),
            ("A drawdown guard can also delay participation in rebounds.",),
        ),
        ExplainableContext(
            "transaction_cost",
            "Transaction cost",
            "risk_control",
            "The assumed cost in basis points charged against portfolio turnover.",
            "The backtest subtracts transaction cost whenever applied weights change between rebalance dates.",
            "transaction_cost = turnover * transaction_cost_bps / 10000.",
            "Higher costs make high-turnover settings less attractive and reveal whether signal quality survives implementation friction.",
            ("Retrain cadence", "Top-K assets"),
            ("avg_turnover", "strategy_return"),
            ("Costs are simplified research assumptions, not broker-specific execution estimates.",),
        ),
        ExplainableContext(
            "market_state",
            "Market state",
            "output",
            "A compact regime label derived from benchmark momentum, volatility, drawdown, breadth, and dispersion.",
            "The Architecture and Data view reports whether the latest market context is risk-on, risk-off, or transition.",
            "Risk-off is assigned when benchmark drawdown or volatility is elevated; risk-on needs nonnegative momentum and broad participation.",
            "Use it as a situational label for interpreting risk posture, not as a standalone trading signal.",
            ("Benchmark", "Risk Command"),
            ("market_vol_20d", "breadth_20d", "dispersion_20d"),
            ("The regime label is rule-based and depends on the selected benchmark and available history.",),
        ),
        ExplainableContext(
            "portfolio_weights",
            "Portfolio weights",
            "output",
            "The fraction of capital assigned to each asset after signal ranking and risk overlays.",
            "Weights start from model predictions, pass through concentration limits, then are scaled by volatility, VaR, and drawdown controls.",
            "final_weight is the applied long-only asset weight; uninvested capital is implicitly cash.",
            "Review latest weights with exposure, cash weight, and risk events to understand why the portfolio is or is not fully invested.",
            ("Top-K assets", "Max single-asset weight"),
            ("final_weight", "cash_weight", "exposure"),
            ("Weights are research outputs and are not executable personalized advice.",),
        ),
        ExplainableContext(
            "exposure",
            "Exposure",
            "risk_metric",
            "The invested share of the portfolio.",
            "Exposure is the sum of applied asset weights. Anything below 100% is effectively uninvested cash in this research model.",
            "exposure = sum(final asset weights).",
            "Falling exposure usually means volatility, VaR, or drawdown controls reduced risk.",
            ("Target annual vol", "1-day VaR limit", "Drawdown guard"),
            ("cash_weight", "latest_exposure", "avg_exposure"),
            ("Low exposure can be intentional risk control rather than a failure to allocate.",),
        ),
        ExplainableContext(
            "sharpe_ratio",
            "Sharpe ratio",
            "metric",
            "A simple risk-adjusted return measure using total volatility.",
            "The workbench computes Sharpe from daily strategy returns annualized with a 252-trading-day convention.",
            "sharpe = mean_daily_return * sqrt(252) / daily_return_std.",
            "Higher Sharpe is generally better, but it can be unstable over short samples and does not isolate downside risk.",
            ("Target annual vol", "Benchmark"),
            ("annual_return", "annual_volatility"),
            ("Sharpe treats upside and downside volatility the same way.",),
        ),
        ExplainableContext(
            "max_drawdown",
            "Max drawdown",
            "metric",
            "The worst peak-to-trough loss in the strategy equity curve.",
            "Drawdown shows path risk that average returns and volatility can miss.",
            "drawdown = equity / running_peak - 1; max drawdown is the minimum drawdown.",
            "More negative values indicate deeper historical losses and should be reviewed with the drawdown guard settings.",
            ("Drawdown guard", "Risk Command"),
            ("calmar_ratio", "strategy_window_drawdown"),
            ("Drawdown is path-dependent and may not predict future stress behavior.",),
        ),
        ExplainableContext(
            "var_95",
            "VaR 95",
            "risk_metric",
            "An estimate of a bad one-day loss threshold at 95% confidence.",
            "The report estimates VaR from historical strategy returns and uses a related recent-history estimate in the overlay.",
            "var_95 = negative 5th percentile of daily strategy returns.",
            "Higher VaR means larger tail-loss exposure and can justify lower portfolio exposure.",
            ("1-day VaR limit", "VaR confidence"),
            ("es_95", "rolling_var_60d"),
            ("VaR says little about losses worse than the threshold.",),
        ),
        ExplainableContext(
            "es_95",
            "Expected Shortfall 95",
            "risk_metric",
            "The average loss after returns breach the VaR 95 threshold.",
            "Expected shortfall looks into the tail rather than stopping at the VaR cutoff.",
            "es_95 = negative average of returns at or below the 5th percentile.",
            "Use it beside VaR to understand how bad the worst days were on average.",
            ("1-day VaR limit", "Risk Command"),
            ("var_95", "rolling_es_60d"),
            ("Expected shortfall is sensitive to sample size and outliers.",),
        ),
        ExplainableContext(
            "feature_importance",
            "Feature importance",
            "explainability",
            "A global view of which features matter most to the latest prediction model.",
            "The Explainability Lab combines coefficient magnitude and permutation-style importance to summarize model drivers.",
            "Features are standardized before Ridge fitting, then importance views compare relative effect sizes and prediction sensitivity.",
            "Use it to understand broad model behavior before drilling into a single asset-date prediction.",
            ("Ridge alpha", "What-if feature"),
            ("permutation_importance", "abs_coefficient"),
            ("Importance is model-specific and can change across retraining windows.",),
        ),
        ExplainableContext(
            "local_contributions",
            "Local contributions",
            "explainability",
            "A per-asset explanation of what pushed one prediction up or down.",
            "The local driver chart decomposes the selected asset-date prediction using the latest model and the selected row's features.",
            "For the standardized linear model, contribution is derived from feature z-scores and model coefficients.",
            "Use it to check whether a recommended position is driven by intuitive features or by noisy outliers.",
            ("Signal date", "Asset to explain"),
            ("prediction", "feature_value", "z_score_vs_median"),
            ("Local explanations describe the model, not the real causal reason an asset moved.",),
        ),
        ExplainableContext(
            "what_if",
            "What-if curve",
            "explainability",
            "A one-feature sensitivity curve for the selected asset-date prediction.",
            "The app varies one feature over a historical range while holding the rest of the selected row constant.",
            "It feeds each synthetic row through the latest Ridge pipeline and plots the predicted return response.",
            "Use it to see whether the model response is directionally sensible and how sensitive it is near the current value.",
            ("What-if feature", "Local contributions"),
            ("predicted_return", "feature_value"),
            ("One-factor what-if charts do not capture feature interactions or future market changes.",),
        ),
        ExplainableContext(
            "risk_events",
            "Risk events",
            "output",
            "Logged moments when risk overlays reduced or adjusted portfolio exposure.",
            "Risk events show when volatility targeting, VaR throttling, or drawdown guards were active during the simulation.",
            "The event table records date, event type, and a human-readable detail string.",
            "Use it to connect performance changes to explicit governance controls.",
            ("Target annual vol", "1-day VaR limit", "Drawdown guard"),
            ("exposure", "dd_scalar", "estimated_var"),
            ("An event log explains model behavior but is not a compliance approval record.",),
        ),
        ExplainableContext(
            "api_key",
            "API key",
            "assistant_control",
            "A session-only credential used to authenticate hosted LLM calls.",
            "The Workbench Assistant accepts a password-masked UI key or reads the provider-specific environment variable.",
            "Keys are passed only to the provider HTTP client and are redacted from prompts, chat history, and answers where possible.",
            "Enter a key when you want live assistant answers; leave it blank if you do not want provider calls.",
            ("Assistant provider", "Assistant model"),
            (),
            ("Never paste API keys into normal chat messages.",),
        ),
    ]
    for entry in entries:
        _register(entry)


_seed_contexts()


def get_context_registry() -> Mapping[str, ExplainableContext]:
    return CONTEXT_REGISTRY


def normalize_context_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def find_context(context_id_or_label: str | None) -> ExplainableContext | None:
    if not context_id_or_label:
        return None
    key = normalize_context_key(context_id_or_label)
    if key in CONTEXT_REGISTRY:
        return CONTEXT_REGISTRY[key]
    for context in CONTEXT_REGISTRY.values():
        aliases = {
            normalize_context_key(context.label),
            normalize_context_key(context.context_id),
            normalize_context_key(context.short_definition),
        }
        if key in aliases:
            return context
    return None


def unknown_context(selected_text: str) -> ExplainableContext:
    clean = selected_text.strip() or "selected workbench content"
    return ExplainableContext(
        context_id="unknown_selection",
        label=clean[:80],
        category="unknown",
        short_definition="A user-selected item that is not yet in the workbench context registry.",
        detailed_description=(
            "The assistant could not match this exact text to a registered workbench concept. "
            "It will explain it cautiously using general platform knowledge and any current run outputs."
        ),
        formula_or_logic="No registered formula is available for this selected text.",
        interpretation="Treat this as a general explanation unless the UI provides a more specific term.",
        warnings_or_common_misunderstandings=("The answer may be less specific than a registered context explanation.",),
    )


def context_to_markdown(context: ExplainableContext, snapshot: Mapping[str, Any] | None = None) -> str:
    lines = [
        f"**{context.label}**",
        "",
        f"**What it means:** {context.short_definition}",
        "",
        f"**Where it appears / how it is used:** {context.detailed_description}",
        "",
        f"**Formula or logic:** {context.formula_or_logic}",
        "",
        f"**How to interpret it:** {context.interpretation}",
    ]
    if context.related_controls:
        lines += ["", "**Related controls:** " + ", ".join(context.related_controls)]
    if context.related_metrics:
        lines += ["", "**Related metrics:** " + ", ".join(context.related_metrics)]
    if context.warnings_or_common_misunderstandings:
        lines += ["", "**Common mistakes:** " + " ".join(context.warnings_or_common_misunderstandings)]
    if snapshot:
        current = snapshot.get("current_settings") or {}
        metrics = snapshot.get("metrics") or {}
        latest_allocation = snapshot.get("latest_allocation") or {}
        hints = []
        if current:
            hints.append(
                "Current settings include "
                f"benchmark={current.get('benchmark', 'n/a')}, "
                f"top_k={current.get('top_k', 'n/a')}, "
                f"target_vol={current.get('target_vol', 'n/a')}, "
                f"var_limit={current.get('var_limit', 'n/a')}."
            )
        if metrics:
            metric_bits = []
            for name in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown", "var_95"):
                if name in metrics:
                    metric_bits.append(f"{name}={metrics[name]}")
            if metric_bits:
                hints.append("Current metrics snapshot: " + ", ".join(metric_bits) + ".")
        if latest_allocation:
            alloc_bits = [f"{ticker}={weight}" for ticker, weight in list(latest_allocation.items())[:5]]
            hints.append("Largest current weights: " + ", ".join(alloc_bits) + ".")
        if hints:
            lines += ["", "**Current run context:** " + " ".join(hints)]
    lines += ["", "**What to check next:** Review related controls, risk warnings, and whether the run uses the reference dataset or uploaded prices."]
    return "\n".join(lines)


def build_run_snapshot(result: Any | None) -> dict[str, Any]:
    if result is None:
        return {
            "current_settings": {},
            "data": {},
            "metrics": {},
            "market_context": {},
            "latest_allocation": {},
            "latest_signals": [],
            "risk_events": [],
        }

    config = getattr(result, "config", None)
    prices = getattr(result, "prices", None)
    metrics = dict(getattr(result, "metrics", {}) or {})
    compact_metrics = {
        key: _format_snapshot_value(value)
        for key, value in metrics.items()
        if key
        in {
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "avg_turnover",
            "var_95",
            "es_95",
            "beta_to_benchmark",
            "corr_to_benchmark",
            "benchmark_total_return",
        }
    }

    latest_allocation: dict[str, str] = {}
    weights = getattr(result, "applied_weights", None)
    if weights is not None and not weights.empty:
        latest_weights = weights.iloc[-1].sort_values(ascending=False)
        latest_allocation = {
            str(ticker): _format_snapshot_value(weight)
            for ticker, weight in latest_weights[latest_weights > 0.0001].head(8).items()
        }

    risk_events = []
    risk_frame = getattr(result, "risk_events", None)
    if risk_frame is not None and not risk_frame.empty:
        for row in risk_frame.tail(5).to_dict("records"):
            risk_events.append({str(key): _stringify_snapshot_value(value) for key, value in row.items()})

    latest_signals = []
    decision_log = getattr(result, "decision_log", None)
    if decision_log is not None and not decision_log.empty:
        latest_date = decision_log["signal_date"].max()
        latest_rows = decision_log.loc[decision_log["signal_date"] == latest_date].sort_values(
            ["prediction", "ticker"], ascending=[False, True]
        )
        for row in latest_rows.head(6).to_dict("records"):
            latest_signals.append(
                {
                    "ticker": str(row.get("ticker", "")),
                    "prediction": _format_snapshot_value(row.get("prediction")),
                    "rank": _stringify_snapshot_value(row.get("rank")),
                    "selected": _stringify_snapshot_value(row.get("selected")),
                    "final_weight": _format_snapshot_value(row.get("final_weight")),
                }
            )

    market_context = {
        str(key): _stringify_snapshot_value(value)
        for key, value in (getattr(result, "market_context", {}) or {}).items()
    }

    tickers: list[str] = []
    date_range: dict[str, str] = {}
    if prices is not None and not prices.empty:
        tickers = [str(item) for item in prices.columns]
        date_range = {
            "start": _stringify_snapshot_value(prices.index.min()),
            "end": _stringify_snapshot_value(prices.index.max()),
        }

    return {
        "current_settings": {
            "benchmark": getattr(config, "benchmark", ""),
            "train_ratio": _format_snapshot_value(getattr(config, "train_ratio", "")),
            "retrain_every": _stringify_snapshot_value(getattr(config, "retrain_every", "")),
            "top_k": _stringify_snapshot_value(getattr(config, "top_k", "")),
            "prediction_horizon": _stringify_snapshot_value(getattr(config, "prediction_horizon", "")),
            "model_alpha": _format_snapshot_value(getattr(config, "model_alpha", "")),
            "max_weight": _format_snapshot_value(getattr(config, "max_weight", "")),
            "target_vol": _format_snapshot_value(getattr(config, "target_vol", "")),
            "var_limit": _format_snapshot_value(getattr(config, "var_limit", "")),
            "var_confidence": _format_snapshot_value(getattr(config, "var_confidence", "")),
            "drawdown_limit": _format_snapshot_value(getattr(config, "drawdown_limit", "")),
            "drawdown_exposure": _format_snapshot_value(getattr(config, "drawdown_exposure", "")),
            "transaction_cost_bps": _format_snapshot_value(getattr(config, "transaction_cost_bps", "")),
            "covariance_lookback": _stringify_snapshot_value(getattr(config, "covariance_lookback", "")),
        },
        "data": {
            "tickers": tickers,
            "asset_count": len(tickers),
            "date_range": date_range,
            "feature_count": len(getattr(result, "feature_columns", []) or []),
        },
        "metrics": compact_metrics,
        "market_context": market_context,
        "latest_allocation": latest_allocation,
        "latest_signals": latest_signals,
        "risk_events": risk_events,
    }


def _stringify_snapshot_value(value: Any) -> str:
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except TypeError:
            pass
    return str(value)


def _format_snapshot_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _stringify_snapshot_value(value)
    if abs(number) <= 2.0:
        return f"{number:.4f}"
    return f"{number:.2f}"


def build_llm_prompt(
    user_message: str,
    *,
    active_context: ExplainableContext | None,
    snapshot: Mapping[str, Any],
    history: list[ChatMessage],
) -> str:
    safe_history = [
        {"role": item.get("role", ""), "content": item.get("content", "")[:1200]}
        for item in history[-8:]
        if item.get("role") in {"user", "assistant"}
    ]
    payload = {
        "instruction": "Answer the user's Responsible Quant Advisory Workbench help question. Return JSON with field answer.",
        "user_message": user_message,
        "active_context": _context_payload(active_context),
        "current_workbench_snapshot": snapshot,
        "recent_chat_history": safe_history,
        "rules": [
            "Do not provide personalized financial advice.",
            "Do not claim hidden data access.",
            "If current run outputs are unavailable, say so.",
            "Never mention or reveal API keys.",
            "Answer in the user's language when possible.",
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def _context_payload(context: ExplainableContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "context_id": context.context_id,
        "label": context.label,
        "category": context.category,
        "short_definition": context.short_definition,
        "detailed_description": context.detailed_description,
        "formula_or_logic": context.formula_or_logic,
        "interpretation": context.interpretation,
        "related_controls": list(context.related_controls),
        "related_metrics": list(context.related_metrics),
        "warnings_or_common_misunderstandings": list(context.warnings_or_common_misunderstandings),
    }


def parse_assistant_payload(text: str) -> str:
    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        return parsed["answer"]
    raise ValueError("Assistant response JSON did not contain an answer field.")


def redact_secrets(text: str, secrets: list[str] | tuple[str, ...]) -> str:
    clean = text
    for secret in secrets:
        if secret:
            clean = clean.replace(secret, "[REDACTED]")
    clean = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", clean)
    clean = re.sub(r"AIza[0-9A-Za-z_\-]{12,}", "[REDACTED]", clean)
    return clean


def _hosted_llm_required_message(selection: ResolvedLLMSelection) -> str:
    provider_label = display_provider(selection.requested_provider)
    env_hint = f"`{selection.api_key_env}`" if selection.api_key_env else "the provider environment variable"
    return (
        f"Workbench Assistant is set to use **{provider_label} / {selection.requested_model}**, "
        "and this assistant is configured for hosted LLM calls only. "
        f"Please enter a session-only API key in the right sidebar or set {env_hint}, then ask again. "
        "No offline assistant answer was generated for this message."
    )


def generate_assistant_response(
    user_message: str,
    settings: AssistantSettings,
    *,
    active_context: ExplainableContext | None = None,
    snapshot: Mapping[str, Any] | None = None,
    history: list[ChatMessage] | None = None,
    env: Mapping[str, str] | None = None,
    client_factory: LLMClientFactory = SimpleLLMClient,
) -> AssistantResponse:
    snapshot = snapshot or {}
    history = history or []
    selection = resolve_model_selection(
        settings.provider,
        settings.model,
        ui_api_key=settings.api_key,
        env=env,
    )
    api_key, _, _ = resolve_api_key(settings.provider, ui_api_key=settings.api_key, env=env)
    warnings = list(selection.warnings)
    if not api_key:
        answer = _hosted_llm_required_message(selection)
        return AssistantResponse(
            answer=redact_secrets(answer, [settings.api_key]),
            selection=selection,
            warnings=tuple(warnings),
        )

    try:
        response: LLMResponse = client_factory(selection, api_key=api_key).generate_text(
            build_llm_prompt(user_message, active_context=active_context, snapshot=snapshot, history=history),
            system=SYSTEM_PROMPT,
        )
        answer = parse_assistant_payload(response.text)
        return AssistantResponse(
            answer=redact_secrets(answer, [settings.api_key, api_key or ""]),
            selection=selection,
            warnings=tuple(warnings),
        )
    except (LLMClientError, ValueError, KeyError, json.JSONDecodeError) as exc:
        safe_error = redact_secrets(str(exc), [settings.api_key, api_key or ""])
        warnings.append(f"Provider assistant call failed. Error: {safe_error}")
        provider_label = display_provider(selection.requested_provider)
        answer = (
            f"I could not get a response from **{provider_label} / {selection.requested_model}**. "
            "Please check the API key, model access, and network connection, then try again. "
            "No offline assistant fallback is enabled."
        )
        return AssistantResponse(
            answer=redact_secrets(answer, [settings.api_key, api_key or ""]),
            selection=selection,
            warnings=tuple(warnings),
        )


def clear_chat_state(state: dict[str, Any]) -> None:
    clear_active_chat_thread(state)


def delete_chat_message(state: dict[str, Any], index: int) -> bool:
    messages = get_active_chat_thread(state)["messages"]
    if index < 0 or index >= len(messages):
        return False
    del messages[index]
    return True


def edit_user_chat_message_for_regeneration(state: dict[str, Any], index: int, new_content: str) -> bool:
    thread = get_active_chat_thread(state)
    messages = thread["messages"]
    if index < 0 or index >= len(messages):
        return False
    if messages[index].get("role") != "user":
        return False
    clean_content = new_content.strip()
    if not clean_content:
        return False
    messages[index]["content"] = clean_content
    if index + 1 < len(messages) and messages[index + 1].get("role") == "assistant":
        del messages[index + 1]
    first_user_index = next((idx for idx, message in enumerate(messages) if message.get("role") == "user"), None)
    if first_user_index == index:
        thread["title"] = _title_from_messages(messages) or DEFAULT_THREAD_TITLE
    return True


def ensure_chat_threads(state: dict[str, Any]) -> list[dict[str, Any]]:
    threads = state.get("assistant_threads")
    if not threads:
        thread = {
            "id": "chat_1",
            "title": DEFAULT_THREAD_TITLE,
            "messages": [],
            "context_id": None,
            "context_text": "",
            "context_explained": False,
        }
        state["assistant_threads"] = [thread]
        state["assistant_active_thread_id"] = "chat_1"
        state["assistant_thread_counter"] = 1
    else:
        state.setdefault("assistant_thread_counter", len(threads))
        if not state.get("assistant_active_thread_id") or not any(thread["id"] == state["assistant_active_thread_id"] for thread in threads):
            state["assistant_active_thread_id"] = threads[0]["id"]
    return state["assistant_threads"]


def get_active_chat_thread(state: dict[str, Any]) -> dict[str, Any]:
    threads = ensure_chat_threads(state)
    active_id = state.get("assistant_active_thread_id")
    for thread in threads:
        if thread["id"] == active_id:
            return thread
    state["assistant_active_thread_id"] = threads[0]["id"]
    return threads[0]


def create_chat_thread(state: dict[str, Any], *, title: str = DEFAULT_THREAD_TITLE) -> dict[str, Any]:
    threads = ensure_chat_threads(state)
    counter = int(state.get("assistant_thread_counter", len(threads))) + 1
    state["assistant_thread_counter"] = counter
    thread = {
        "id": f"chat_{counter}",
        "title": title,
        "messages": [],
        "context_id": None,
        "context_text": "",
        "context_explained": False,
    }
    threads.append(thread)
    state["assistant_active_thread_id"] = thread["id"]
    return thread


def delete_chat_thread(state: dict[str, Any], thread_id: str) -> bool:
    threads = ensure_chat_threads(state)
    index = next((idx for idx, thread in enumerate(threads) if thread["id"] == thread_id), None)
    if index is None:
        return False
    del threads[index]
    if not threads:
        state["assistant_thread_counter"] = 1
        thread = {
            "id": "chat_1",
            "title": DEFAULT_THREAD_TITLE,
            "messages": [],
            "context_id": None,
            "context_text": "",
            "context_explained": False,
        }
        threads.append(thread)
        state["assistant_active_thread_id"] = "chat_1"
        return True
    if state.get("assistant_active_thread_id") == thread_id:
        state["assistant_active_thread_id"] = threads[min(index, len(threads) - 1)]["id"]
    return True


def clear_active_chat_thread(state: dict[str, Any]) -> None:
    thread = get_active_chat_thread(state)
    thread["messages"] = []
    thread["context_id"] = None
    thread["context_text"] = ""
    thread["context_explained"] = False
    thread["title"] = DEFAULT_THREAD_TITLE


def set_active_thread_title_from_messages(state: dict[str, Any]) -> None:
    thread = get_active_chat_thread(state)
    if thread.get("title") != DEFAULT_THREAD_TITLE:
        return
    title = _title_from_messages(thread.get("messages", []))
    if title:
        thread["title"] = title


def _title_from_messages(messages: list[ChatMessage]) -> str:
    for message in messages:
        if message.get("role") == "user" and message.get("content", "").strip():
            content = re.sub(r"\s+", " ", message["content"].strip())
            return content[:34] + ("..." if len(content) > 34 else "")
    return ""
