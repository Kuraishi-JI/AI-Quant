from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd
import plotly.express as px
import pytest

from ai_quant_investing.core.data import generate_demo_prices, load_prices_from_csv
from ai_quant_investing.core.pipeline import PrototypeConfig, run_prototype
from ai_quant_investing.explainability.analysis import (
    build_signal_narrative,
    feature_percentile_table,
    local_contributions,
    prediction_bucket_table,
    standardized_importance,
    what_if_curve,
)
from ai_quant_investing.ui.app import (
    APP_DISPLAY_NAME,
    AUTO_THEME_NAME,
    DISPLAY_PLACEHOLDER,
    REPORT_DISPLAY_NAME,
    STRATEGY_DISPLAY_NAME,
    THEME_PRESETS,
    _active_palette,
    _build_alert_feed,
    _panel_header_markup,
    _apply_plot_style,
    _gauge_figure,
    _prepare_display_frame,
    _in_follow_streamlit_mode,
    _market_tape_html,
    _palette_var_block,
    _build_report_markdown,
    _series_metric_table,
    _strategy_comparison_frame,
    _workstation_summary_html,
)


@pytest.fixture(scope="module")
def prototype_result():
    prices = generate_demo_prices("2020-01-01", "2024-12-31", seed=9)
    config = PrototypeConfig(top_k=3, retrain_every=42, model_alpha=1.2)
    return run_prototype(prices, config)


def test_pipeline_smoke(prototype_result):
    assert not prototype_result.performance.empty
    assert not prototype_result.predictions.empty
    assert not prototype_result.applied_weights.empty
    assert prototype_result.applied_weights.sum(axis=1).max() <= 1.000001
    assert prototype_result.market_context["market_state"] in {"risk_on", "risk_off", "transition"}
    assert {"annual_return", "sharpe_ratio", "var_95"}.issubset(prototype_result.metrics)


def test_explainability_helpers(prototype_result):
    predictions = prototype_result.predictions
    sample = predictions.sample(min(500, len(predictions)), random_state=42)
    importance = standardized_importance(
        prototype_result.latest_model,
        sample[prototype_result.feature_columns],
        sample["target"],
        prototype_result.feature_columns,
    )

    row = predictions.iloc[-1]
    history = prototype_result.panel[prototype_result.panel["date"] <= row["date"]]
    contributions = local_contributions(prototype_result.latest_model, row, prototype_result.feature_columns)
    percentiles = feature_percentile_table(
        row,
        history,
        prototype_result.feature_columns,
        focus_features=contributions.head(5)["feature"].tolist(),
    )
    what_if = what_if_curve(
        prototype_result.latest_model,
        row,
        "mom_20d",
        history,
        prototype_result.feature_columns,
    )
    buckets = prediction_bucket_table(predictions)
    narrative = build_signal_narrative(
        contributions,
        predicted_return=float(row["prediction"]),
        realized_return=float(row["target"]),
        final_weight=0.25,
    )

    assert not importance.empty
    assert not contributions.empty
    assert not percentiles.empty
    merged = contributions.head(5).merge(
        percentiles.drop(columns=["feature_value"], errors="ignore"),
        on="feature",
        how="left",
    )
    assert "feature_value" in merged.columns
    assert len(what_if) == 25
    assert buckets["bucket"].min() == 1
    assert "target weight" in narrative


def test_load_prices_from_csv():
    csv_bytes = b"date,SPY,QQQ\n2024-01-02,470.12,402.55\n2024-01-03,468.42,399.81\n"
    prices = load_prices_from_csv(csv_bytes)
    assert list(prices.columns) == ["SPY", "QQQ"]
    assert len(prices) == 2


def test_dashboard_helper_outputs(prototype_result):
    comparison = _strategy_comparison_frame(prototype_result, "SPY")
    metric_table = _series_metric_table(comparison[[STRATEGY_DISPLAY_NAME, "SPY", "Equal Weight Universe", "Top-K Equal Weight"]])
    memo = _build_report_markdown(prototype_result, "SPY")
    alerts = _build_alert_feed(prototype_result, "SPY")
    tape_html = _market_tape_html(prototype_result, "SPY")
    workstation_html = _workstation_summary_html(prototype_result, "SPY")

    assert STRATEGY_DISPLAY_NAME in comparison.columns
    assert not metric_table.empty
    assert REPORT_DISPLAY_NAME in memo
    assert len(alerts) >= 1
    assert "Desk Tape" in tape_html
    assert "Institutional Workstation" in workstation_html
    compact_workstation_html = workstation_html.replace("\n", "")
    assert '<div class="workstation-grid"><div class="workstation-cell">' in compact_workstation_html


def test_dark_theme_preset_available():
    assert set(THEME_PRESETS) == {"Institutional Emerald", "Midnight Graphite"}
    dark_palette = _active_palette("Midnight Graphite")
    assert dark_palette["mode"] == "dark"
    assert dark_palette["plot_bg"] != THEME_PRESETS["Institutional Emerald"]["plot_bg"]


def test_follow_streamlit_theme_scaffold_exists():
    assert _in_follow_streamlit_mode(AUTO_THEME_NAME)
    css_vars = _palette_var_block(THEME_PRESETS["Midnight Graphite"])
    config_text = Path(".streamlit/config.toml").read_text(encoding="utf-8")

    assert "--bg-left" in css_vars
    assert "[theme.dark]" in config_text
    assert 'primaryColor = "#3db8a2"' in config_text


def test_streamlit_chart_theme_palettes_have_required_lengths():
    config = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf-8"))
    theme = config["theme"]

    assert len(theme["chartSequentialColors"]) == 10
    assert len(theme["chartDivergingColors"]) == 10


def test_panel_header_markup_is_inline_html():
    markup = _panel_header_markup("Signal Opportunity Map", "Cross-section of momentum and volatility.")

    assert markup.startswith('<div class="panel-head">')
    assert '<div class="panel-title">Signal Opportunity Map</div>' in markup
    assert '<div class="panel-note">Cross-section of momentum and volatility.</div>' in markup
    assert "\n    <div" not in markup


def test_external_plot_title_is_removed():
    fig = px.line(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), x="x", y="y", title=None)
    styled = _apply_plot_style(fig, 320, external_title=True)
    layout = styled.to_plotly_json()["layout"]

    assert styled.layout.title.text is None
    assert layout.get("title") is None


def test_external_treemap_title_is_removed():
    fig = px.treemap(
        pd.DataFrame({"ticker": ["AAA", "BBB"], "weight": [0.6, 0.4]}),
        path=["ticker"],
        values="weight",
        title=None,
    )
    styled = _apply_plot_style(fig, 320, external_title=True)
    layout = styled.to_plotly_json()["layout"]

    assert styled.layout.title.text is None
    assert layout.get("title") is None


def test_external_gauge_title_is_removed():
    fig = _gauge_figure(
        "Risk Control Score",
        50,
        100,
        [([0, 35], "rgba(0, 0, 0, 0.1)"), ([35, 65], "rgba(0, 0, 0, 0.2)")],
        external_title=True,
    )
    layout = fig.to_plotly_json()["layout"]

    assert fig.layout.title.text is None
    assert layout.get("title") is None


def test_prepare_display_frame_normalizes_empty_values_and_long_text():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-02"), pd.NaT],
            "detail": ["undefined", "signal detail " + "x" * 120],
            "value": [1.23, float("nan")],
        }
    )
    prepared = _prepare_display_frame(frame)

    assert prepared.loc[0, "date"] == "2025-01-02"
    assert prepared.loc[0, "detail"] == DISPLAY_PLACEHOLDER
    assert prepared.loc[1, "detail"].endswith("...")
    assert len(prepared.loc[1, "detail"]) <= 88
