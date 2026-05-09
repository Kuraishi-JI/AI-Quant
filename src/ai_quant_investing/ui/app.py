from __future__ import annotations

from datetime import date
from html import escape
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ..core.data import DEFAULT_UNIVERSE, generate_demo_prices, load_prices_from_csv
from ..core.pipeline import PrototypeConfig, PrototypeResult, run_prototype
from ..explainability.analysis import (
    build_signal_narrative,
    feature_percentile_table,
    local_contributions,
    prediction_bucket_table,
    standardized_importance,
    what_if_curve,
)

WINDOW_LOOKBACKS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "Full": None}
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FEATURE_DASHBOARD_SET = [
    "mom_20d",
    "mom_60d",
    "vol_20d",
    "drawdown_60d",
    "beta_60d",
    "rsi_14",
    "market_mom_20d",
    "breadth_20d",
]

AUTO_THEME_NAME = "Follow Streamlit"
AUTO_LIGHT_THEME_NAME = "Institutional Emerald"
AUTO_DARK_THEME_NAME = "Midnight Graphite"
DEFAULT_THEME_NAME = AUTO_THEME_NAME
DISPLAY_PLACEHOLDER = "--"
TABLE_TEXT_MAX_CHARS = 88
APP_DISPLAY_NAME = "Responsible Quant Advisory Workbench"
TEAM_DISPLAY_NAME = "HKUST(GZ) SOCH 5000 Team 1"
STRATEGY_DISPLAY_NAME = "Workbench Strategy"
REPORT_DISPLAY_NAME = f"{APP_DISPLAY_NAME} Memo"
REFERENCE_MARKET_LABEL = "Reference synthetic market"
UPLOAD_PRICE_LABEL = "Upload price CSV"
RUN_ACTION_LABEL = "Run Analysis"
EXPORT_FILE_PREFIX = "advisory_workbench"

THEME_PRESETS = {
    "Institutional Emerald": {
        "mode": "light",
        "bg_left": "rgba(15, 98, 83, 0.08)",
        "bg_right": "rgba(168, 99, 44, 0.09)",
        "bg_start": "#f7f4ee",
        "bg_end": "#f3f7f5",
        "sidebar_start": "rgba(246, 243, 236, 0.98)",
        "sidebar_end": "rgba(240, 245, 243, 0.98)",
        "hero_start": "rgba(18, 102, 79, 0.95)",
        "hero_end": "rgba(13, 62, 71, 0.97)",
        "accent": "#12664f",
        "accent_dark": "#0d3e47",
        "accent_soft": "#d4b483",
        "series_secondary": "#b86b32",
        "series_tertiary": "#6f8fa1",
        "danger": "#b22234",
        "warning": "#d4b483",
        "heading": "#0d3e47",
        "text": "#1d3b40",
        "muted": "#64777c",
        "surface": "rgba(255, 255, 255, 0.78)",
        "surface_soft": "rgba(255, 255, 255, 0.68)",
        "surface_strong": "rgba(255, 255, 255, 0.92)",
        "surface_alt": "rgba(248, 250, 249, 0.82)",
        "note_bg": "rgba(255, 255, 255, 0.68)",
        "chip_bg": "rgba(18, 102, 79, 0.08)",
        "hero_chip_bg": "rgba(255, 255, 255, 0.14)",
        "border": "rgba(13, 62, 71, 0.08)",
        "border_soft": "rgba(13, 62, 71, 0.06)",
        "grid": "rgba(13, 62, 71, 0.08)",
        "plot_bg": "rgba(255, 255, 255, 0.72)",
        "legend_bg": "rgba(255, 255, 255, 0.72)",
        "hover_bg": "rgba(255, 255, 255, 0.96)",
        "control_bg": "rgba(255, 255, 255, 0.88)",
        "shadow": "0 12px 28px rgba(12, 49, 53, 0.06)",
        "gauge_safe": "rgba(18, 102, 79, 0.18)",
        "gauge_warn": "rgba(212, 180, 131, 0.24)",
        "gauge_danger": "rgba(178, 34, 52, 0.20)",
        "gauge_neutral": "rgba(13, 62, 71, 0.16)",
        "heat_low": "#b22234",
        "heat_mid": "#f4efe3",
        "heat_high": "#12664f",
        "sequential_low": "#d8c2a4",
        "sequential_high": "#12664f",
    },
    "Midnight Graphite": {
        "mode": "dark",
        "bg_left": "rgba(21, 109, 138, 0.22)",
        "bg_right": "rgba(48, 160, 138, 0.14)",
        "bg_start": "#09131a",
        "bg_end": "#101c24",
        "sidebar_start": "rgba(10, 17, 23, 0.98)",
        "sidebar_end": "rgba(15, 25, 33, 0.98)",
        "hero_start": "rgba(15, 88, 116, 0.95)",
        "hero_end": "rgba(12, 30, 43, 0.98)",
        "accent": "#3db8a2",
        "accent_dark": "#163847",
        "accent_soft": "#98c7c0",
        "series_secondary": "#f1a65b",
        "series_tertiary": "#7cb7dc",
        "danger": "#e16b73",
        "warning": "#d8b46b",
        "heading": "#d5f0ea",
        "text": "#e7f0f2",
        "muted": "#9ab0bb",
        "surface": "rgba(14, 24, 31, 0.78)",
        "surface_soft": "rgba(18, 29, 37, 0.74)",
        "surface_strong": "rgba(18, 30, 39, 0.94)",
        "surface_alt": "rgba(11, 19, 25, 0.88)",
        "note_bg": "rgba(14, 24, 31, 0.72)",
        "chip_bg": "rgba(61, 184, 162, 0.12)",
        "hero_chip_bg": "rgba(255, 255, 255, 0.10)",
        "border": "rgba(130, 165, 184, 0.18)",
        "border_soft": "rgba(130, 165, 184, 0.12)",
        "grid": "rgba(130, 165, 184, 0.12)",
        "plot_bg": "rgba(12, 20, 27, 0.92)",
        "legend_bg": "rgba(12, 20, 27, 0.88)",
        "hover_bg": "rgba(21, 32, 41, 0.96)",
        "control_bg": "rgba(11, 18, 24, 0.96)",
        "shadow": "0 16px 34px rgba(0, 0, 0, 0.30)",
        "gauge_safe": "rgba(61, 184, 162, 0.20)",
        "gauge_warn": "rgba(216, 180, 107, 0.24)",
        "gauge_danger": "rgba(225, 107, 115, 0.22)",
        "gauge_neutral": "rgba(124, 183, 220, 0.18)",
        "heat_low": "#d96b72",
        "heat_mid": "#223643",
        "heat_high": "#38aa82",
        "sequential_low": "#5d7181",
        "sequential_high": "#3db8a2",
    },
}


def _pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else DISPLAY_PLACEHOLDER


def _num(value: float) -> str:
    return f"{value:.3f}" if pd.notna(value) else DISPLAY_PLACEHOLDER


def _signed_pct(value: float) -> str:
    return f"{value:+.2%}" if pd.notna(value) else DISPLAY_PLACEHOLDER


def _date_label(value: pd.Timestamp | str | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return DISPLAY_PLACEHOLDER
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _market_state_label(state: str | None) -> str:
    labels = {
        "risk_on": "Risk-on",
        "risk_off": "Risk-off",
        "transition": "Transition",
    }
    return labels.get(state or "", DISPLAY_PLACEHOLDER)


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _truncate_display_text(text: str, max_chars: int = TABLE_TEXT_MAX_CHARS) -> str:
    compact = " ".join(str(text).split())
    if not compact or compact.lower() in {"nan", "none", "nat", "undefined"}:
        return DISPLAY_PLACEHOLDER
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _format_table_timestamp(value: object) -> str:
    if _is_missing_scalar(value):
        return DISPLAY_PLACEHOLDER
    timestamp = pd.Timestamp(value)
    if timestamp == timestamp.normalize():
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _display_cell(value: object, *, max_chars: int = TABLE_TEXT_MAX_CHARS) -> object:
    if _is_missing_scalar(value):
        return DISPLAY_PLACEHOLDER
    if isinstance(value, (pd.Timestamp, np.datetime64, date)):
        return _format_table_timestamp(value)
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, str):
        return _truncate_display_text(value, max_chars=max_chars)
    if isinstance(value, (list, tuple, set, dict)):
        return _truncate_display_text(str(value), max_chars=max_chars)
    return value


def _prepare_display_frame(frame: pd.DataFrame, *, max_text: int = TABLE_TEXT_MAX_CHARS) -> pd.DataFrame:
    display = frame.copy()
    if isinstance(display.index, pd.DatetimeIndex):
        display.index = display.index.map(_format_table_timestamp)
    elif display.index.dtype == "object":
        display.index = display.index.map(lambda value: _display_cell(value, max_chars=max_text))

    for column in display.columns:
        series = display[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            display[column] = series.map(_format_table_timestamp)
        elif series.dtype == "object":
            display[column] = series.map(lambda value: _display_cell(value, max_chars=max_text))
    return display


def _styled_frame(
    frame: pd.DataFrame,
    formats: str | dict[str, str] | None = None,
    *,
    max_text: int = TABLE_TEXT_MAX_CHARS,
):
    prepared = _prepare_display_frame(frame, max_text=max_text)
    styler = prepared.style
    if formats is None:
        return styler.format(na_rep=DISPLAY_PLACEHOLDER)
    return styler.format(formats, na_rep=DISPLAY_PLACEHOLDER)


@st.cache_data(show_spinner=False)
def _cached_demo_prices(start_date: date, end_date: date, seed: int) -> pd.DataFrame:
    return generate_demo_prices(str(start_date), str(end_date), seed=seed, tickers=DEFAULT_UNIVERSE)


def _model_for_signal_date(result: PrototypeResult, signal_date: pd.Timestamp):
    eligible = [ts for ts in result.model_snapshots if ts <= signal_date]
    if not eligible:
        return result.latest_model
    return result.model_snapshots[max(eligible)]


def _prediction_quality_metrics(predictions: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    frame = predictions[["prediction", "target"]].dropna().copy()
    if frame.empty:
        return {}, pd.DataFrame()

    residual = frame["target"] - frame["prediction"]
    metrics = {
        "rank_ic": float(frame["prediction"].rank().corr(frame["target"].rank())),
        "sign_hit_rate": float((np.sign(frame["prediction"]) == np.sign(frame["target"])).mean()),
        "mae": float(residual.abs().mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
    }

    bucket_table = prediction_bucket_table(frame, bucket_count=5)
    if len(bucket_table) >= 2:
        metrics["top_minus_bottom"] = float(
            bucket_table["avg_realized_return"].iloc[-1] - bucket_table["avg_realized_return"].iloc[0]
        )
    else:
        metrics["top_minus_bottom"] = float("nan")
    return metrics, bucket_table


def _shock_table(latest_weights: pd.Series) -> pd.DataFrame:
    if latest_weights.empty:
        return pd.DataFrame(columns=["uniform_market_shock", "approx_portfolio_pnl", "net_exposure"])

    exposure = float(latest_weights.sum())
    rows = []
    for shock in (-0.03, -0.05, -0.08, -0.10, -0.15):
        rows.append(
            {
                "uniform_market_shock": f"{shock:.0%}",
                "approx_portfolio_pnl": exposure * shock,
                "net_exposure": exposure,
            }
        )
    return pd.DataFrame(rows)


def _importance_frame(result: PrototypeResult) -> pd.DataFrame:
    predictions = result.predictions
    eval_frame = predictions.sample(2500, random_state=42) if len(predictions) > 2500 else predictions
    return standardized_importance(
        result.latest_model,
        eval_frame[result.feature_columns],
        eval_frame["target"],
        result.feature_columns,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _compound_return(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float((1.0 + clean).prod() - 1.0)


def _slice_indexed(frame: pd.DataFrame | pd.Series, window_label: str):
    lookback = WINDOW_LOOKBACKS.get(window_label)
    if lookback is None or len(frame) <= lookback:
        return frame.copy()
    return frame.tail(lookback).copy()


def _in_follow_streamlit_mode(theme_name: str | None = None) -> bool:
    name = theme_name or st.session_state.get("interface_theme", DEFAULT_THEME_NAME)
    return name == AUTO_THEME_NAME


def _active_palette(theme_name: str | None = None) -> dict[str, str]:
    name = theme_name or st.session_state.get("interface_theme", DEFAULT_THEME_NAME)
    if name == AUTO_THEME_NAME:
        name = AUTO_LIGHT_THEME_NAME
    return THEME_PRESETS.get(name, THEME_PRESETS[AUTO_LIGHT_THEME_NAME])


def _palette_var_block(palette: dict[str, str]) -> str:
    return "\n".join(
        [
            f"            --bg-left: {palette['bg_left']};",
            f"            --bg-right: {palette['bg_right']};",
            f"            --bg-start: {palette['bg_start']};",
            f"            --bg-end: {palette['bg_end']};",
            f"            --hero-start: {palette['hero_start']};",
            f"            --hero-end: {palette['hero_end']};",
            f"            --accent-color: {palette['accent']};",
            f"            --accent-dark: {palette['accent_dark']};",
            f"            --accent-soft: {palette['accent_soft']};",
            f"            --danger-color: {palette['danger']};",
            f"            --heading-color: {palette['heading']};",
            f"            --text-color: {palette['text']};",
            f"            --muted-color: {palette['muted']};",
            f"            --surface-bg: {palette['surface']};",
            f"            --surface-soft: {palette['surface_soft']};",
            f"            --surface-strong: {palette['surface_strong']};",
            f"            --surface-alt: {palette['surface_alt']};",
            f"            --sidebar-start: {palette['sidebar_start']};",
            f"            --sidebar-end: {palette['sidebar_end']};",
            f"            --border-color: {palette['border']};",
            f"            --border-soft: {palette['border_soft']};",
            f"            --grid-color: {palette['grid']};",
            f"            --control-bg: {palette['control_bg']};",
            f"            --note-bg: {palette['note_bg']};",
            f"            --chip-bg: {palette['chip_bg']};",
            f"            --hero-chip-bg: {palette['hero_chip_bg']};",
            f"            --shadow-lg: {palette['shadow']};",
        ]
    )


def _mount_streamlit_theme_bridge() -> None:
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;

        const parseRgb = (value) => {
          const match = String(value || "").match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/i);
          return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : null;
        };

        const applyMode = () => {
          const header = parentDoc.querySelector('header[data-testid="stHeader"]');
          const target = header || parentDoc.body;
          const bg = window.parent.getComputedStyle(target).backgroundColor;
          const rgb = parseRgb(bg);
          if (!rgb) return;
          const [r, g, b] = rgb;
          const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          parentDoc.documentElement.setAttribute("data-aq-mode", luminance < 140 ? "dark" : "light");
        };

        applyMode();
        const observer = new MutationObserver(() => applyMode());
        observer.observe(parentDoc.body, { attributes: true, childList: true, subtree: true });
        window.setInterval(applyMode, 900);
        </script>
        """,
        height=0,
        width=0,
    )


def _theme_scale() -> list[str]:
    palette = _active_palette()
    return [palette["heat_low"], palette["heat_mid"], palette["heat_high"]]


def _clear_internal_plot_title(fig):
    fig.update_layout(title=None, title_text=None)
    fig.layout.title = None
    return fig


def _apply_plot_style(fig, height: int, *, external_title: bool = False):
    palette = _active_palette()
    follow_streamlit = _in_follow_streamlit_mode()
    base_layout = dict(
        height=height,
        margin=dict(l=10, r=16, t=18 if external_title else 52, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if external_title:
        base_layout["title"] = None
    else:
        base_layout["title"] = dict(x=0.02, xanchor="left")
    if follow_streamlit:
        follow_layout = dict(
            **base_layout,
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family='"Aptos, "Segoe UI", "Helvetica Neue", sans-serif', size=11),
            hoverlabel=dict(font_size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=10)),
        )
        if not external_title:
            follow_layout["title_font"] = dict(
                family='"Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif',
                size=15,
            )
        fig.update_layout(**follow_layout)
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(127, 140, 152, 0.18)",
            zeroline=False,
            tickfont=dict(size=10),
            automargin=True,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(127, 140, 152, 0.18)",
            zeroline=False,
            tickfont=dict(size=10),
            automargin=True,
        )
    else:
        themed_layout = dict(
            **base_layout,
            plot_bgcolor=palette["plot_bg"],
            font=dict(family='"Aptos, "Segoe UI", "Helvetica Neue", sans-serif', size=11, color=palette["text"]),
            hoverlabel=dict(bgcolor=palette["hover_bg"], font_size=11, font_color=palette["text"]),
            legend=dict(
                bgcolor=palette["legend_bg"],
                bordercolor=palette["border_soft"],
                borderwidth=1,
                font=dict(size=10),
            ),
        )
        if not external_title:
            themed_layout["title_font"] = dict(
                family='"Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif',
                size=15,
                color=palette["heading"],
            )
        fig.update_layout(**themed_layout)
        fig.update_xaxes(
            showgrid=True,
            gridcolor=palette["grid"],
            zeroline=False,
            tickfont=dict(color=palette["muted"], size=10),
            automargin=True,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=palette["grid"],
            zeroline=False,
            tickfont=dict(color=palette["muted"], size=10),
            automargin=True,
        )
    if external_title:
        _clear_internal_plot_title(fig)
    return fig


def _panel_columns(spec, *, gap: str = "large"):
    return st.columns(spec, gap=gap, vertical_alignment="top")


PANEL_HEIGHT_PRESETS = {
    "compact": {"panel": 330, "chart": 242, "table_min": 190, "table_max": 250},
    "dense": {"panel": 398, "chart": 310, "table_min": 228, "table_max": 308},
    "small": {"panel": 450, "chart": 360, "table_min": 240, "table_max": 330},
    "medium": {"panel": 505, "chart": 410, "table_min": 295, "table_max": 395},
    "large": {"panel": 540, "chart": 450, "table_min": 330, "table_max": 430},
}


def _panel_preset(size: str) -> dict[str, int]:
    return PANEL_HEIGHT_PRESETS.get(size, PANEL_HEIGHT_PRESETS["medium"])


def _table_height(
    row_count: int,
    *,
    min_height: int = 180,
    max_height: int = 520,
    row_px: int = 36,
    chrome_px: int = 54,
) -> int:
    rows = max(1, int(row_count))
    return int(max(min_height, min(max_height, chrome_px + rows * row_px)))


def _panel_chart_height(size: str) -> int:
    return _panel_preset(size)["chart"]


def _panel_table_height(
    row_count: int,
    size: str,
    *,
    min_height: int | None = None,
    max_height: int | None = None,
    row_px: int = 36,
    chrome_px: int = 54,
) -> int:
    preset = _panel_preset(size)
    return _table_height(
        row_count,
        min_height=min_height if min_height is not None else preset["table_min"],
        max_height=max_height if max_height is not None else preset["table_max"],
        row_px=row_px,
        chrome_px=chrome_px,
    )


def _panel_box(*, size: str):
    return st.container(height=_panel_preset(size)["panel"], border=False)


def _panel_header_markup(title: str, subtitle: str | None = None, *, eyebrow: str | None = None) -> str:
    parts = ['<div class="panel-head">']
    if eyebrow:
        parts.append(f'<div class="panel-eyebrow">{escape(eyebrow)}</div>')
    parts.append(f'<div class="panel-title">{escape(title)}</div>')
    parts.append(f'<div class="panel-note">{escape(subtitle) if subtitle else "&nbsp;"}</div>')
    parts.append("</div>")
    return "".join(parts)


def _panel_header(title: str, subtitle: str | None = None, *, eyebrow: str | None = None) -> None:
    st.markdown(_panel_header_markup(title, subtitle, eyebrow=eyebrow), unsafe_allow_html=True)


def _capture_metrics(performance: pd.DataFrame) -> dict[str, float]:
    strat = performance["strategy_return"]
    bench = performance["benchmark_return"]
    active = strat - bench
    up_mask = bench > 0.0
    down_mask = bench < 0.0
    return {
        "up_capture": _safe_div(_compound_return(strat[up_mask]), _compound_return(bench[up_mask])),
        "down_capture": _safe_div(_compound_return(strat[down_mask]), _compound_return(bench[down_mask])),
        "tracking_error": float(active.std() * np.sqrt(252)),
        "information_ratio": _safe_div(active.mean() * np.sqrt(252), active.std()),
        "active_return": _compound_return(active),
    }


def _risk_posture(result: PrototypeResult) -> dict[str, float | str]:
    metrics = result.metrics
    config = result.config

    vol_ratio = _safe_div(metrics.get("annual_volatility", 0.0), config.target_vol)
    drawdown_ratio = _safe_div(abs(metrics.get("max_drawdown", 0.0)), config.drawdown_limit * 1.35)
    var_ratio = _safe_div(metrics.get("var_95", 0.0), config.var_limit)

    vol_ratio = 0.0 if not np.isfinite(vol_ratio) else vol_ratio
    drawdown_ratio = 0.0 if not np.isfinite(drawdown_ratio) else drawdown_ratio
    var_ratio = 0.0 if not np.isfinite(var_ratio) else var_ratio

    vol_component = 35.0 * min(1.3, vol_ratio)
    drawdown_component = 30.0 * min(1.3, drawdown_ratio)
    var_component = 20.0 * min(1.3, var_ratio)
    overlay_component = 15.0 * min(1.0, len(result.risk_events.tail(30)) / 12.0)
    score = float(max(0.0, min(100.0, vol_component + drawdown_component + var_component + overlay_component)))

    if score < 35:
        label = "Contained"
    elif score < 65:
        label = "Elevated"
    else:
        label = "Stressed"
    return {"score": score, "label": label}


def _gauge_figure(
    title: str,
    value: float,
    max_value: float,
    steps: list[tuple[list[float], str]],
    suffix: str = "",
    *,
    external_title: bool = False,
) -> go.Figure:
    palette = _active_palette()
    follow_streamlit = _in_follow_streamlit_mode()
    indicator_kwargs = dict(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, max_value], "tickwidth": 1, "tickcolor": palette["muted"]},
            "bar": {"color": palette["accent"]},
            "borderwidth": 0,
            "bgcolor": "rgba(255,255,255,0.0)",
            "steps": [{"range": rng, "color": color} for rng, color in steps],
        },
    )
    if not external_title:
        indicator_kwargs["title"] = {"text": title, "font": {"size": 16}}
    fig = go.Figure(
        go.Indicator(**indicator_kwargs)
    )
    fig.update_layout(
        height=240,
        margin=dict(l=18, r=18, t=16 if external_title else 48, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='"Aptos, "Segoe UI", "Helvetica Neue", sans-serif',
            **({} if follow_streamlit else {"color": palette["text"]}),
        ),
    )
    if external_title:
        _clear_internal_plot_title(fig)
    return fig


def _monthly_return_pivot(performance: pd.DataFrame) -> pd.DataFrame:
    monthly = performance["strategy_return"].resample("M").apply(_compound_return)
    monthly_frame = monthly.to_frame("monthly_return")
    monthly_frame["year"] = monthly_frame.index.year
    monthly_frame["month"] = monthly_frame.index.strftime("%b")
    pivot = monthly_frame.pivot(index="year", columns="month", values="monthly_return")
    return pivot.reindex(columns=MONTH_ORDER).sort_index(ascending=False)


def _yearly_summary(performance: pd.DataFrame) -> pd.DataFrame:
    frame = performance.copy()
    frame["year"] = frame.index.year
    summary = (
        frame.groupby("year", sort=False)
        .agg(
            strategy_return=("strategy_return", _compound_return),
            benchmark_return=("benchmark_return", _compound_return),
            annual_vol=("strategy_return", lambda x: float(x.std() * np.sqrt(252))),
            max_drawdown=("drawdown", "min"),
            avg_turnover=("turnover", "mean"),
            avg_exposure=("exposure", "mean"),
        )
        .reset_index()
    )
    summary["active_return"] = summary["strategy_return"] - summary["benchmark_return"]
    return summary.sort_values("year", ascending=False)


def _asset_contribution_summary(result: PrototypeResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = result.decision_log.copy()
    frame["daily_pnl_contribution"] = frame["final_weight"] * frame["target"]
    summary = (
        frame.groupby("ticker", sort=False)
        .agg(
            total_pnl_contribution=("daily_pnl_contribution", "sum"),
            avg_weight=("final_weight", "mean"),
            avg_prediction=("prediction", "mean"),
            avg_realized=("target", "mean"),
            hit_rate=("target", lambda x: float((x > 0.0).mean())),
            selection_rate=("selected", lambda x: float(x.mean())),
        )
        .reset_index()
        .sort_values("total_pnl_contribution", ascending=False)
    )

    daily = (
        frame.groupby(["realized_date", "ticker"], sort=True)["daily_pnl_contribution"]
        .sum()
        .reset_index()
        .rename(columns={"realized_date": "date"})
    )
    return summary, daily


def _latest_feature_exposure(result: PrototypeResult) -> pd.DataFrame:
    latest_signal = pd.Timestamp(result.decision_log["signal_date"].max())
    frame = result.decision_log[result.decision_log["signal_date"] == latest_signal].copy()
    if frame.empty:
        return pd.DataFrame(columns=["feature", "weighted_exposure", "exposure_zscore"])

    weights = frame["final_weight"].fillna(0.0)
    if float(weights.sum()) <= 1e-12:
        weights = frame["pre_drawdown_weight"].fillna(0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return pd.DataFrame(columns=["feature", "weighted_exposure", "exposure_zscore"])

    records = []
    for feature in FEATURE_DASHBOARD_SET:
        if feature not in result.feature_columns:
            continue
        exposure = float(np.average(frame[feature].astype(float), weights=weights))
        history = result.panel[feature].dropna()
        std = float(history.std()) if not history.empty else float("nan")
        zscore = 0.0 if not np.isfinite(std) or abs(std) < 1e-12 else (exposure - float(history.median())) / std
        records.append(
            {
                "feature": feature,
                "weighted_exposure": exposure,
                "exposure_zscore": zscore,
            }
        )
    return pd.DataFrame(records)


def _correlation_matrix(result: PrototypeResult, window_label: str = "1Y") -> pd.DataFrame:
    returns = _slice_indexed(result.returns.dropna(how="all"), window_label)
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


def _strategy_comparison_frame(result: PrototypeResult, benchmark: str) -> pd.DataFrame:
    idx = result.performance.index
    returns = result.returns.reindex(idx).fillna(0.0)

    comparison = pd.DataFrame(index=idx)
    comparison[STRATEGY_DISPLAY_NAME] = result.performance["strategy_return"]
    comparison[benchmark] = result.performance["benchmark_return"]
    comparison["Equal Weight Universe"] = returns.mean(axis=1)

    topk_equal = (
        result.decision_log[result.decision_log["selected"]]
        .groupby("realized_date", sort=True)["target"]
        .mean()
        .reindex(idx)
        .fillna(0.0)
    )
    comparison["Top-K Equal Weight"] = topk_equal

    equity = (1.0 + comparison).cumprod()
    equity.columns = [f"{name} Equity" for name in equity.columns]
    return comparison.join(equity)


def _series_metric_table(comparison_returns: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in comparison_returns.columns:
        series = comparison_returns[column].dropna()
        if series.empty:
            continue
        equity = (1.0 + series).cumprod()
        drawdown = equity.div(equity.cummax()) - 1.0
        records.append(
            {
                "strategy": column,
                "total_return": _compound_return(series),
                "annual_vol": float(series.std() * np.sqrt(252)),
                "sharpe": _safe_div(series.mean() * np.sqrt(252), series.std()),
                "max_drawdown": float(drawdown.min()),
                "win_rate": float((series > 0.0).mean()),
            }
        )
    return pd.DataFrame(records).sort_values("total_return", ascending=False)


def _build_alert_feed(result: PrototypeResult, benchmark: str) -> list[dict[str, str]]:
    posture = _risk_posture(result)
    latest_perf = result.performance.iloc[-1]
    latest_signal_date = pd.Timestamp(result.decision_log["signal_date"].max())
    latest_signal = (
        result.decision_log[result.decision_log["signal_date"] == latest_signal_date]
        .sort_values("prediction", ascending=False)
        .head(1)
    )
    alerts: list[dict[str, str]] = []

    alerts.append(
        {
            "level": "Risk",
            "headline": f"Portfolio posture is {posture['label']}",
            "detail": f"Risk score {posture['score']:.0f}/100 with max drawdown {_pct(result.metrics.get('max_drawdown', float('nan')))}.",
        }
    )

    if result.market_context.get("market_state") == "risk_off":
        alerts.append(
            {
                "level": "Macro",
                "headline": f"{benchmark} regime flagged risk-off",
                "detail": f"20D market vol {_pct(result.market_context.get('market_vol_20d', float('nan')))} and breadth {_pct(result.market_context.get('breadth_20d', float('nan')))}.",
            }
        )

    if float(latest_perf["cash_weight"]) >= 0.45:
        alerts.append(
            {
                "level": "Exposure",
                "headline": "Cash buffer has expanded materially",
                "detail": f"Latest cash weight is {_pct(float(latest_perf['cash_weight']))}, signaling a defensive stance.",
            }
        )

    if not latest_signal.empty:
        top_row = latest_signal.iloc[0]
        alerts.append(
            {
                "level": "Signal",
                "headline": f"Top current conviction: {top_row['ticker']}",
                "detail": f"Predicted next-day return {_pct(float(top_row['prediction']))} with final weight {_pct(float(top_row['final_weight']))}.",
            }
        )

    if not result.risk_events.empty:
        event = result.risk_events.iloc[-1]
        alerts.append(
            {
                "level": "Overlay",
                "headline": f"Latest overlay event: {event['event']}",
                "detail": str(event["detail"]),
            }
        )

    return alerts[:4]


def _build_report_markdown(result: PrototypeResult, benchmark: str) -> str:
    latest_alloc = result.applied_weights.iloc[-1]
    latest_alloc = latest_alloc[latest_alloc > 0.0001].sort_values(ascending=False).head(5)
    importance = _importance_frame(result).head(5)
    posture = _risk_posture(result)
    alerts = _build_alert_feed(result, benchmark)

    lines = [
        f"# {REPORT_DISPLAY_NAME}",
        "",
        "## Snapshot",
        f"- Benchmark: `{benchmark}`",
        f"- Market state: `{_market_state_label(result.market_context.get('market_state'))}`",
        f"- Risk posture: `{posture['label']}` ({posture['score']:.0f}/100)",
        f"- Annual return: `{_pct(result.metrics.get('annual_return', float('nan')))}`",
        f"- Sharpe: `{_num(result.metrics.get('sharpe_ratio', float('nan')))}`",
        f"- Max drawdown: `{_pct(result.metrics.get('max_drawdown', float('nan')))}`",
        "",
        "## Current Allocation",
    ]
    for ticker, weight in latest_alloc.items():
        lines.append(f"- {ticker}: `{weight:.1%}`")

    lines.extend(["", "## Top Feature Drivers"])
    for _, row in importance.iterrows():
        lines.append(
            f"- {row['feature']}: permutation `{row['permutation_importance']:.5f}`, abs coefficient `{row['abs_coefficient']:.5f}`"
        )

    lines.extend(["", "## Alert Feed"])
    for alert in alerts:
        lines.append(f"- [{alert['level']}] {alert['headline']}: {alert['detail']}")

    return "\n".join(lines)


def _set_page_config() -> None:
    st.set_page_config(
        page_title=APP_DISPLAY_NAME,
        page_icon="RQ",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _render_theme(theme_name: str) -> None:
    follow_streamlit = _in_follow_streamlit_mode(theme_name)
    light_palette = THEME_PRESETS[AUTO_LIGHT_THEME_NAME] if follow_streamlit else _active_palette(theme_name)
    dark_palette = THEME_PRESETS[AUTO_DARK_THEME_NAME] if follow_streamlit else light_palette
    root_vars = _palette_var_block(light_palette)
    dark_vars = _palette_var_block(dark_palette)
    dark_css_block = f"html[data-aq-mode='dark'] {{\n{dark_vars}\n        }}" if follow_streamlit else ""
    st.markdown(
        f"""
        <style>
        :root {{
{root_vars}
        }}
        {dark_css_block}
        html, body, [class*="css"] {{
            font-family: Aptos, "Segoe UI", "Helvetica Neue", sans-serif;
        }}
        .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: var(--text-color);
        }}
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption {{
            color: var(--text-color);
        }}
        .stApp {{
            background:
                radial-gradient(circle at top left, var(--bg-left), transparent 34%),
                radial-gradient(circle at top right, var(--bg-right), transparent 28%),
                linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
        }}
        [data-testid="stAppViewContainer"] {{
            background: transparent;
        }}
        section[data-testid="stSidebar"] {{
            background:
                radial-gradient(circle at top, var(--bg-right), transparent 24%),
                linear-gradient(180deg, var(--sidebar-start), var(--sidebar-end));
            border-right: 1px solid var(--border-color);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--text-color);
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        .stDateInput > div > div,
        .stNumberInput > div > div,
        .stTextInput > div > div,
        .stMultiSelect > div > div {{
            background: var(--control-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
        }}
        input, textarea {{
            color: var(--text-color) !important;
            -webkit-text-fill-color: var(--text-color) !important;
        }}
        div[role="listbox"] {{
            background: var(--control-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
        }}
        div[role="option"] {{
            color: var(--text-color);
        }}
        div[role="option"][aria-selected="true"] {{
            background: var(--chip-bg);
        }}
        .stButton > button,
        .stDownloadButton > button {{
            background: linear-gradient(135deg, var(--accent-color), var(--accent-dark));
            color: white !important;
            border: 1px solid transparent;
            border-radius: 0.9rem;
            box-shadow: var(--shadow-lg);
        }}
        .stButton > button *,
        .stDownloadButton > button * {{
            color: white !important;
            -webkit-text-fill-color: white !important;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border: 1px solid var(--border-color);
            filter: brightness(1.04);
        }}
        .stSlider [data-baseweb="slider"] > div > div {{
            background: var(--accent-color);
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-soft);
            padding: 0.96rem 1.02rem 0.9rem 1.02rem;
            border-radius: 1.05rem;
            box-shadow: var(--shadow-lg);
            animation: fadeInUp 0.5s ease both;
            position: relative;
            overflow: hidden;
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease;
        }}
        div[data-testid="stMetric"]::before,
        .mini-card::before,
        .status-card::before,
        .guide-card::before,
        .workstation-cell::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-color), var(--accent-soft));
            opacity: 0.95;
        }}
        div[data-testid="stMetric"]:hover,
        .mini-card:hover,
        .status-card:hover,
        .guide-card:hover,
        .workstation-cell:hover {{
            transform: translateY(-2px);
            border-color: var(--border-color);
            box-shadow: 0 18px 36px rgba(8, 22, 32, 0.16);
        }}
        div[data-testid="stMetricLabel"] {{
            color: var(--muted-color);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--heading-color);
            line-height: 1.02;
        }}
        div[data-testid="stMetricDelta"] {{
            color: var(--text-color);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.42rem;
            background: var(--surface-soft);
            border: 1px solid var(--border-soft);
            padding: 0.32rem;
            border-radius: 1.1rem;
            box-shadow: var(--shadow-lg);
            overflow-x: auto;
            scrollbar-width: none;
        }}
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
            display: none;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            height: auto;
            min-height: 2.48rem;
            padding: 0.48rem 0.9rem;
            font-size: 0.95rem;
            color: var(--heading-color);
            font-weight: 600;
            letter-spacing: 0.01em;
            border: 1px solid transparent;
            background: transparent;
            opacity: 0.9;
            white-space: nowrap;
            transition:
                background 0.18s ease,
                border-color 0.18s ease,
                color 0.18s ease,
                transform 0.18s ease,
                box-shadow 0.18s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: var(--chip-bg);
            border-color: var(--border-color);
            color: var(--heading-color) !important;
            transform: translateY(-1px);
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, var(--accent-color), var(--accent-dark));
            color: white !important;
            border-color: rgba(255, 255, 255, 0.08);
            box-shadow: var(--shadow-lg);
            opacity: 1;
        }}
        .stTabs [aria-selected="true"] *,
        .stTabs [aria-selected="true"] p,
        .stTabs [aria-selected="true"] span,
        .stTabs [aria-selected="true"] div {{
            color: white !important;
            -webkit-text-fill-color: white !important;
        }}
        .hero {{
            position: relative;
            overflow: hidden;
            padding: 1.22rem 1.28rem;
            border-radius: 1.08rem;
            background: linear-gradient(135deg, var(--hero-start), var(--hero-end));
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: var(--shadow-lg);
            margin-bottom: 0.62rem;
            animation: fadeInUp 0.55s ease both;
        }}
        .hero::before {{
            content: "";
            position: absolute;
            top: -18%;
            right: -10%;
            width: 340px;
            height: 340px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.02) 48%, transparent 72%);
            pointer-events: none;
        }}
        .hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(255,255,255,0.03) 0%, transparent 22%, transparent 78%, rgba(255,255,255,0.03) 100%),
                repeating-linear-gradient(
                    180deg,
                    transparent 0px,
                    transparent 30px,
                    rgba(255,255,255,0.025) 31px,
                    transparent 32px
                );
            pointer-events: none;
        }}
        .hero-grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(250px, 0.78fr);
            gap: 1rem;
            align-items: stretch;
        }}
        .hero-copy {{
            min-width: 0;
        }}
        .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.62rem;
            padding: 0.24rem 0.62rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.14);
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.92);
        }}
        .hero-kicker::before {{
            content: "";
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: rgba(255,255,255,0.88);
            box-shadow: 0 0 0 4px rgba(255,255,255,0.12);
        }}
        .hero h1 {{
            font-size: clamp(2.05rem, 3.35vw, 3.1rem);
            line-height: 0.98;
            letter-spacing: -0.03em;
            margin: 0 0 0.4rem 0;
            color: white;
            max-width: 15ch;
        }}
        .hero p {{
            margin: 0;
            opacity: 0.94;
            line-height: 1.44;
            max-width: 46rem;
            color: white;
            font-size: 0.99rem;
        }}
        .chip-row {{
            margin-top: 0.82rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }}
        .caption-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.66rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.12);
            font-size: 0.82rem;
            font-weight: 600;
            color: white;
        }}
        .caption-chip::before {{
            content: "";
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: rgba(255,255,255,0.82);
        }}
        .hero-side {{
            position: relative;
            z-index: 1;
            display: grid;
            gap: 0.62rem;
            align-content: start;
        }}
        .hero-side-panel {{
            padding: 0.8rem 0.88rem;
            border-radius: 1rem;
            background: rgba(9, 22, 31, 0.22);
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
            backdrop-filter: blur(8px);
        }}
        .hero-side-kicker {{
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.72);
            margin-bottom: 0.38rem;
        }}
        .hero-side-value {{
            font-size: 1rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.18rem;
        }}
        .hero-side-copy {{
            color: rgba(255,255,255,0.88);
            line-height: 1.36;
            font-size: 0.84rem;
        }}
        .hero-stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.56rem;
        }}
        .hero-stat {{
            padding: 0.66rem 0.72rem;
            border-radius: 0.95rem;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.11);
        }}
        .hero-stat-label {{
            font-size: 0.66rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.68);
            margin-bottom: 0.24rem;
        }}
        .hero-stat-value {{
            font-size: 0.92rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.12rem;
        }}
        .hero-stat-copy {{
            font-size: 0.76rem;
            color: rgba(255,255,255,0.84);
            line-height: 1.28;
        }}
        .mini-card {{
            padding: 1.02rem 1rem 0.92rem 1rem;
            border-radius: 1rem;
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
            min-height: 8.4rem;
            animation: fadeInUp 0.6s ease both;
            position: relative;
            overflow: hidden;
        }}
        .mini-card-meta {{
            margin-bottom: 0.55rem;
        }}
        .alert-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.24rem 0.58rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-color);
            color: var(--heading-color);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .mini-card h3 {{
            font-size: 1.02rem;
            line-height: 1.24;
            margin-bottom: 0.32rem;
            color: var(--heading-color);
        }}
        .mini-card p {{
            font-size: 0.92rem;
            margin: 0;
            line-height: 1.45;
            color: var(--text-color);
        }}
        .status-card {{
            padding: 1.08rem 1.1rem 1rem 1.1rem;
            border-radius: 1.05rem;
            background: linear-gradient(180deg, var(--surface-strong), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
            min-height: 9rem;
            animation: fadeInUp 0.65s ease both;
            position: relative;
            overflow: hidden;
        }}
        .status-kicker {{
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted-color);
            margin-bottom: 0.55rem;
        }}
        .status-value {{
            font-size: 1.6rem;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
            color: var(--heading-color);
            margin-bottom: 0.3rem;
            line-height: 1.06;
        }}
        .status-sub {{
            font-size: 0.92rem;
            color: var(--text-color);
            line-height: 1.45;
        }}
        .status-card.compact {{
            padding: 0.82rem 0.88rem 0.8rem 0.88rem;
            border-radius: 0.95rem;
            min-height: 7.2rem;
            background: linear-gradient(180deg, var(--surface-soft), var(--surface-alt));
        }}
        .status-card.compact .status-kicker {{
            font-size: 0.7rem;
            margin-bottom: 0.4rem;
        }}
        .status-card.compact .status-value {{
            font-size: 1.08rem;
            line-height: 1.08;
            margin-bottom: 0.24rem;
        }}
        .status-card.compact .status-sub {{
            font-size: 0.84rem;
            line-height: 1.34;
        }}
        .guide-card {{
            padding: 1rem 1.02rem 0.96rem 1.02rem;
            border-radius: 1.05rem;
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
            min-height: 7.5rem;
            animation: fadeInUp 0.7s ease both;
            position: relative;
            overflow: hidden;
        }}
        .guide-card h4 {{
            margin: 0 0 0.35rem 0;
            font-size: 0.95rem;
            color: var(--heading-color);
        }}
        .guide-card p {{
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.45;
            color: var(--text-color);
        }}
        .brand-shell {{
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: 92px 1fr;
            gap: 0.88rem;
            align-items: center;
            padding: 0.8rem 0.92rem;
            margin-bottom: 0.52rem;
            border-radius: 1.04rem;
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
            animation: fadeInUp 0.45s ease both;
        }}
        .brand-shell::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-color), var(--accent-soft));
            opacity: 0.95;
        }}
        .brand-mark {{
            width: 74px;
            height: 74px;
            border-radius: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(145deg, var(--accent-color), var(--accent-dark));
            color: white;
            font-size: 1.7rem;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
            letter-spacing: 0.04em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 16px 34px rgba(13, 62, 71, 0.18);
        }}
        .brand-overline {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--muted-color);
            margin-bottom: 0.18rem;
        }}
        .brand-title {{
            font-size: 1.72rem;
            line-height: 1.02;
            color: var(--heading-color);
            margin-bottom: 0.14rem;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        }}
        .brand-subtitle {{
            font-size: 0.92rem;
            color: var(--text-color);
            line-height: 1.38;
            max-width: 50rem;
        }}
        .brand-pills {{
            margin-top: 0.42rem;
        }}
        .brand-pill {{
            display: inline-block;
            margin-right: 0.34rem;
            margin-bottom: 0.24rem;
            padding: 0.2rem 0.54rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-color);
            color: var(--heading-color);
            font-size: 0.78rem;
        }}
        .landing-note {{
            padding: 0.6rem 0.8rem;
            margin: 0.04rem 0 0.58rem 0;
            border-left: 3px solid var(--accent-color);
            border-radius: 0.9rem;
            background: var(--note-bg);
            color: var(--text-color);
            box-shadow: var(--shadow-lg);
            animation: fadeInUp 0.5s ease both;
            font-size: 0.88rem;
            line-height: 1.36;
        }}
        .section-head {{
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.1rem 0 0.8rem 0;
        }}
        .section-kicker {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted-color);
            margin-bottom: 0.28rem;
        }}
        .section-title {{
            font-size: 1.45rem;
            line-height: 1.08;
            color: var(--heading-color);
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        }}
        .section-copy {{
            margin-top: 0.25rem;
            color: var(--text-color);
            line-height: 1.48;
            max-width: 56rem;
        }}
        .section-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.36rem 0.72rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-color);
            color: var(--heading-color);
            font-size: 0.84rem;
            white-space: nowrap;
        }}
        .panel-head {{
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            min-height: 3.35rem;
            margin: 0 0 0.18rem 0;
            padding: 0 0.04rem;
        }}
        .panel-eyebrow {{
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted-color);
            margin-bottom: 0.14rem;
        }}
        .panel-title {{
            font-size: 1.12rem;
            line-height: 1.08;
            color: var(--heading-color);
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        }}
        .panel-note {{
            margin-top: 0.1rem;
            min-height: 0.92rem;
            color: var(--muted-color);
            line-height: 1.3;
            font-size: 0.8rem;
        }}
        .command-shell {{
            position: relative;
            overflow: hidden;
            padding: 0.9rem 0.96rem 0.96rem 0.96rem;
            margin: 0.06rem 0 1rem 0;
            border-radius: 1.05rem;
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
        }}
        .command-shell::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-color), var(--accent-soft));
            opacity: 0.95;
        }}
        .command-shell .section-head {{
            margin: 0 0 0.72rem 0;
        }}
        .command-shell .section-kicker {{
            font-size: 0.72rem;
            margin-bottom: 0.22rem;
        }}
        .command-shell .section-title {{
            font-size: 1.26rem;
            line-height: 1.04;
        }}
        .command-shell .section-copy {{
            margin-top: 0.16rem;
            max-width: 48rem;
            line-height: 1.38;
            font-size: 0.88rem;
        }}
        .command-shell .section-badge {{
            padding: 0.3rem 0.64rem;
            font-size: 0.79rem;
        }}
        .command-rail {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.72rem;
        }}
        .command-tag {{
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.56rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-color);
            color: var(--heading-color);
            font-size: 0.76rem;
        }}
        .command-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
        }}
        .alert-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.7rem;
        }}
        .mini-card.compact {{
            padding: 0.82rem 0.88rem 0.78rem 0.88rem;
            border-radius: 0.95rem;
            min-height: 6.45rem;
        }}
        .mini-card.compact .mini-card-meta {{
            margin-bottom: 0.42rem;
        }}
        .mini-card.compact .alert-badge {{
            padding: 0.2rem 0.5rem;
            font-size: 0.72rem;
        }}
        .mini-card.compact h3 {{
            font-size: 0.94rem;
            line-height: 1.18;
            margin-bottom: 0.24rem;
        }}
        .mini-card.compact p {{
            font-size: 0.84rem;
            line-height: 1.34;
        }}
        .workstation-shell {{
            position: relative;
            overflow: hidden;
            padding: 0.98rem 1.02rem 1.05rem 1.02rem;
            margin: 0.15rem 0 0.85rem 0;
            border-radius: 1.08rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01)),
                linear-gradient(180deg, var(--surface-strong), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
        }}
        .workstation-shell::after {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.03) 49%, transparent 100%),
                repeating-linear-gradient(
                    180deg,
                    transparent 0px,
                    transparent 28px,
                    rgba(255,255,255,0.025) 29px,
                    transparent 30px
                );
            pointer-events: none;
        }}
        .workstation-head {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            position: relative;
            z-index: 1;
        }}
        .workstation-overline {{
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted-color);
            margin-bottom: 0.18rem;
        }}
        .workstation-title {{
            font-size: 1.5rem;
            line-height: 1.02;
            color: var(--heading-color);
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        }}
        .workstation-badge {{
            padding: 0.3rem 0.68rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-color);
            color: var(--heading-color);
            font-size: 0.8rem;
            white-space: nowrap;
        }}
        .workstation-subtitle {{
            position: relative;
            z-index: 1;
            margin-top: 0.35rem;
            color: var(--text-color);
            max-width: 72rem;
            line-height: 1.42;
            font-size: 0.92rem;
        }}
        .workstation-rail {{
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.78rem;
        }}
        .workstation-tag {{
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.58rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-color);
            color: var(--heading-color);
            font-size: 0.78rem;
        }}
        .workstation-grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.72rem;
            margin-top: 0.82rem;
        }}
        .workstation-cell {{
            position: relative;
            overflow: hidden;
            padding: 0.82rem 0.9rem 0.8rem 0.9rem;
            border-radius: 0.95rem;
            background: linear-gradient(180deg, var(--surface-soft), var(--surface-alt));
            border: 1px solid var(--border-soft);
            min-height: 7.4rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }}
        .workstation-cell::after {{
            content: "";
            position: absolute;
            top: 0.7rem;
            right: 0.72rem;
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: var(--accent-color);
            opacity: 0.88;
            box-shadow: 0 0 0 4px rgba(18, 102, 79, 0.10);
        }}
        .workstation-kicker {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: var(--muted-color);
            margin-bottom: 0.42rem;
        }}
        .workstation-value {{
            font-size: 1.08rem;
            color: var(--heading-color);
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.28rem;
        }}
        .workstation-copy {{
            color: var(--text-color);
            line-height: 1.36;
            font-size: 0.86rem;
        }}
        .workstation-meta {{
            margin-top: 0.52rem;
        }}
        .terminal-chip {{
            display: inline-block;
            margin-right: 0.34rem;
            margin-bottom: 0.3rem;
            padding: 0.2rem 0.5rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            font-size: 0.76rem;
        }}
        .tape-band {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.42rem;
            padding: 0.56rem 0.72rem;
            margin: 0.06rem 0 0.84rem 0;
            border-radius: 0.95rem;
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
        }}
        .tape-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted-color);
            margin-right: 0.2rem;
        }}
        .tape-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.48rem;
            border-radius: 999px;
            background: var(--chip-bg);
            border: 1px solid var(--border-soft);
        }}
        .tape-pill-label {{
            font-size: 0.69rem;
            color: var(--muted-color);
        }}
        .tape-pill strong {{
            color: var(--heading-color);
            font-size: 0.76rem;
            font-weight: 700;
        }}
        div[data-testid="stHeadingWithActionElements"] h2,
        div[data-testid="stHeadingWithActionElements"] h3 {{
            color: var(--heading-color);
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
            letter-spacing: -0.01em;
        }}
        div[data-testid="stHeadingWithActionElements"] h2 {{
            font-size: 1.22rem;
            line-height: 1.1;
        }}
        div[data-testid="stHeadingWithActionElements"] h3 {{
            font-size: 1.06rem;
            line-height: 1.14;
        }}
        div[data-testid="stPlotlyChart"] {{
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-color);
            border-radius: 1.05rem;
            padding: 0.08rem;
            box-shadow: var(--shadow-lg);
            overflow: visible;
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease;
        }}
        div[data-testid="stPlotlyChart"]:hover {{
            transform: translateY(-2px);
            border-color: var(--border-color);
            box-shadow: 0 18px 36px rgba(8, 22, 32, 0.16);
        }}
        .stAlert {{
            border-radius: 1rem;
            background: var(--surface-bg);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
        }}
        .stAlert p {{
            color: var(--text-color);
        }}
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            background: var(--surface-bg);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 0.35rem;
            box-shadow: var(--shadow-lg);
        }}
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stTable"] table {{
            background: transparent;
            color: var(--text-color);
        }}
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] thead th,
        div[data-testid="stTable"] thead th {{
            background: linear-gradient(180deg, var(--surface-soft), var(--surface-alt)) !important;
            color: var(--heading-color) !important;
            border-bottom: 1px solid var(--border-color) !important;
            font-size: 0.73rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase;
            font-weight: 700 !important;
        }}
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stTable"] tbody td {{
            border-top: 1px solid var(--border-soft) !important;
        }}
        div[data-testid="stExpander"] {{
            background: var(--surface-bg);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            box-shadow: var(--shadow-lg);
        }}
        div[data-testid="stExpander"] details {{
            background: transparent;
        }}
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {{
            color: var(--heading-color) !important;
        }}
        code {{
            color: var(--heading-color);
            background: var(--surface-alt);
            border: 1px solid var(--border-soft);
            border-radius: 0.45rem;
            padding: 0.12rem 0.35rem;
        }}
        pre {{
            background: var(--surface-alt) !important;
            border: 1px solid var(--border-color);
            border-radius: 0.95rem;
            color: var(--text-color) !important;
        }}
        pre code {{
            background: transparent;
            border: none;
            padding: 0;
            color: inherit;
        }}
        .export-shell {{
            position: relative;
            overflow: hidden;
            padding: 0.92rem 1rem;
            margin: 0.15rem 0 0.7rem 0;
            border-radius: 1.02rem;
            background: linear-gradient(180deg, var(--surface-bg), var(--surface-alt));
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-lg);
        }}
        .export-shell::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-color), var(--accent-soft));
            opacity: 0.95;
        }}
        .export-title {{
            font-size: 1rem;
            color: var(--heading-color);
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
            margin-bottom: 0.25rem;
        }}
        .export-copy {{
            color: var(--text-color);
            line-height: 1.45;
            font-size: 0.92rem;
        }}
        .export-item-kicker {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: var(--muted-color);
            margin: 0.1rem 0 0.18rem 0;
        }}
        .export-item-title {{
            font-size: 1rem;
            color: var(--heading-color);
            margin-bottom: 0.18rem;
            font-weight: 700;
        }}
        .export-item-copy {{
            color: var(--text-color);
            line-height: 1.42;
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
        }}
        .export-divider {{
            height: 0.7rem;
        }}
        hr {{
            border-color: var(--border-soft);
        }}
        @media (max-width: 1120px) {{
            .command-grid,
            .alert-grid {{
                grid-template-columns: 1fr 1fr;
            }}
            .workstation-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
        @media (max-width: 780px) {{
            .brand-shell {{
                grid-template-columns: 1fr;
            }}
            .hero-grid {{
                grid-template-columns: 1fr;
            }}
            .workstation-head {{
                flex-direction: column;
            }}
            .command-shell .section-head {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .command-grid,
            .alert-grid,
            .workstation-grid {{
                grid-template-columns: 1fr;
            }}
            .hero-side-panel,
            .hero-stat {{
                backdrop-filter: none;
            }}
        }}
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(12px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="hero-kicker">Collaborative Research Platform</div>
                    <h1>{APP_DISPLAY_NAME}</h1>
                    <p>
                        Developed by the {TEAM_DISPLAY_NAME}, this workbench links interpretable signal generation,
                        portfolio construction, risk guardrails, and explanation-led review in one interface. The
                        default workflow runs on a reproducible reference market dataset, and the same analysis can
                        switch to your own wide price CSV.
                    </p>
                    <div class="chip-row">
                        <span class="caption-chip">Signals</span>
                        <span class="caption-chip">Risk Guardrails</span>
                        <span class="caption-chip">Stress Review</span>
                        <span class="caption-chip">Model Explainability</span>
                    </div>
                </div>
                <div class="hero-side">
                    <div class="hero-side-panel">
                        <div class="hero-side-kicker">Platform Positioning</div>
                        <div class="hero-side-value">Team-built decision-support workbench</div>
                        <div class="hero-side-copy">
                            Designed to support analyst review of model-driven portfolio recommendations rather than
                            position the system as fully autonomous trading software.
                        </div>
                    </div>
                    <div class="hero-stat-grid">
                        <div class="hero-stat">
                            <div class="hero-stat-label">Backbone</div>
                            <div class="hero-stat-value">Weight-centric</div>
                            <div class="hero-stat-copy">Signals flow directly into target weights and execution overlays.</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Risk Layer</div>
                            <div class="hero-stat-value">VaR-aware</div>
                            <div class="hero-stat-copy">Vol targeting, drawdown guard, and stress framing stay visible.</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Explainability</div>
                            <div class="hero-stat-value">Local + Global</div>
                            <div class="hero-stat-copy">Each signal can be explained with drivers, buckets, and what-if views.</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Deployment</div>
                            <div class="hero-stat-value">Offline-first</div>
                            <div class="hero-stat-copy">Runs on the reference dataset and can switch to uploaded prices.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].markdown(
        """
        <div class="mini-card compact intro-card">
            <h3>Signal Research Layer</h3>
            <p>Signals become portfolio weights first, then the same weight ledger drives both risk control and backtesting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        """
        <div class="mini-card compact intro-card">
            <h3>Risk Governance Layer</h3>
            <p>Rolling volatility, VaR, expected shortfall, drawdowns, crisis windows, and scenario shocks remain visible for review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        """
        <div class="mini-card compact intro-card">
            <h3>Explanation Layer</h3>
            <p>Global feature importance, local driver breakdowns, and one-factor what-if views explain why the model chose each position.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_brand_banner() -> None:
    st.markdown(
        f"""
        <div class="brand-shell">
            <div class="brand-mark">RQ</div>
            <div>
                <div class="brand-overline">{TEAM_DISPLAY_NAME}</div>
                <div class="brand-title">{APP_DISPLAY_NAME}</div>
                <div class="brand-subtitle">
                    A team-developed research interface for portfolio recommendation, risk-governed backtesting, and
                    explanation-led signal review.
                </div>
                <div class="brand-pills">
                    <span class="brand-pill">Team Developed</span>
                    <span class="brand-pill">Decision Support</span>
                    <span class="brand-pill">Signals + Risk + Explainability</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _latest_signal_snapshot(result: PrototypeResult) -> dict[str, str | float | int]:
    if result.decision_log.empty:
        return {
            "ticker": DISPLAY_PLACEHOLDER,
            "prediction": float("nan"),
            "final_weight": float("nan"),
            "active_names": 0,
            "signal_date": DISPLAY_PLACEHOLDER,
        }

    latest_signal_date = pd.Timestamp(result.decision_log["signal_date"].max())
    latest = result.decision_log[result.decision_log["signal_date"] == latest_signal_date].copy()
    latest = latest.sort_values("prediction", ascending=False)
    active_names = int((latest["final_weight"].fillna(0.0) > 1e-4).sum())
    if latest.empty:
        return {
            "ticker": DISPLAY_PLACEHOLDER,
            "prediction": float("nan"),
            "final_weight": float("nan"),
            "active_names": active_names,
            "signal_date": _date_label(latest_signal_date),
        }
    row = latest.iloc[0]
    return {
        "ticker": str(row["ticker"]),
        "prediction": float(row["prediction"]),
        "final_weight": float(row["final_weight"]),
        "active_names": active_names,
        "signal_date": _date_label(latest_signal_date),
    }


def _market_tape_html(result: PrototypeResult, benchmark: str) -> str:
    latest_perf = result.performance.iloc[-1]
    latest_signal = _latest_signal_snapshot(result)
    posture = _risk_posture(result)
    context = result.market_context
    tape_items = [
        ("Benchmark", f"{benchmark} {_pct(result.metrics.get('benchmark_total_return', float('nan')))}"),
        ("State", _market_state_label(context.get("market_state"))),
        ("Breadth", _pct(context.get("breadth_20d", float("nan")))),
        ("20D Vol", _pct(context.get("market_vol_20d", float("nan")))),
        ("Risk Score", f"{posture['score']:.0f}/100"),
        ("Exposure", _pct(float(latest_perf["exposure"]))),
        ("Cash", _pct(float(latest_perf["cash_weight"]))),
        (
            "Top Signal",
            f"{latest_signal['ticker']} {_signed_pct(float(latest_signal['prediction']))}",
        ),
        ("Book", f"{int(latest_signal['active_names'])} names"),
        ("Events", str(len(result.risk_events))),
        ("Signal", str(latest_signal["signal_date"])),
    ]
    pills = "".join(
        f'<span class="tape-pill"><span class="tape-pill-label">{label}</span><strong>{value}</strong></span>'
        for label, value in tape_items
    )
    return dedent(
        f"""
        <div class="tape-band">
            <div class="tape-label">Desk Tape</div>
            {pills}
        </div>
        """
    ).strip()


def _build_market_tape(result: PrototypeResult, benchmark: str) -> None:
    st.markdown(_market_tape_html(result, benchmark), unsafe_allow_html=True)


def _workstation_summary_html(result: PrototypeResult, benchmark: str) -> str:
    context = result.market_context
    posture = _risk_posture(result)
    latest_perf = result.performance.iloc[-1]
    latest_signal = _latest_signal_snapshot(result)
    run_span = f"{_date_label(result.performance.index.min())} to {_date_label(result.performance.index.max())}"
    rail_items = [
        f"Run span {run_span}",
        f"{result.prices.shape[1]} assets / {len(result.feature_columns)} features",
        f"{len(result.risk_events)} overlay events",
        f"Benchmark {benchmark}",
    ]
    cells = [
        (
            "Desk Status",
            "Backtest + Risk + Explainability",
            "Walk-forward allocation, overlay control, and diagnostics stay aligned in one surface.",
            [
                "Integrated stack",
                "Operator view",
            ],
        ),
        (
            "Portfolio Pulse",
            f"{_pct(float(latest_perf['exposure']))} net invested",
            f"Cash {_pct(float(latest_perf['cash_weight']))} with turnover {_pct(float(latest_perf['turnover']))}.",
            [
                f"Annual return {_pct(result.metrics.get('annual_return', float('nan')))}",
                f"Sharpe {_num(result.metrics.get('sharpe_ratio', float('nan')))}",
            ],
        ),
        (
            "Benchmark Lens",
            f"{benchmark} {_signed_pct(context.get('market_mom_20d', float('nan')))} 20D",
            f"Breadth {_pct(context.get('breadth_20d', float('nan')))} and vol {_pct(context.get('market_vol_20d', float('nan')))}.",
            [
                f"State {_market_state_label(context.get('market_state'))}",
                f"Dispersion {_num(context.get('dispersion_20d', float('nan')))}",
            ],
        ),
        (
            "Signal Engine",
            f"{latest_signal['ticker']} {_signed_pct(float(latest_signal['prediction']))}",
            f"Final weight {_pct(float(latest_signal['final_weight']))} under a {posture['label'].lower()} posture.",
            [
                f"Signal date {latest_signal['signal_date']}",
                f"Top-{result.config.top_k} selection / {result.config.prediction_horizon}D horizon",
            ],
        ),
    ]

    cell_html = []
    for kicker, value, copy, meta_items in cells:
        meta = "".join(f'<span class="terminal-chip">{item}</span>' for item in meta_items)
        cell_html.append(
            dedent(
                f"""
                <div class="workstation-cell">
                    <div class="workstation-kicker">{kicker}</div>
                    <div class="workstation-value">{value}</div>
                    <div class="workstation-copy">{copy}</div>
                    <div class="workstation-meta">{meta}</div>
                </div>
                """
            ).strip()
        )
    rail_html = "".join(f'<span class="workstation-tag">{item}</span>' for item in rail_items)

    return dedent(
        f"""
        <div class="workstation-shell">
            <div class="workstation-head">
                <div>
                    <div class="workstation-overline">Quant Desk</div>
                    <div class="workstation-title">Institutional Workstation</div>
                </div>
                <div class="workstation-badge">{_market_state_label(context.get("market_state"))} / {posture['label']}</div>
            </div>
            <div class="workstation-subtitle">
                A unified operator view for allocation, risk posture, benchmark context, and signal conviction.
            </div>
            <div class="workstation-rail">{rail_html}</div>
            <div class="workstation-grid">{''.join(cell_html)}</div>
        </div>
        """
    ).strip()


def _build_workstation_header(result: PrototypeResult, benchmark: str) -> None:
    st.markdown(_workstation_summary_html(result, benchmark), unsafe_allow_html=True)


def _build_landing_note() -> None:
    note = (
        "Lead with Executive Overview, then move through Portfolio Lab, Risk Command, and Explainability Lab "
        "to tell a clean strategy story."
    )
    st.markdown(f'<div class="landing-note">{note}</div>', unsafe_allow_html=True)


def _build_quickstart_guide() -> None:
    st.caption("Suggested review path for project meetings, analyst discussion, or stakeholder evaluation.")
    cards = [
        (
            "1. Executive Overview",
            "Start with total return, risk posture, equity curve, and exposure profile to establish what the strategy did.",
        ),
        (
            "2. Portfolio And Risk",
            "Move to Portfolio Lab and Risk Command to show where P&L came from, how the book was positioned, and when overlays triggered.",
        ),
        (
            "3. Explainability And Deliverables",
            "Finish with Explainability Lab for why a position was taken, then export the memo or CSV outputs for handoff.",
        ),
    ]
    cols = st.columns(3)
    for col, (title, text) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="guide-card">
                <h4>{title}</h4>
                <p>{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _build_command_center(result: PrototypeResult, benchmark: str) -> None:
    posture = _risk_posture(result)
    context = result.market_context
    capture = _capture_metrics(result.performance)
    latest_date = result.performance.index.max()
    start_date = result.performance.index.min()
    alerts = _build_alert_feed(result, benchmark)
    rail_items = [
        f"Snapshot {_date_label(latest_date)}",
        f"Benchmark {benchmark}",
        f"Risk score {posture['score']:.0f}/100",
        f"{len(alerts)} active alerts",
    ]
    rail_html = "".join(f'<span class="command-tag">{item}</span>' for item in rail_items)

    cards = [
        (
            "Market Regime",
            _market_state_label(context.get("market_state")),
            f"{context.get('benchmark', benchmark)} breadth {_pct(context.get('breadth_20d', float('nan')))} and 20D vol {_pct(context.get('market_vol_20d', float('nan')))}.",
        ),
        (
            "Risk Posture",
            f"{posture['score']:.0f}/100",
            f"{posture['label']} regime with VaR {_pct(result.metrics.get('var_95', float('nan')))} and drawdown {_pct(result.metrics.get('max_drawdown', float('nan')))}.",
        ),
        (
            "Alpha Engine",
            f"Ridge / {result.config.retrain_every}D",
            f"Horizon {result.config.prediction_horizon}D, top-{result.config.top_k} selection, max weight {_pct(result.config.max_weight)}.",
        ),
        (
            "Run Envelope",
            f"{_date_label(start_date)} to {_date_label(latest_date)}",
            f"{result.prices.shape[1]} assets, {len(result.feature_columns)} features, up capture {_num(capture.get('up_capture', float('nan')))}.",
        ),
    ]

    card_html = []
    for kicker, value, subtext in cards:
        card_html.append(
            dedent(
                f"""
                <div class="status-card compact">
                    <div class="status-kicker">{kicker}</div>
                    <div class="status-value">{value}</div>
                    <div class="status-sub">{subtext}</div>
                </div>
                """
            ).strip()
        )

    alert_html = []
    for alert in alerts:
        alert_html.append(
            dedent(
                f"""
                <div class="mini-card compact">
                    <div class="mini-card-meta"><span class="alert-badge">{alert['level']}</span></div>
                    <h3>{alert['headline']}</h3>
                    <p>{alert['detail']}</p>
                </div>
                """
            ).strip()
        )

    st.markdown(
        dedent(
            f"""
            <div class="command-shell">
                <div class="section-head">
                    <div>
                        <div class="section-kicker">Operator Layer</div>
                        <div class="section-title">Command Center</div>
                        <div class="section-copy">
                            A fast read on regime, posture, model cadence, and live alerts before moving into deeper tabs.
                        </div>
                    </div>
                    <div class="section-badge">{_market_state_label(context.get("market_state"))} / {posture['label']}</div>
                </div>
                <div class="command-rail">{rail_html}</div>
                <div class="command-grid">{''.join(card_html)}</div>
                <div class="alert-grid">{''.join(alert_html)}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def _build_overview_tab(result: PrototypeResult, benchmark: str) -> None:
    palette = _active_palette()
    metrics = result.metrics
    mean_ic = float(result.daily_ic["ic"].mean()) if not result.daily_ic.empty else float("nan")
    latest_exposure = float(result.performance["exposure"].iloc[-1])
    capture = _capture_metrics(result.performance)
    posture = _risk_posture(result)
    window_label = st.select_slider(
        "Display window",
        options=list(WINDOW_LOOKBACKS.keys()),
        value="1Y",
        key="overview_window",
    )
    performance_window = _slice_indexed(result.performance, window_label)
    applied_window = _slice_indexed(result.applied_weights, window_label)
    ic_window = _slice_indexed(result.daily_ic, window_label)

    cols = st.columns(6)
    cols[0].metric("Annual Return", _pct(metrics.get("annual_return", float("nan"))))
    cols[1].metric("Sharpe", _num(metrics.get("sharpe_ratio", float("nan"))))
    cols[2].metric("Max Drawdown", _pct(metrics.get("max_drawdown", float("nan"))))
    cols[3].metric("Total Return", _pct(metrics.get("total_return", float("nan"))))
    cols[4].metric(f"{benchmark} Return", _pct(metrics.get("benchmark_total_return", float("nan"))))
    cols[5].metric("Mean Daily IC", _num(mean_ic))

    gauge_cols = st.columns([0.8, 0.8, 1.4])
    with gauge_cols[0]:
        st.plotly_chart(
            _gauge_figure(
                "Risk Posture",
                posture["score"],
                100,
                [
                    ([0, 35], palette["gauge_safe"]),
                    ([35, 65], palette["gauge_warn"]),
                    ([65, 100], palette["gauge_danger"]),
                ],
            ),
            use_container_width=True,
        )
    with gauge_cols[1]:
        st.plotly_chart(
            _gauge_figure(
                "Average Exposure",
                float(result.performance["exposure"].mean() * 100.0),
                100,
                [
                    ([0, 35], palette["gauge_warn"]),
                    ([35, 75], palette["gauge_safe"]),
                    ([75, 100], palette["gauge_neutral"]),
                ],
                suffix="%",
            ),
            use_container_width=True,
        )
    with gauge_cols[2]:
        comparison = pd.DataFrame(
            {
                "metric": ["Up Capture", "Down Capture", "Information Ratio", "Tracking Error"],
                "value": [
                    capture.get("up_capture", float("nan")),
                    capture.get("down_capture", float("nan")),
                    capture.get("information_ratio", float("nan")),
                    capture.get("tracking_error", float("nan")),
                ],
            }
        )
        st.subheader("Relative Performance Profile")
        st.dataframe(
            _styled_frame(comparison, {"value": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    equity_frame = performance_window.reset_index()[["date", "equity", "benchmark_equity"]].melt(
        id_vars="date",
        var_name="series",
        value_name="value",
    )
    equity_frame["series"] = equity_frame["series"].replace({"equity": "Strategy", "benchmark_equity": benchmark})
    fig_equity = px.line(
        equity_frame,
        x="date",
        y="value",
        color="series",
        color_discrete_map={"Strategy": palette["accent"], benchmark: palette["series_secondary"]},
        title="Equity Curve",
    )
    _apply_plot_style(fig_equity, 420)
    fig_equity.update_layout(legend_title_text="")
    st.plotly_chart(fig_equity, use_container_width=True)

    exposure_frame = performance_window.reset_index()[["date", "exposure", "cash_weight"]].melt(
        id_vars="date",
        var_name="series",
        value_name="weight",
    )
    exposure_frame["series"] = exposure_frame["series"].replace(
        {"exposure": "Exposure", "cash_weight": "Cash"}
    )
    fig_exposure = px.area(
        exposure_frame,
        x="date",
        y="weight",
        color="series",
        title="Exposure Profile",
        color_discrete_map={"Exposure": palette["accent"], "Cash": palette["accent_soft"]},
    )
    _apply_plot_style(fig_exposure, 280)
    fig_exposure.update_layout(legend_title_text="")
    st.plotly_chart(fig_exposure, use_container_width=True)

    active_frame = performance_window.reset_index().copy()
    active_frame["active_return"] = active_frame["strategy_return"] - active_frame["benchmark_return"]
    active_frame["rolling_active_return_21d"] = (1.0 + active_frame["active_return"]).rolling(21).apply(np.prod, raw=True) - 1.0
    active_frame["rolling_sharpe_63d"] = (
        active_frame["strategy_return"].rolling(63).mean() / active_frame["strategy_return"].rolling(63).std()
    ) * np.sqrt(252)
    rolling_panel = "dense"
    summary_panel = "medium"
    left, right = _panel_columns(2)
    with left:
        with _panel_box(size=rolling_panel):
            _panel_header("21D Rolling Active Return", "Short-horizon active edge versus the benchmark.")
            fig_active = px.line(
                active_frame,
                x="date",
                y="rolling_active_return_21d",
                title=None,
            )
            fig_active.update_traces(line_color=palette["series_secondary"])
            _apply_plot_style(fig_active, _panel_chart_height(rolling_panel), external_title=True)
            st.plotly_chart(fig_active, use_container_width=True)
    with right:
        with _panel_box(size=rolling_panel):
            _panel_header("63D Rolling Sharpe", "Medium-horizon risk-adjusted performance trend.")
            fig_sharpe = px.line(
                active_frame,
                x="date",
                y="rolling_sharpe_63d",
                title=None,
            )
            fig_sharpe.update_traces(line_color=palette["heading"])
            _apply_plot_style(fig_sharpe, _panel_chart_height(rolling_panel), external_title=True)
            st.plotly_chart(fig_sharpe, use_container_width=True)

    left, right = _panel_columns([1.45, 1.0])
    with left:
        with _panel_box(size=summary_panel):
            _panel_header("Applied Portfolio Weights", "Realized weights after overlays and cash management.")
            weights_long = applied_window.reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
            fig_weights = px.area(
                weights_long,
                x="date",
                y="weight",
                color="ticker",
                title=None,
            )
            _apply_plot_style(fig_weights, _panel_chart_height(summary_panel), external_title=True)
            fig_weights.update_layout(legend_title_text="")
            st.plotly_chart(fig_weights, use_container_width=True)

    with right:
        with _panel_box(size=summary_panel):
            latest_date = result.applied_weights.index.max()
            latest_alloc = result.applied_weights.loc[latest_date]
            latest_alloc = latest_alloc[latest_alloc > 0.0001].sort_values(ascending=False)
            _panel_header(
                "Latest Allocation",
                f"Latest realized date: {_date_label(latest_date)}. Net invested exposure: {_pct(latest_exposure)}.",
            )
            latest_day = result.decision_log[result.decision_log["realized_date"] == latest_date].copy()
            latest_day = latest_day.sort_values("prediction", ascending=False) if not latest_day.empty else latest_day
            alloc_tab, decision_tab = st.tabs(["Allocation", "Decision Snapshot"])
            with alloc_tab:
                if latest_alloc.empty:
                    st.info("The strategy is fully in cash on the latest rebalance date.")
                else:
                    st.dataframe(
                        _styled_frame(latest_alloc.rename("weight").to_frame(), {"weight": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=False,
                        height=_panel_table_height(len(latest_alloc), summary_panel, min_height=240, max_height=340),
                    )
            with decision_tab:
                if latest_day.empty:
                    st.info("No decision snapshot is available for the latest realized date.")
                else:
                    st.dataframe(
                        _styled_frame(
                            latest_day[["ticker", "prediction", "raw_weight", "pre_drawdown_weight", "final_weight"]].rename(
                                columns={
                                    "prediction": "predicted_next_day_return",
                                    "pre_drawdown_weight": "risk_scaled_weight",
                                }
                            ),
                            {
                                "predicted_next_day_return": "{:.3%}",
                                "raw_weight": "{:.1%}",
                                "risk_scaled_weight": "{:.1%}",
                                "final_weight": "{:.1%}",
                            },
                        ),
                        use_container_width=True,
                        hide_index=True,
                        height=_panel_table_height(len(latest_day), summary_panel, min_height=260, max_height=360),
                    )

    ic_frame = ic_window.reset_index()
    fig_ic = px.line(ic_frame, x="date", y="ic", title="Daily Cross-Sectional Information Coefficient")
    fig_ic.update_traces(line_color=palette["heading"])
    _apply_plot_style(fig_ic, 320)
    st.plotly_chart(fig_ic, use_container_width=True)


def _build_portfolio_lab_tab(result: PrototypeResult, benchmark: str) -> None:
    palette = _active_palette()
    window_label = st.select_slider(
        "Contribution window",
        options=list(WINDOW_LOOKBACKS.keys()),
        value="1Y",
        key="portfolio_window",
    )
    contribution_summary, daily_contrib = _asset_contribution_summary(result)
    daily_contrib_window = _slice_indexed(
        daily_contrib.sort_values("date").set_index("date"),
        window_label,
    ).reset_index()
    monthly_pivot = _monthly_return_pivot(result.performance)
    yearly_summary = _yearly_summary(result.performance)
    factor_exposure = _latest_feature_exposure(result)
    lab_panel = "small"

    left, right = _panel_columns([1.2, 1.0])
    with left:
        with _panel_box(size=lab_panel):
            _panel_header("Daily P&L Contribution By Asset", "Attribution over the selected contribution window.")
            fig_contrib = px.area(
                daily_contrib_window,
                x="date",
                y="daily_pnl_contribution",
                color="ticker",
                title=None,
            )
            _apply_plot_style(fig_contrib, _panel_chart_height(lab_panel), external_title=True)
            fig_contrib.update_layout(legend_title_text="")
            st.plotly_chart(fig_contrib, use_container_width=True)

    with right:
        with _panel_box(size=lab_panel):
            _panel_header("Latest Allocation Treemap", "Current active book by realized weight.")
            latest_weights = result.applied_weights.iloc[-1]
            latest_weights = latest_weights[latest_weights > 0.0001].sort_values(ascending=False)
            if latest_weights.empty:
                st.info("No active positions are held on the latest rebalance date.")
            else:
                treemap_frame = latest_weights.rename("weight").reset_index()
                treemap_frame.columns = ["ticker", "weight"]
                fig_tree = px.treemap(
                    treemap_frame,
                    path=["ticker"],
                    values="weight",
                    color="weight",
                    color_continuous_scale=[palette["sequential_low"], palette["sequential_high"]],
                    title=None,
                )
                fig_tree.update_traces(root_color="rgba(0,0,0,0)")
                _apply_plot_style(fig_tree, _panel_chart_height(lab_panel), external_title=True)
                fig_tree.update_layout(
                    margin=dict(l=10, r=10, t=24, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family='Aptos, "Segoe UI", "Helvetica Neue", sans-serif', color=palette["text"]),
                    coloraxis_showscale=False,
                )
                _clear_internal_plot_title(fig_tree)
                st.plotly_chart(fig_tree, use_container_width=True)

    left, right = _panel_columns([1.0, 1.1])
    with left:
        with _panel_box(size=lab_panel):
            _panel_header("Asset Contribution Scorecard", "Average signal quality, exposure, and hit rate by asset.")
            st.dataframe(
                _styled_frame(
                    contribution_summary,
                    {
                        "total_pnl_contribution": "{:.2%}",
                        "avg_weight": "{:.1%}",
                        "avg_prediction": "{:.3%}",
                        "avg_realized": "{:.3%}",
                        "hit_rate": "{:.0%}",
                        "selection_rate": "{:.0%}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(contribution_summary), lab_panel, min_height=340, max_height=420),
            )

    with right:
        with _panel_box(size=lab_panel):
            _panel_header("Latest Portfolio Feature Radar", "Current style exposure relative to recent feature history.")
            if not factor_exposure.empty:
                polar = go.Figure()
                polar.add_trace(
                    go.Scatterpolar(
                        r=factor_exposure["exposure_zscore"],
                        theta=factor_exposure["feature"],
                        fill="toself",
                        line=dict(color=palette["heading"], width=2),
                        fillcolor=palette["gauge_safe"],
                        name="Current portfolio",
                    )
                )
                _apply_plot_style(polar, _panel_chart_height(lab_panel), external_title=True)
                polar.update_layout(
                    margin=dict(l=26, r=26, t=26, b=26),
                    polar=dict(
                        domain=dict(x=[0.10, 0.90], y=[0.10, 0.90]),
                        bgcolor=palette["plot_bg"],
                        radialaxis=dict(
                            showline=False,
                            gridcolor=palette["grid"],
                            tickfont=dict(size=10),
                        ),
                        angularaxis=dict(
                            gridcolor=palette["grid"],
                            tickfont=dict(size=9),
                        ),
                    ),
                    showlegend=False,
                )
                _clear_internal_plot_title(polar)
                st.plotly_chart(polar, use_container_width=True)
            else:
                st.info("Feature radar becomes available once the portfolio holds active positions.")

    left, right = _panel_columns([1.15, 0.95])
    with left:
        with _panel_box(size=lab_panel):
            _panel_header("Monthly Return Heatmap", "Calendar view of compounded monthly strategy returns.")
            if not monthly_pivot.empty:
                fig_monthly = px.imshow(
                    monthly_pivot,
                    text_auto=".1%",
                    aspect="auto",
                    color_continuous_scale=_theme_scale(),
                    title=None,
                )
                _apply_plot_style(fig_monthly, _panel_chart_height(lab_panel), external_title=True)
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.info("Monthly return heatmap is unavailable for the current sample.")

    with right:
        with _panel_box(size=lab_panel):
            _panel_header("Yearly Summary", "Annual return, volatility, drawdown, turnover, and exposure.")
            st.dataframe(
                _styled_frame(
                    yearly_summary,
                    {
                        "strategy_return": "{:.2%}",
                        "benchmark_return": "{:.2%}",
                        "annual_vol": "{:.2%}",
                        "max_drawdown": "{:.2%}",
                        "avg_turnover": "{:.2%}",
                        "avg_exposure": "{:.1%}",
                        "active_return": "{:.2%}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(yearly_summary), lab_panel, min_height=280, max_height=380),
            )


def _build_strategy_compare_tab(result: PrototypeResult, benchmark: str) -> None:
    comparison = _strategy_comparison_frame(result, benchmark)
    returns_only = comparison[[benchmark, STRATEGY_DISPLAY_NAME, "Equal Weight Universe", "Top-K Equal Weight"]]
    metric_table = _series_metric_table(returns_only)
    compare_panel = "small"

    left, right = _panel_columns([1.25, 0.95])
    with left:
        with _panel_box(size=compare_panel):
            _panel_header("Strategy Comparison Equity Curves", "Workbench performance against the benchmark and simple baselines.")
            equity_cols = [col for col in comparison.columns if col.endswith("Equity")]
            equity_frame = comparison[equity_cols].reset_index().melt(id_vars="date", var_name="series", value_name="value")
            equity_frame["series"] = equity_frame["series"].str.replace(" Equity", "", regex=False)
            fig = px.line(
                equity_frame,
                x="date",
                y="value",
                color="series",
                title=None,
            )
            _apply_plot_style(fig, _panel_chart_height(compare_panel), external_title=True)
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with _panel_box(size=compare_panel):
            _panel_header("Comparison Scorecard", "Return, volatility, Sharpe, drawdown, and win rate in one view.")
            st.dataframe(
                _styled_frame(
                    metric_table,
                    {
                        "total_return": "{:.2%}",
                        "annual_vol": "{:.2%}",
                        "sharpe": "{:.3f}",
                        "max_drawdown": "{:.2%}",
                        "win_rate": "{:.0%}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(metric_table), compare_panel, min_height=300, max_height=420),
            )

    rolling = returns_only.copy()
    rolling["Strategy vs Equal Weight"] = (1.0 + rolling[STRATEGY_DISPLAY_NAME] - rolling["Equal Weight Universe"]).rolling(21).apply(np.prod, raw=True) - 1.0
    rolling["Strategy vs Benchmark"] = (1.0 + rolling[STRATEGY_DISPLAY_NAME] - rolling[benchmark]).rolling(21).apply(np.prod, raw=True) - 1.0
    compare_window = rolling.reset_index().melt(
        id_vars="date",
        value_vars=["Strategy vs Equal Weight", "Strategy vs Benchmark"],
        var_name="series",
        value_name="value",
    )
    fig_compare = px.line(compare_window, x="date", y="value", color="series", title="21D Relative Edge")
    _apply_plot_style(fig_compare, 320)
    fig_compare.update_layout(legend_title_text="")
    st.plotly_chart(fig_compare, use_container_width=True)


def _build_signal_monitor_tab(result: PrototypeResult, benchmark: str) -> None:
    palette = _active_palette()
    latest_signal_date = pd.Timestamp(result.decision_log["signal_date"].max())
    latest = (
        result.decision_log[result.decision_log["signal_date"] == latest_signal_date]
        .copy()
        .sort_values("prediction", ascending=False)
    )
    alerts = _build_alert_feed(result, benchmark)

    st.subheader(f"Signal Monitor | {latest_signal_date.strftime('%Y-%m-%d')}")
    alert_frame = pd.DataFrame(alerts)
    st.dataframe(
        _prepare_display_frame(alert_frame, max_text=96),
        use_container_width=True,
        hide_index=True,
        height=max(180, _table_height(len(alert_frame), min_height=160, max_height=220)),
    )
    monitor_panel = "small"

    left, right = _panel_columns([1.15, 1.0])
    with left:
        with _panel_box(size=monitor_panel):
            _panel_header("Signal Opportunity Map", "Cross-section of momentum, volatility, prediction, and final weight.")
            monitor_fig = px.scatter(
                latest,
                x="mom_20d",
                y="vol_20d",
                size="final_weight",
                color="prediction",
                hover_name="ticker",
                color_continuous_scale=_theme_scale(),
                title=None,
            )
            _apply_plot_style(monitor_fig, _panel_chart_height(monitor_panel), external_title=True)
            st.plotly_chart(monitor_fig, use_container_width=True)

    with right:
        with _panel_box(size=monitor_panel):
            signal_mix = latest[["ticker", "prediction", "target", "final_weight", "estimated_vol", "estimated_var"]].copy()
            _panel_header("Latest Ranked Book", "Latest cross-sectional ranking with realized portfolio sizing inputs.")
            st.dataframe(
                _styled_frame(
                    signal_mix,
                    {
                        "prediction": "{:.3%}",
                        "target": "{:.3%}",
                        "final_weight": "{:.1%}",
                        "estimated_vol": "{:.2%}",
                        "estimated_var": "{:.2%}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(signal_mix), monitor_panel, min_height=280, max_height=380),
            )

    watch_cols = st.columns(3)
    top_pred = latest.iloc[0] if not latest.empty else None
    watch_cols[0].metric("Top Signal", top_pred["ticker"] if top_pred is not None else DISPLAY_PLACEHOLDER)
    watch_cols[1].metric("Top Prediction", _pct(float(top_pred["prediction"])) if top_pred is not None else DISPLAY_PLACEHOLDER)
    watch_cols[2].metric("Latest Overlay Count", str(len(result.risk_events.tail(10))))
def _build_risk_tab(result: PrototypeResult) -> None:
    palette = _active_palette()
    risk_window = st.select_slider(
        "Risk window",
        options=list(WINDOW_LOOKBACKS.keys()),
        value="1Y",
        key="risk_window",
    )
    risk_frame = _slice_indexed(result.risk_series, risk_window).reset_index()
    latest_weights = result.applied_weights.iloc[-1] if not result.applied_weights.empty else pd.Series(dtype=float)
    shock_table = _shock_table(latest_weights)
    posture = _risk_posture(result)
    correlation_matrix = _correlation_matrix(result, risk_window)
    trend_panel = "dense"
    context_panel = "dense"
    compact_panel = "compact"
    detail_panel = "small"

    cols = st.columns(6)
    cols[0].metric("1D VaR 95", _pct(result.metrics.get("var_95", float("nan"))))
    cols[1].metric("1D ES 95", _pct(result.metrics.get("es_95", float("nan"))))
    cols[2].metric("Annual Vol", _pct(result.metrics.get("annual_volatility", float("nan"))))
    cols[3].metric("Beta To Benchmark", _num(result.metrics.get("beta_to_benchmark", float("nan"))))
    cols[4].metric("Avg Turnover", _pct(result.metrics.get("avg_turnover", float("nan"))))
    cols[5].metric("Latest Cash Buffer", _pct(result.performance["cash_weight"].iloc[-1]))

    left, right = _panel_columns(2)
    with left:
        with _panel_box(size=trend_panel):
            _panel_header("Rolling Risk Metrics", "Volatility, VaR, and expected shortfall over time.")
            fig_risk = px.line(
                risk_frame,
                x="date",
                y=["rolling_vol_20d", "rolling_var_60d", "rolling_es_60d"],
                title=None,
            )
            _apply_plot_style(fig_risk, _panel_chart_height(trend_panel), external_title=True)
            fig_risk.update_layout(legend_title_text="")
            st.plotly_chart(fig_risk, use_container_width=True)

    with right:
        with _panel_box(size=trend_panel):
            _panel_header("Drawdown And Benchmark Correlation", "Stress depth and rolling co-movement with the benchmark.")
            fig_dd = px.line(
                risk_frame,
                x="date",
                y=["drawdown", "rolling_corr_60d"],
                title=None,
            )
            _apply_plot_style(fig_dd, _panel_chart_height(trend_panel), external_title=True)
            fig_dd.update_layout(legend_title_text="")
            st.plotly_chart(fig_dd, use_container_width=True)

    left, right = _panel_columns([0.8, 1.2])
    with left:
        with _panel_box(size=context_panel):
            _panel_header("Risk Control Score", "Composite view of volatility, drawdown, and overlay activity.")
            st.plotly_chart(
                _gauge_figure(
                    "Risk Control Score",
                    posture["score"],
                    100,
                    [
                        ([0, 35], palette["gauge_safe"]),
                        ([35, 65], palette["gauge_warn"]),
                        ([65, 100], palette["gauge_danger"]),
                    ],
                    external_title=True,
                ),
                use_container_width=True,
            )
            st.caption(
                f"Current posture is **{posture['label']}** based on realized volatility, drawdown, VaR usage, and overlay activity."
            )

    with right:
        with _panel_box(size=context_panel):
            _panel_header(
                f"Trailing {risk_window} Asset Correlation Map",
                "Cross-asset co-movement over the selected risk window.",
            )
            if not correlation_matrix.empty:
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=".2f",
                    color_continuous_scale=_theme_scale(),
                    zmin=-1.0,
                    zmax=1.0,
                    title=None,
                )
                _apply_plot_style(fig_corr, _panel_chart_height(context_panel), external_title=True)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Correlation heatmap is unavailable for the selected window.")

    left, right = _panel_columns([1.0, 1.1])
    with left:
        with _panel_box(size=compact_panel):
            _panel_header("Scenario Shock Matrix", "Approximate portfolio impact under uniform market stress.")
            st.dataframe(
                _styled_frame(
                    shock_table,
                    {
                        "approx_portfolio_pnl": "{:.2%}",
                        "net_exposure": "{:.1%}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(shock_table), compact_panel, min_height=240, max_height=300),
            )

    with right:
        with _panel_box(size=compact_panel):
            crisis = result.crisis_summary
            _panel_header("Worst Benchmark Window", "Most adverse benchmark stress interval in the current sample.")
            if crisis:
                crisis_cols = st.columns(4)
                crisis_cols[0].metric("Window Start", _date_label(crisis["start"]))
                crisis_cols[1].metric("Window End", _date_label(crisis["end"]))
                crisis_cols[2].metric("Benchmark Return", _pct(crisis["benchmark_window_return"]))
                crisis_cols[3].metric("Strategy Return", _pct(crisis["strategy_window_return"]))
                st.caption(
                    f"During this {crisis['window_days']}-day stress window the strategy reached "
                    f"{_pct(crisis['strategy_window_drawdown'])} drawdown."
                )
            else:
                st.info("The sample is too short to compute a crisis window summary.")

    left, right = _panel_columns([1.0, 1.1])
    with left:
        with _panel_box(size=detail_panel):
            worst_days = result.performance.reset_index().nsmallest(10, "strategy_return")[
                ["date", "strategy_return", "benchmark_return", "drawdown", "turnover"]
            ]
            _panel_header("Worst Strategy Days", "Largest single-day losses, benchmark context, and turnover.")
            st.dataframe(
                _styled_frame(
                    worst_days,
                    {
                        "strategy_return": "{:.2%}",
                        "benchmark_return": "{:.2%}",
                        "drawdown": "{:.2%}",
                        "turnover": "{:.2%}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(worst_days), detail_panel, min_height=320, max_height=420),
            )

    with right:
        with _panel_box(size=detail_panel):
            _panel_header("Risk Overlay Events", "Trigger mix and recent event log from the active overlays.")
            if result.risk_events.empty:
                st.success("No risk overlays were triggered in this run.")
            else:
                counts = result.risk_events["event"].value_counts().rename_axis("event").reset_index(name="count")
                overlay_chart_tab, overlay_table_tab = st.tabs(["Trigger Count", "Event Log"])
                with overlay_chart_tab:
                    fig_events = px.bar(counts, x="event", y="count", color="event", title=None)
                    _apply_plot_style(fig_events, _panel_chart_height(detail_panel), external_title=True)
                    fig_events.update_layout(
                        showlegend=False,
                        margin=dict(l=10, r=10, t=18, b=10),
                    )
                    _clear_internal_plot_title(fig_events)
                    st.plotly_chart(fig_events, use_container_width=True)
                with overlay_table_tab:
                    st.dataframe(
                        _prepare_display_frame(result.risk_events.tail(40), max_text=84),
                        use_container_width=True,
                        hide_index=True,
                        height=_panel_table_height(len(result.risk_events.tail(40)), detail_panel, min_height=320, max_height=420),
                    )


def _build_explainability_tab(result: PrototypeResult) -> None:
    palette = _active_palette()
    predictions = result.predictions.copy()
    importance = _importance_frame(result)
    quality_metrics, bucket_table = _prediction_quality_metrics(predictions)
    predictions["residual"] = predictions["target"] - predictions["prediction"]
    insight_panel = "small"
    compact_panel = "dense"

    cols = st.columns(5)
    cols[0].metric("Rank IC", _num(quality_metrics.get("rank_ic", float("nan"))))
    cols[1].metric("Sign Hit Rate", _pct(quality_metrics.get("sign_hit_rate", float("nan"))))
    cols[2].metric("MAE", _pct(quality_metrics.get("mae", float("nan"))))
    cols[3].metric("RMSE", _pct(quality_metrics.get("rmse", float("nan"))))
    cols[4].metric("Top-Bottom Spread", _pct(quality_metrics.get("top_minus_bottom", float("nan"))))

    left, right = _panel_columns(2)
    with left:
        with _panel_box(size=insight_panel):
            _panel_header("Predicted Vs Realized Next-Day Return", "Model fit by ticker across the full prediction panel.")
            scatter = px.scatter(
                predictions,
                x="prediction",
                y="target",
                color="ticker",
                title=None,
            )
            _apply_plot_style(scatter, _panel_chart_height(insight_panel), external_title=True)
            st.plotly_chart(scatter, use_container_width=True)

    with right:
        with _panel_box(size=insight_panel):
            _panel_header("Prediction Bucket Calibration", "Average prediction versus realized return by model bucket.")
            bucket_long = bucket_table.melt(
                id_vars="bucket",
                value_vars=["avg_prediction", "avg_realized_return"],
                var_name="series",
                value_name="value",
            )
            bucket_long["series"] = bucket_long["series"].replace(
                {
                    "avg_prediction": "Average prediction",
                    "avg_realized_return": "Average realized return",
                }
            )
            fig_bucket = px.line(
                bucket_long,
                x="bucket",
                y="value",
                color="series",
                markers=True,
                title=None,
                color_discrete_map={
                    "Average prediction": palette["heading"],
                    "Average realized return": palette["series_secondary"],
                },
            )
            _apply_plot_style(fig_bucket, _panel_chart_height(insight_panel), external_title=True)
            fig_bucket.update_layout(legend_title_text="")
            st.plotly_chart(fig_bucket, use_container_width=True)

    top_perm = importance.head(10).sort_values("permutation_importance")
    fig_importance = px.bar(
        top_perm,
        x="permutation_importance",
        y="feature",
        orientation="h",
        color="abs_coefficient",
        color_continuous_scale=_theme_scale(),
        title="Global Feature Importance",
    )
    _apply_plot_style(fig_importance, 360)
    st.plotly_chart(fig_importance, use_container_width=True)

    feature_focus = st.selectbox(
        "Global dependence feature",
        options=result.feature_columns,
        index=result.feature_columns.index("mom_20d"),
        key="global_dependence_feature",
    )
    left, right = _panel_columns(2)
    with left:
        with _panel_box(size=compact_panel):
            _panel_header("Prediction Dependence", f"How the selected feature loads into predicted return for {feature_focus}.")
            dependence = px.scatter(
                predictions.sample(min(1800, len(predictions)), random_state=42),
                x=feature_focus,
                y="prediction",
                color="target",
                color_continuous_scale=_theme_scale(),
                title=None,
            )
            _apply_plot_style(dependence, _panel_chart_height(compact_panel), external_title=True)
            st.plotly_chart(dependence, use_container_width=True)

    with right:
        with _panel_box(size=compact_panel):
            _panel_header("Residual Distribution", "Histogram of realized minus predicted return.")
            residual_hist = px.histogram(
                predictions,
                x="residual",
                nbins=45,
                title=None,
                color_discrete_sequence=[palette["heading"]],
            )
            _apply_plot_style(residual_hist, _panel_chart_height(compact_panel), external_title=True)
            st.plotly_chart(residual_hist, use_container_width=True)

    signal_dates = sorted(predictions["date"].unique())
    selected_signal_date = st.select_slider(
        "Signal date",
        options=signal_dates,
        value=signal_dates[-1],
        format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"),
    )
    day_snapshot = result.decision_log[result.decision_log["signal_date"] == pd.Timestamp(selected_signal_date)].copy()
    day_snapshot = day_snapshot.sort_values("prediction", ascending=False)
    if day_snapshot.empty:
        st.warning("No signal snapshot is available for the selected date.")
        return

    selected_ticker = st.selectbox("Ticker to explain", options=day_snapshot["ticker"].tolist(), index=0)
    row = predictions[
        (predictions["date"] == pd.Timestamp(selected_signal_date)) & (predictions["ticker"] == selected_ticker)
    ].iloc[0]
    model = _model_for_signal_date(result, pd.Timestamp(selected_signal_date))
    contributions = local_contributions(model, row, result.feature_columns)
    feature_choice = st.selectbox(
        "What-if feature",
        options=result.feature_columns,
        index=result.feature_columns.index("mom_20d"),
    )
    feature_history = result.panel[result.panel["date"] <= pd.Timestamp(selected_signal_date)]
    what_if = what_if_curve(model, row, feature_choice, feature_history, result.feature_columns)

    selected_weight = float(day_snapshot.loc[day_snapshot["ticker"] == selected_ticker, "final_weight"].iloc[0])
    selected_cash = float(day_snapshot["cash_weight"].iloc[0])
    narrative = build_signal_narrative(
        contributions,
        predicted_return=float(row["prediction"]),
        realized_return=float(row["target"]),
        final_weight=selected_weight,
    )
    st.info(narrative)

    cols = st.columns(4)
    cols[0].metric("Predicted Return", _pct(float(row["prediction"])))
    cols[1].metric("Realized Return", _pct(float(row["target"])))
    cols[2].metric("Final Weight", _pct(selected_weight))
    cols[3].metric("Cash After Overlay", _pct(selected_cash))

    left, right = _panel_columns(2)
    with left:
        contrib_chart = px.bar(
            contributions.head(10).sort_values("contribution"),
            x="contribution",
            y="feature",
            orientation="h",
            color="contribution",
            color_continuous_scale=_theme_scale(),
            title="Local Contribution Breakdown",
        )
        _apply_plot_style(contrib_chart, 380)
        st.plotly_chart(contrib_chart, use_container_width=True)

    with right:
        what_if_chart = px.line(
            what_if,
            x="feature_value",
            y="predicted_return",
            title=f"What If {feature_choice} Changes?",
        )
        what_if_chart.update_traces(line_color=palette["heading"])
        _apply_plot_style(what_if_chart, 380)
        st.plotly_chart(what_if_chart, use_container_width=True)

    local_driver_features = contributions.head(8)["feature"].tolist()
    feature_context = feature_percentile_table(
        row,
        feature_history,
        result.feature_columns,
        focus_features=local_driver_features,
    )
    local_table = contributions.head(8).merge(
        feature_context.drop(columns=["feature_value"], errors="ignore"),
        on="feature",
        how="left",
    )

    left, right = _panel_columns([1.0, 1.15])
    with left:
        with _panel_box(size=insight_panel):
            _panel_header("Top Local Drivers", "Feature values, percentiles, and local contribution strength.")
            st.dataframe(
                _styled_frame(
                    local_table[
                        ["feature", "feature_value", "history_percentile", "contribution", "z_score_vs_median"]
                    ],
                    {
                        "feature_value": "{:.3f}",
                        "history_percentile": "{:.0%}",
                        "contribution": "{:.4f}",
                        "z_score_vs_median": "{:.2f}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(local_table), insight_panel, min_height=280, max_height=390),
            )

    with right:
        with _panel_box(size=insight_panel):
            _panel_header("Signal-Day Allocation Table", "Prediction, realized return, sizing, and overlay scalars for the selected day.")
            st.dataframe(
                _styled_frame(
                    day_snapshot[
                        [
                            "ticker",
                            "prediction",
                            "target",
                            "raw_weight",
                            "pre_drawdown_weight",
                            "final_weight",
                            "vol_scalar",
                            "var_scalar",
                        ]
                    ],
                    {
                        "prediction": "{:.3%}",
                        "target": "{:.3%}",
                        "raw_weight": "{:.1%}",
                        "pre_drawdown_weight": "{:.1%}",
                        "final_weight": "{:.1%}",
                        "vol_scalar": "{:.2f}",
                        "var_scalar": "{:.2f}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(day_snapshot), insight_panel, min_height=280, max_height=390),
            )


def _build_architecture_and_data_tab(result: PrototypeResult) -> None:
    palette = _active_palette()
    context = result.market_context
    benchmark = str(context.get("benchmark", result.config.benchmark))
    asset_profile = pd.DataFrame(
        {
            "ticker": result.prices.columns,
            "latest_price": result.prices.iloc[-1].values,
            "20d_vol": result.returns.rolling(20).std().iloc[-1].values * np.sqrt(252),
            "missing_ratio": result.prices.isna().mean().values,
        }
    )
    overview_panel = "dense"
    compact_panel = "dense"

    cols = st.columns(6)
    cols[0].metric("Universe Size", str(result.prices.shape[1]))
    cols[1].metric("Feature Count", str(len(result.feature_columns)))
    cols[2].metric("Signal Days", str(result.predictions["date"].nunique()))
    cols[3].metric("Overlay Events", str(len(result.risk_events)))
    cols[4].metric("Market State", _market_state_label(context.get("market_state")))
    cols[5].metric("Context Date", _date_label(context.get("date")))

    left, right = _panel_columns([1.0, 1.1])
    with left:
        with _panel_box(size=overview_panel):
            st.subheader("Latest Market Context")
            context_frame = pd.DataFrame(
                [
                    {"metric": "Benchmark", "value": context.get("benchmark", DISPLAY_PLACEHOLDER)},
                    {"metric": "20D market momentum", "value": _pct(context.get("market_mom_20d", float("nan")))},
                    {"metric": "20D market volatility", "value": _pct(context.get("market_vol_20d", float("nan")))},
                    {"metric": "60D market drawdown", "value": _pct(context.get("market_drawdown_60d", float("nan")))},
                    {"metric": "Breadth above 20D MA", "value": _pct(context.get("breadth_20d", float("nan")))},
                    {"metric": "Cross-asset dispersion", "value": _num(context.get("dispersion_20d", float("nan")))},
                ]
            )
            st.dataframe(_prepare_display_frame(context_frame), use_container_width=True, hide_index=True, height=270)

    with right:
        with _panel_box(size=overview_panel):
            st.subheader("Current Configuration")
            config_frame = pd.DataFrame(
                [{"parameter": key, "value": value} for key, value in vars(result.config).items()]
            )
            st.dataframe(
                _prepare_display_frame(config_frame),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(config_frame), overview_panel, min_height=320, max_height=420),
            )

    st.markdown(
        """
        <div class="export-shell">
            <div class="export-title">Export Deliverables</div>
            <div class="export-copy">
                Use these outputs for handoff, memo writing, offline review, or follow-up analysis outside the dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    perf_csv = result.performance.reset_index().to_csv(index=False)
    decisions_csv = result.decision_log.to_csv(index=False)
    predictions_csv = result.predictions.to_csv(index=False)
    report_md = _build_report_markdown(result, benchmark)
    with st.expander("Open Export Center", expanded=False):
        export_items = [
            (
                "Performance Ledger",
                "Performance CSV",
                "Daily strategy and benchmark returns, equity, turnover, cash, and drawdown.",
                perf_csv,
                f"{EXPORT_FILE_PREFIX}_performance.csv",
            ),
            (
                "Execution Trace",
                "Decision Log CSV",
                "Per-asset predictions, selected book, overlay scalars, and realized final weights.",
                decisions_csv,
                f"{EXPORT_FILE_PREFIX}_decision_log.csv",
            ),
            (
                "Model Panel",
                "Predictions CSV",
                "Model-ready prediction table with features and realized targets.",
                predictions_csv,
                f"{EXPORT_FILE_PREFIX}_predictions.csv",
            ),
            (
                "Executive Note",
                "Investment Memo",
                "One-page markdown summary suitable for a presentation appendix or internal note.",
                report_md,
                f"{EXPORT_FILE_PREFIX}_investment_memo.md",
            ),
        ]
        export_cols = _panel_columns(2, gap="medium")
        for idx, (kicker, title, copy, payload, file_name) in enumerate(export_items):
            with export_cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="export-item-kicker">{kicker}</div>
                    <div class="export-item-title">{title}</div>
                    <div class="export-item-copy">{copy}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.download_button(
                    f"Download {title}",
                    payload,
                    file_name=file_name,
                    use_container_width=True,
                )
                if idx < len(export_items) - 2:
                    st.markdown('<div class="export-divider"></div>', unsafe_allow_html=True)

    left, right = _panel_columns([1.0, 1.0])
    with left:
        with _panel_box(size=compact_panel):
            _panel_header("1Y Asset Correlation Matrix", "Cross-asset dependence over the trailing one-year window.")
            corr = _correlation_matrix(result, "1Y")
            if not corr.empty:
                corr_fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale=_theme_scale(),
                    zmin=-1.0,
                    zmax=1.0,
                    title=None,
                )
                _apply_plot_style(corr_fig, _panel_chart_height(compact_panel), external_title=True)
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("Not enough data to display the 1Y correlation matrix.")

    with right:
        with _panel_box(size=compact_panel):
            _panel_header("Asset Universe Profile", "Latest price snapshot, trailing volatility, and data completeness.")
            st.dataframe(
                _styled_frame(
                    asset_profile,
                    {
                        "latest_price": "{:.2f}",
                        "20d_vol": "{:.2%}",
                        "missing_ratio": "{:.1%}",
                    },
                ),
                use_container_width=True,
                hide_index=True,
                height=_panel_table_height(len(asset_profile), compact_panel, min_height=260, max_height=340),
            )

    st.subheader("Price Table")
    st.dataframe(
        _styled_frame(result.prices.tail(18), "{:.2f}"),
        use_container_width=True,
        height=420,
    )

    st.subheader("Feature Panel Sample")
    st.dataframe(
        _styled_frame(
            result.panel.tail(24)[["date", "ticker"] + result.feature_columns[:8] + ["target"]],
            {name: "{:.3f}" for name in result.feature_columns[:8]} | {"target": "{:.3%}"},
        ),
        use_container_width=True,
        hide_index=True,
        height=480,
    )

def main() -> None:
    _set_page_config()
    _mount_streamlit_theme_bridge()
    st.session_state["interface_theme"] = AUTO_THEME_NAME

    with st.sidebar:
        st.header("Workbench Controls")
        st.caption("Start with the reference dataset for a reproducible review, then switch to your own price table if needed.")
        st.markdown("**Data Source**")
        data_mode = st.radio("Data source", [REFERENCE_MARKET_LABEL, UPLOAD_PRICE_LABEL], index=0)

        if data_mode == REFERENCE_MARKET_LABEL:
            start_date = st.date_input("Start date", value=date(2020, 1, 1))
            end_date = st.date_input("End date", value=date(2025, 12, 31))
            seed = st.number_input("Random seed", min_value=1, max_value=999, value=14, step=1)
            prices = _cached_demo_prices(start_date, end_date, seed)
        else:
            uploaded_file = st.file_uploader("Upload price CSV", type=["csv"])
            prices = load_prices_from_csv(uploaded_file.getvalue()) if uploaded_file is not None else None

        if prices is not None:
            benchmark_default = list(prices.columns).index("SPY") if "SPY" in prices.columns else 0
            benchmark = st.selectbox("Benchmark", options=list(prices.columns), index=benchmark_default)
        else:
            benchmark = "SPY"

        st.markdown("---")
        st.markdown("**Strategy Backbone**")
        top_k = st.slider("Top-K assets", min_value=1, max_value=5, value=3)
        retrain_every = st.slider("Retrain cadence (days)", min_value=5, max_value=63, value=21)
        model_alpha = st.slider("Ridge alpha", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
        max_weight = st.slider("Max single-asset weight", min_value=0.10, max_value=0.70, value=0.40, step=0.05)
        st.markdown("**Risk Overlay**")
        target_vol = st.slider("Target annual vol", min_value=0.08, max_value=0.35, value=0.18, step=0.01)
        var_limit = st.slider("1-day VaR limit", min_value=0.01, max_value=0.06, value=0.025, step=0.005)
        drawdown_limit = st.slider("Drawdown guard trigger", min_value=0.05, max_value=0.25, value=0.12, step=0.01)
        drawdown_exposure = st.slider("Exposure after drawdown guard", min_value=0.10, max_value=0.60, value=0.35, step=0.05)
        tc_bps = st.slider("Transaction cost (bps)", min_value=0.0, max_value=30.0, value=8.0, step=1.0)

        with st.expander("Advanced Engine Settings", expanded=False):
            train_ratio = st.slider("Training history ratio", min_value=0.45, max_value=0.85, value=0.65, step=0.05)
            prediction_horizon = st.slider("Prediction horizon (days)", min_value=1, max_value=5, value=1, step=1)
            covariance_lookback = st.slider("Covariance lookback", min_value=20, max_value=126, value=60, step=5)
            var_confidence = st.slider("VaR confidence", min_value=0.90, max_value=0.99, value=0.95, step=0.01)

        with st.expander("Review Guide", expanded=False):
            st.markdown("1. Start in `Executive Overview` for performance and portfolio posture.")
            st.markdown("2. Use `Portfolio Lab` and `Risk Command` to review allocation, risk controls, and overlay events.")
            st.markdown("3. Use `Explainability Lab` and `Architecture And Data` to inspect drivers and export review materials.")
            st.markdown(f"4. Keep `{REFERENCE_MARKET_LABEL}` selected when you want a reproducible baseline run.")

        run_clicked = st.button(RUN_ACTION_LABEL, type="primary", use_container_width=True)

    _render_theme(AUTO_THEME_NAME)
    _build_brand_banner()
    _build_landing_note()
    _hero()
    _build_quickstart_guide()

    if run_clicked or ("prototype_result" not in st.session_state and prices is not None):
        st.session_state.pop("prototype_error", None)
        if prices is None:
            st.warning("Upload a price CSV or switch back to the reference market dataset.")
        else:
            config = PrototypeConfig(
                benchmark=benchmark,
                train_ratio=train_ratio,
                top_k=top_k,
                retrain_every=retrain_every,
                prediction_horizon=prediction_horizon,
                model_alpha=model_alpha,
                max_weight=max_weight,
                target_vol=target_vol,
                var_limit=var_limit,
                var_confidence=var_confidence,
                drawdown_limit=drawdown_limit,
                drawdown_exposure=drawdown_exposure,
                transaction_cost_bps=tc_bps,
                covariance_lookback=covariance_lookback,
            )
            try:
                with st.spinner("Running signal generation, risk overlay, and explainability analysis..."):
                    st.session_state["prototype_result"] = run_prototype(prices, config)
                    st.session_state["prototype_benchmark"] = benchmark
            except Exception as exc:
                st.session_state.pop("prototype_result", None)
                st.session_state["prototype_error"] = str(exc)

    error_message = st.session_state.get("prototype_error")
    if error_message:
        st.error(error_message)

    result: PrototypeResult | None = st.session_state.get("prototype_result")
    benchmark = st.session_state.get("prototype_benchmark", benchmark)

    if result is None:
        st.info(f"Keep the baseline settings and click {RUN_ACTION_LABEL} to populate the full research workbench.")
        st.markdown(
            """
            **Recommended first run**

            - Leave the reference synthetic market selected for the first run.
            - Click `Run Analysis` once and wait for the pipeline to finish.
            - Review the tabs from left to right for the clearest project walkthrough.
            """
        )
        st.stop()

    _build_workstation_header(result, benchmark)
    _build_market_tape(result, benchmark)
    _build_command_center(result, benchmark)

    tabs = st.tabs(
        [
            "Executive Overview",
            "Portfolio Lab",
            "Strategy Compare",
            "Risk Command",
            "Signal Monitor",
            "Explainability Lab",
            "Architecture And Data",
        ]
    )
    with tabs[0]:
        _build_overview_tab(result, benchmark)
    with tabs[1]:
        _build_portfolio_lab_tab(result, benchmark)
    with tabs[2]:
        _build_strategy_compare_tab(result, benchmark)
    with tabs[3]:
        _build_risk_tab(result)
    with tabs[4]:
        _build_signal_monitor_tab(result, benchmark)
    with tabs[5]:
        _build_explainability_tab(result)
    with tabs[6]:
        _build_architecture_and_data_tab(result)

    st.stop()
