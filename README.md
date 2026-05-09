# Responsible Quant Advisory Workbench

A team-developed Streamlit workbench for explainable, risk-governed portfolio recommendation research.

This repository packages a collaborative project from the Project Integration research team. It is framed as a decision-support environment rather than an autonomous trading product: the interface combines interpretable signal generation, portfolio construction, risk guardrails, and review-ready reporting in one place.

The project brings together three practical layers:

- a weight-centric portfolio construction backbone
- a risk and stress-testing layer with visible guardrails
- an explainability workflow that supports analyst review

The result is a reproducible local application that can backtest, surface portfolio risk, and document why the model selected each position.

## Highlights

- walk-forward cross-sectional return modeling
- conversion from model predictions into target portfolio weights
- risk overlays for max position, volatility targeting, VaR throttling, and drawdown guard
- benchmark-aware backtesting with turnover, cash buffer, and drawdown tracking
- explainability views for global importance, local contribution, prediction buckets, and what-if analysis
- downloadable deliverables for performance, decision logs, predictions, and memo export

## Repository Layout

```text
.
|-- app.py              # Streamlit entrypoint
|-- src/
|   `-- ai_quant_investing/
|       |-- __init__.py
|       |-- core/
|       |   |-- __init__.py
|       |   |-- data.py         # Demo data generation and CSV loading helpers
|       |   `-- pipeline.py     # Backtest, model training, and risk overlay pipeline
|       |-- explainability/
|       |   |-- __init__.py
|       |   `-- analysis.py     # Explainability and interpretation helpers
|       `-- ui/
|           |-- __init__.py
|           `-- app.py          # Streamlit layout, styling, and page composition
|-- tests/
|   |-- conftest.py
|   `-- test_smoke.py   # Smoke and regression tests
|-- .streamlit/         # Streamlit configuration
|-- pyproject.toml      # Project metadata and tool configuration
|-- requirements.txt    # Runtime dependencies
|-- requirements-dev.txt
`-- .github/            # CI and collaboration templates
```

## Quickstart

### Option 1: Anaconda Prompt

```bash
cd /d D:\Coding\2026\Project-integration
conda activate ai_quant_demo
pip install -r requirements-dev.txt
python -m streamlit run app.py
```

### Option 2: Plain Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m streamlit run app.py
```

The default mode runs on a reproducible synthetic multi-asset reference market. You can also upload a wide-form price CSV from the sidebar.

## Testing

Run the smoke suite from the repository root:

```bash
python -m pytest tests/test_smoke.py -q
```

## Expected CSV Format

```csv
date,SPY,QQQ,TLT,GLD
2023-01-03,380.82,265.13,101.44,171.24
2023-01-04,383.76,267.90,102.10,172.15
```

Recommendations:

- use at least two assets
- use close or adjusted close prices
- provide roughly 18 months or more of history for stable walk-forward features

## Design Principles

- Explainable by default: model interpretation is based on an interpretable Ridge model, permutation importance, local contribution decomposition, and one-factor what-if curves.
- Governance-aware: risk overlays, drawdown guardrails, and exportable decision logs keep model behavior reviewable.
- Reproducible by design: the application stays intentionally self-contained and avoids heavyweight market-data or explainability dependencies such as `bt`, `yfinance`, and `shap`.
- Team presentation ready: the interface is suitable for project review, analyst discussion, and stakeholder communication without being framed as a black-box trading bot.

## Development Notes

- `app.py` is the public Streamlit entrypoint.
- `src/ai_quant_investing/ui/` contains the Streamlit shell and dashboard behavior.
- `src/ai_quant_investing/core/` contains the data and portfolio engine.
- `src/ai_quant_investing/explainability/` contains model interpretation logic.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local development guidance.

## License

This repository is currently published under the [MIT License](LICENSE).
