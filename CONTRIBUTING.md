# Contributing

Thanks for contributing to this project.

## Local Setup

1. Create or activate a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Start the app locally:

```bash
python -m streamlit run app.py
```

## Testing

Run the smoke suite before opening a pull request:

```bash
python -m pytest tests/test_smoke.py -q
```

## Coding Guidelines

- Keep the dashboard presentation-quality, reproducible, and stable offline.
- Prefer simple, interpretable logic over heavyweight abstractions.
- Avoid introducing network-bound data dependencies without a strong reason.
- Preserve the current visual language unless the change clearly improves the product.

## Pull Requests

Please include:

- a short summary of what changed
- screenshots or notes for UI changes
- test results or a clear explanation if tests were not run

## Scope

Good contributions include:

- bug fixes
- UI polish
- reliability improvements
- test coverage
- documentation upgrades

Please avoid large structural refactors unless they are discussed first.
