# Contributing to AutoML Platform

Thank you for your interest in contributing! Here's how to get started.

## Development setup

```bash
git clone https://github.com/yourusername/automl-platform.git
cd automl-platform
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # edit FLASK_SECRET_KEY
python app.py
```

## Running tests

```bash
pip install pytest
pytest test_app.py -v
```

All tests must pass before submitting a pull request.

## Coding standards

- **Python** — follow PEP 8. Run `flake8 app.py` before committing.
- **Commits** — use conventional commit messages: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- **One concern per PR** — keep pull requests focused and small.

## What to work on

Check the Issues tab for `good first issue` labels. Some good starting points:

- Adding new ML models to the classification/regression pipeline
- Improving the preprocessing pipeline (feature engineering options)
- Writing more tests
- UI/UX improvements

## Reporting bugs

Open an issue with: Python version, OS, steps to reproduce, expected vs actual behaviour, and the full traceback from the console.