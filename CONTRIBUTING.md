# Contributing to MergeLens

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/shuhulx/mergelens.git
cd mergelens
python -m pip install -e ".[dev,all]"
pre-commit install
```

## Running Tests

```bash
make test        # run tests
make lint        # run linters
make format      # auto-format
make ci          # static checks, tests, and package build
```

## Pull Requests

1. Fork the repo and create a branch
2. Make your changes
3. Run `make ci` to verify
4. Submit a PR

Please open an issue first for major changes.
