# MASER - Measles Simulation for ERadication

Spatial measles models implemented with the LASER toolkit.

## Setup
Example using [uv](https://github.com/astral-sh/uv):

0. Create and activate virtual environment
```bash
uv venv
source .venv/bin/activate
```
1. Install
```bash
uv pip install -e .
```
2. Test that the model runs (`measles --help` for options)
```bash
measles
```

## Development notes

For linting I find it useful to intall [pre-commit](https://pre-commit.com/) and [ruff](https://docs.astral.sh/ruff/) and then, before committing to github, running:

```bash
pre-commit run --all-file
ruff check
ruff check --fix
```
