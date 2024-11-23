# MASER - Measles Simulation for ERadication

Spatial measles models implemented with the LASER toolkit.

## Development notes

For linting I find it useful to intall [pre-commit](https://pre-commit.com/) and [ruff](https://docs.astral.sh/ruff/) and then, before committing to github, running:

```bash
pre-commit run --all-file
ruff check
ruff check --fix
```
