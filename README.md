# MASER - Measles Simulation for ERadication

Spatial measles models implemented with the LASER toolkit.

# Gettings started
(Using [uv](https://github.com/astral-sh/uv) for managing packages)

0. (optional) [Open in GitHub Codespaces](https://codespaces.new/InstituteforDiseaseModeling/laser-measles) and create a virtual environment:
```
pip3 install uv
uv venv
source .venv/bin/activate
uv pip install llvmlite numba
```
1. Install `laser_measles` (using uv)
```
uv pip install -e .
```

Once you have the above you can either code against the model and components in `laser-measles` or try the built-in commands, `measles` or `nigeria` (the `--help` option will give you some hints). For example:
```
measles --help
measles --verbose
```
 