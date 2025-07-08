"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -m laser_measles.generic` python will execute
    ``__main__.py`` as a script. That means there will not be any
    ``laser_measles.generic.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there"s no ``laser_measles.generic.__main__`` in ``sys.modules``.

  Also see (1) from https://typer.tiangolo.com/tutorial/
"""


import typer

from .core import compute

app = typer.Typer()


@app.command()
def run(names: list[str]):
    """Main command to run laser-measles ABM CLI."""
    typer.echo(compute(names))


if __name__ == "__main__":
    app()
