"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mlaser_measles` python will execute
    ``__main__.py`` as a script. That means there will not be any
    ``laser_measles.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there"s no ``laser_measles.__main__`` in ``sys.modules``.

  Also see (1) from https://typer.tiangolo.com/tutorial/
"""

import typer
from typing import List

from .core import compute

app = typer.Typer()


@app.command()
def run(names: List[str] = typer.Argument(help="Names to process")):
    """Main command to run laser-measles CLI."""
    typer.echo(compute(names))


if __name__ == "__main__":
    app()
