"""CLI interface for laser-measles."""

import typer

app = typer.Typer()


@app.command()
def main():
    """Laser Measles CLI - Coming Soon!"""
    typer.echo(
        """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                  LASER MEASLES CLI                           ║
    ║                                                              ║
    ║         ╭─────────────────────────────────────╮              ║
    ║         │                                     │              ║
    ║         │           COMING SOON!              │              ║
    ║         │                                     │              ║
    ║         │                                     │              ║
    ║         ╰─────────────────────────────────────╯              ║
    ║                                                              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    )


if __name__ == "__main__":
    app()
