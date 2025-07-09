"""
Tests for the main CLI functionality converted from Click to Typer.
"""

import pytest
from typer.testing import CliRunner
from laser_measles.cli import app


def test_cli_help():
    """Test that the main CLI help command works with typer."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_cli_run_command():
    """Test that the main CLI run command works with typer."""
    runner = CliRunner()
    result = runner.invoke(app, [])
    # Should complete without error (exact behavior depends on core.compute implementation)
    assert result.exit_code == 0


def test_cli_run_with_arguments():
    """Test that the main CLI accepts arguments."""
    runner = CliRunner()
    result = runner.invoke(app, ["arg1", "arg2"])
    # Should complete without error
    assert result.exit_code == 0