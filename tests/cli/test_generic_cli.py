"""
Tests for the generic CLI functionality converted from Click to Typer.
"""

import pytest
from typer.testing import CliRunner
from laser_measles.abm.cli import app


def test_generic_cli_help():
    """Test that the generic CLI help command works with typer."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_generic_cli_run_with_arguments():
    """Test that the generic CLI accepts arguments."""
    runner = CliRunner()
    result = runner.invoke(app, ["arg1", "arg2"])
    # Should complete without error
    assert result.exit_code == 0