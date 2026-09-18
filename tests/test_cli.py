"""Tests for CLI entry points."""

from click.testing import CliRunner

from tissage_cosmique.cli.local.top import cli


class TestLocalCLI:

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Tissage-cosmique" in result.output

    def test_cosmology_params_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cosmology-params", "--help"])
        assert result.exit_code == 0
        assert "cosmology parameter sets" in result.output.lower()

    def test_cosmology_params_subcommands(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cosmology-params", "--help"])
        assert "create" in result.output
        assert "get-row" in result.output
        assert "filter" in result.output
