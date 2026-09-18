"""Tests for server CLI entry point."""

from click.testing import CliRunner

from tissage_cosmique.cli.server.top import serve


class TestServerCLI:

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "tissage-cosmique" in result.output.lower()

    def test_has_options(self):
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output
