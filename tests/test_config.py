"""Tests for configuration module."""

from tissage_cosmique.config import Configuration, DatabaseConfiguration, EmulatorConfiguration, get_config


class TestConfiguration:

    def test_default_db_url(self):
        config = Configuration()
        assert "sqlite" in config.db.url
        assert "tissage_cosmique" in config.db.url

    def test_default_db_echo(self):
        config = Configuration()
        assert config.db.echo is False

    def test_default_emulator_disabled(self):
        config = Configuration()
        assert config.emulator.enabled is False

    def test_database_configuration(self):
        db = DatabaseConfiguration(url="sqlite:///test.db", echo=True)
        assert db.url == "sqlite:///test.db"
        assert db.echo is True

    def test_emulator_configuration(self):
        emu = EmulatorConfiguration(enabled=True)
        assert emu.enabled is True

    def test_get_config_returns_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
