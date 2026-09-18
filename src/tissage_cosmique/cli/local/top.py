import click
from macon.cli.local.base import make_table_group

from ...local_sync import codec_record, cosmology_params, emulator_record


@click.group()
def cli() -> None:
    """Tissage-cosmique local database CLI."""


cli.add_command(make_table_group("cosmology-params", cosmology_params, "Manage cosmology parameter sets"))
cli.add_command(make_table_group("codec-record", codec_record, "Manage codec records"))
cli.add_command(make_table_group("emulator-record", emulator_record, "Manage emulator records"))
