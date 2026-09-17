import click
from macon.cli.local.base import make_table_group

from ...local_sync import cosmology_params


@click.group()
def cli() -> None:
    """Tissage-cosmique local database CLI."""


cli.add_command(make_table_group("cosmology-params", cosmology_params, "Manage cosmology parameter sets"))
