import click


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8050, type=int, help="Port to listen on")
@click.option("--db-url", default=None, help="Database URL")
@click.option("--debug/--no-debug", default=True, help="Enable debug mode")
def dashboard(host: str, port: int, db_url: str | None, debug: bool) -> None:  # noqa: FBT001
    """Launch the tissage-cosmique dashboard."""
    from tissage_cosmique.dashboard import create_dashboard

    app = create_dashboard(db_url=db_url)
    app.run(host=host, port=port, debug=debug)  # pragma: no cover
