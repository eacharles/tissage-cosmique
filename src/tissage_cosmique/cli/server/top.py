import click
import uvicorn


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8080, type=int, help="Port to listen on")
@click.option("--reload/--no-reload", default=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:  # noqa: FBT001
    """Start the tissage-cosmique API server."""
    uvicorn.run(  # pragma: no cover
        "tissage_cosmique.router.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
