from fastapi import FastAPI
from macon.router.base import create_table_router

from ..local_async import cosmology_params


def create_app() -> FastAPI:
    """Create the tissage-cosmique FastAPI application."""
    app = FastAPI(title="tissage-cosmique", description="Cosmological computations API")

    app.include_router(create_table_router("cosmology_params", cosmology_params))

    return app
