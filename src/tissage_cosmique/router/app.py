from fastapi import FastAPI
from macon.router.base import create_table_router

from ..local_async import codec_record, cosmology_params, emulator_record


def create_app() -> FastAPI:
    """Create the tissage-cosmique FastAPI application."""
    app = FastAPI(title="tissage-cosmique", description="Cosmological computations API")

    app.include_router(create_table_router("cosmology_params", cosmology_params))
    app.include_router(create_table_router("codec_record", codec_record))
    app.include_router(create_table_router("emulator_record", emulator_record))

    return app
