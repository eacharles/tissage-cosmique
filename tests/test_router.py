"""Tests for FastAPI router creation."""

from tissage_cosmique.router.app import create_app


class TestRouterApp:

    def test_create_app(self):
        app = create_app()
        assert app.title == "tissage-cosmique"

    def test_cosmology_params_routes_in_schema(self):
        app = create_app()
        schema = app.openapi()
        paths = list(schema["paths"].keys())
        assert any("cosmology_params" in p for p in paths)
