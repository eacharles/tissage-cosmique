"""Dash application factory for the tissage-cosmique dashboard."""

from __future__ import annotations

import asyncio

from dash import Dash, dcc, html

from . import callbacks  # noqa: F401 — registers callbacks on import
from .components import (
    make_bar_chart,
    make_data_table,
    make_histogram,
    make_scatter_matrix,
    make_summary_cards,
)
from .queries import (
    get_codec_records,
    get_cosmology_params,
    get_emulator_records,
    get_executions,
)


def _init_db(db_url: str | None) -> None:
    """Initialize the database connection."""
    import tisserande.db  # noqa: F401

    from ..db.base import Base, close_db, get_session, init_db

    asyncio.run(close_db())
    init_db(db_url or "sqlite+aiosqlite:///tissage_cosmique.db")

    async def _ensure_tables() -> None:
        async with get_session() as session:
            conn = await session.connection()
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_ensure_tables())


def _tab_cosmology() -> dcc.Tab:
    df = get_cosmology_params()
    cosmo_cols = ["Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa"]
    return dcc.Tab(
        label="Cosmology Parameters",
        children=[
            html.Div(
                [
                    html.H3("Cosmology Parameter Sets"),
                    make_data_table(df, "cosmo-table"),
                    html.Hr(),
                    make_scatter_matrix(
                        df,
                        cosmo_cols,
                        title="Cosmological Parameter Space",
                    ),
                ],
                style={"padding": "15px"},
            )
        ],
    )


def _tab_provenance() -> dcc.Tab:
    df = get_executions()
    items = []
    if not df.empty:
        items = [
            ("Total Executions", len(df)),
            ("Successful", int((df["status"] == "success").sum()) if "status" in df else 0),
            ("Failed", int((df["status"] == "failure").sum()) if "status" in df else 0),
            ("Avg Duration", f"{df['duration_seconds'].mean():.3f}s" if "duration_seconds" in df else "N/A"),
        ]
    return dcc.Tab(
        label="Provenance Explorer",
        children=[
            html.Div(
                [
                    html.H3("Execution Provenance"),
                    make_summary_cards(items),
                    make_histogram(df, "duration_seconds", title="Execution Duration Distribution")
                    if not df.empty and "duration_seconds" in df
                    else html.P("No executions found."),
                    make_bar_chart(df, "status", "status", title="Executions by Status")
                    if not df.empty and "status" in df
                    else html.Div(),
                    html.Hr(),
                    html.H4("Select an execution to inspect its provenance graph:"),
                    dcc.Dropdown(
                        id="execution-dropdown",
                        options=[{"label": str(r), "value": str(r)} for r in df["id_"]]
                        if not df.empty and "id_" in df
                        else [],
                        placeholder="Select execution...",
                    ),
                    html.Div(id="execution-detail"),
                ],
                style={"padding": "15px"},
            )
        ],
    )


def _tab_emulators() -> dcc.Tab:
    df = get_emulator_records()
    display_cols = [
        "name", "emulator_type", "function_name",
        "training_score", "n_training_samples", "created_at",
    ]
    df_display = df[display_cols] if not df.empty and all(c in df for c in display_cols) else df
    return dcc.Tab(
        label="Emulator Registry",
        children=[
            html.Div(
                [
                    html.H3("Trained Emulators"),
                    make_data_table(df_display, "emulator-table"),
                    html.Hr(),
                    make_bar_chart(
                        df,
                        x="name",
                        y="training_score",
                        color="emulator_type",
                        title="Training Score by Emulator",
                    )
                    if not df.empty and "training_score" in df
                    else html.P("No emulators found."),
                    html.Div(id="emulator-detail"),
                ],
                style={"padding": "15px"},
            )
        ],
    )


def _tab_codecs() -> dcc.Tab:
    df = get_codec_records()
    display_cols = ["name", "codec_type", "n_latent", "n_features", "reconstruction_loss", "created_at"]
    df_display = df[display_cols] if not df.empty and all(c in df for c in display_cols) else df
    return dcc.Tab(
        label="Codec Registry",
        children=[
            html.Div(
                [
                    html.H3("Fitted Codecs"),
                    make_data_table(df_display, "codec-table"),
                    html.Hr(),
                    html.Div(id="codec-detail"),
                ],
                style={"padding": "15px"},
            )
        ],
    )


def create_dashboard(db_url: str | None = None) -> Dash:
    """Create the tissage-cosmique Dash dashboard.

    Parameters
    ----------
    db_url
        Database URL. Defaults to ``sqlite+aiosqlite:///tissage_cosmique.db``.
    """
    _init_db(db_url)

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "tissage-cosmique"

    app.layout = html.Div(
        [
            html.H1(
                "tissage-cosmique Dashboard",
                style={
                    "textAlign": "center", "padding": "15px",
                    "backgroundColor": "#1f77b4", "color": "white",
                },
            ),
            dcc.Tabs(
                [
                    _tab_cosmology(),
                    _tab_provenance(),
                    _tab_emulators(),
                    _tab_codecs(),
                ]
            ),
        ]
    )

    return app
