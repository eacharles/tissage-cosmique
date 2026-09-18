"""Reusable Dash/Plotly components for the dashboard."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, dcc, html

import pandas as pd


def make_data_table(
    df: pd.DataFrame,
    table_id: str,
    *,
    page_size: int = 20,
) -> Any:
    """Create a sortable, filterable Dash DataTable."""
    if df.empty:
        return dash_table.DataTable(id=table_id, data=[], columns=[])

    columns = [{"name": c, "id": c} for c in df.columns]
    return dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=columns,
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        row_selectable="single",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
        ],
    )


def make_scatter_matrix(
    df: pd.DataFrame,
    columns: list[str],
    *,
    color: str | None = None,
    title: str = "",
) -> dcc.Graph:
    """Create a Plotly scatter matrix."""
    if df.empty:
        return dcc.Graph(figure=go.Figure())

    valid_cols = [c for c in columns if c in df.columns]
    fig = px.scatter_matrix(df, dimensions=valid_cols, color=color, title=title)
    fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    fig.update_layout(height=600)
    return dcc.Graph(figure=fig)


def make_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    title: str = "",
) -> dcc.Graph:
    """Create a Plotly bar chart."""
    if df.empty:
        return dcc.Graph(figure=go.Figure())

    fig = px.bar(df, x=x, y=y, color=color, title=title)
    fig.update_layout(height=400)
    return dcc.Graph(figure=fig)


def make_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    nbins: int = 30,
    title: str = "",
) -> dcc.Graph:
    """Create a Plotly histogram."""
    if df.empty:
        return dcc.Graph(figure=go.Figure())

    fig = px.histogram(df, x=column, nbins=nbins, title=title)
    fig.update_layout(height=350)
    return dcc.Graph(figure=fig)


def make_summary_cards(items: list[tuple[str, Any]]) -> html.Div:
    """Create a row of summary metric cards."""
    cards = []
    for label, value in items:
        cards.append(
            html.Div(
                [
                    html.H4(str(value), style={"margin": "0"}),
                    html.P(label, style={"margin": "0", "color": "#666"}),
                ],
                style={
                    "display": "inline-block",
                    "padding": "15px 25px",
                    "margin": "5px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "minWidth": "120px",
                },
            )
        )
    return html.Div(cards, style={"display": "flex", "flexWrap": "wrap", "marginBottom": "15px"})


def make_execution_dag(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> dcc.Graph:
    """Create a simple DAG visualization for an execution's provenance."""
    if nodes_df.empty:
        return dcc.Graph(figure=go.Figure())

    node_ids = list(nodes_df["id_"])
    node_labels = [
        f"{row.get('type_', '?')}\n{row.get('arg_name', '')}" for _, row in nodes_df.iterrows()
    ]

    n = len(node_ids)
    x_pos = list(range(n))
    y_pos = [0] * n

    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for _, row in edges_df.iterrows():
        src = node_id_to_idx.get(row["from_id"])
        dst = node_id_to_idx.get(row["to_id"])
        if src is not None and dst is not None:
            edge_x.extend([x_pos[src], x_pos[dst], None])
            edge_y.extend([y_pos[src], y_pos[dst], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"), hoverinfo="none")
    )
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=y_pos,
            mode="markers+text",
            marker=dict(size=20, color="#1f77b4"),
            text=node_labels,
            textposition="top center",
            hoverinfo="text",
        )
    )
    fig.update_layout(
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return dcc.Graph(figure=fig)
