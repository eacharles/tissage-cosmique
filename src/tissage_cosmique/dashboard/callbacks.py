"""Dash callbacks for interactivity across all tabs."""

from __future__ import annotations

from uuid import UUID

from dash import Input, Output, callback, html

from .components import make_execution_dag
from .queries import get_edges_for_execution, get_nodes_for_execution


@callback(
    Output("execution-detail", "children"),
    Input("execution-dropdown", "value"),
    prevent_initial_call=True,
)
def update_execution_detail(exec_id_str: str | None) -> html.Div:
    """Show nodes, edges, and DAG for a selected execution."""
    if not exec_id_str:
        return html.Div()

    exec_id = UUID(exec_id_str)
    nodes_df = get_nodes_for_execution(exec_id)
    edges_df = get_edges_for_execution(exec_id)

    children = [
        html.H5(f"Execution: {exec_id_str[:12]}..."),
        html.P(f"Nodes: {len(nodes_df)}, Edges: {len(edges_df)}"),
    ]

    if not nodes_df.empty:
        from .components import make_data_table

        children.append(html.H6("Nodes"))
        children.append(make_data_table(nodes_df, "exec-nodes-table", page_size=10))

        children.append(html.H6("Provenance Graph"))
        children.append(make_execution_dag(nodes_df, edges_df))

    return html.Div(children)
