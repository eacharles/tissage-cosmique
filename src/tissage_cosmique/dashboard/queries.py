"""Data access layer: query DB tables and return pandas DataFrames."""

from __future__ import annotations

from uuid import UUID

import pandas as pd


def get_cosmology_params() -> pd.DataFrame:
    """Query all CosmologyParams records."""
    from ..local_sync import cosmology_params

    rows = cosmology_params.get_rows()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([_dump(r) for r in rows])


def get_emulator_records() -> pd.DataFrame:
    """Query all EmulatorRecord records."""
    from ..local_sync import emulator_record

    rows = emulator_record.get_rows()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([_dump(r) for r in rows])


def get_codec_records() -> pd.DataFrame:
    """Query all CodecRecord records."""
    from ..local_sync import codec_record

    rows = codec_record.get_rows()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([_dump(r) for r in rows])


def get_executions() -> pd.DataFrame:
    """Query all tisserande function call records (PythonFunctionCall nodes)."""
    from macon.models.filtering import Filter, FilterOp
    from tisserande.local_sync import node as node_ops

    rows = node_ops.filter_rows([Filter(field="node_type", op=FilterOp.EQ, value="python_function_call")])
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        d: dict = {"uid_": str(r.uid_), "node_type": r.node_type.value}
        if hasattr(r, "status"):
            d["status"] = r.status.value if hasattr(r.status, "value") else str(r.status)
        if hasattr(r, "start_time"):
            d["start_time"] = r.start_time
        if hasattr(r, "end_time"):
            d["end_time"] = r.end_time
        data.append(d)
    return pd.DataFrame(data)


def get_nodes_for_execution(func_node_uid: UUID) -> pd.DataFrame:
    """Query the function call node and all data nodes connected to it."""
    from macon.models.filtering import Filter, FilterOp
    from tisserande.local_sync import edge as edge_ops
    from tisserande.local_sync import node as node_ops

    input_edges = edge_ops.filter_rows([Filter(field="to_id", op=FilterOp.EQ, value=func_node_uid)])
    output_edges = edge_ops.filter_rows([Filter(field="from_id", op=FilterOp.EQ, value=func_node_uid)])

    all_node_ids = {e.from_id for e in input_edges} | {e.to_id for e in output_edges} | {func_node_uid}
    all_nodes = [node_ops.get_row(nid) for nid in all_node_ids]

    if not all_nodes:
        return pd.DataFrame()
    data = []
    for r in all_nodes:
        ntype = r.node_type.value if hasattr(r.node_type, "value") else str(r.node_type)
        d: dict = {"uid_": str(r.uid_), "node_type": ntype}
        if hasattr(r, "config_data") and r.config_data:
            d["value"] = str(r.config_data)[:100]
        elif hasattr(r, "value_json") and r.value_json is not None:
            v = r.value_json
            d["value"] = f"[{len(v)} items]" if isinstance(v, list) else str(v)[:100]
        else:
            d["value"] = ""
        data.append(d)
    return pd.DataFrame(data)


def get_edges_for_execution(func_node_uid: UUID) -> pd.DataFrame:
    """Query all edges connected to a function call node."""
    from macon.models.filtering import Filter, FilterOp
    from tisserande.local_sync import edge as edge_ops

    input_edges = edge_ops.filter_rows([Filter(field="to_id", op=FilterOp.EQ, value=func_node_uid)])
    output_edges = edge_ops.filter_rows([Filter(field="from_id", op=FilterOp.EQ, value=func_node_uid)])

    all_edges = input_edges + output_edges
    if not all_edges:
        return pd.DataFrame()
    return pd.DataFrame([{"id_": r.id_, "from_id": str(r.from_id), "to_id": str(r.to_id)} for r in all_edges])


def _dump(row: object) -> dict:
    """Convert a Pydantic model to a dict with stringified UUIDs."""
    d = row.model_dump()  # type: ignore[attr-defined]
    for k, v in d.items():
        if isinstance(v, UUID):
            d[k] = str(v)
    return d
