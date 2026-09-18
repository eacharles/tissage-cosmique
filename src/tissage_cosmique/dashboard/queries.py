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
    """Query all tisserande Execution records."""
    from tisserande.local_sync import execution

    rows = execution.get_rows()
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        d = _dump(r)
        if hasattr(r.status, "value"):
            d["status"] = r.status.value
        data.append(d)
    return pd.DataFrame(data)


def get_nodes_for_execution(exec_id: UUID) -> pd.DataFrame:
    """Query all nodes for a given execution."""
    from tisserande.local_sync import node

    rows = node.find_by(execution_id=exec_id)
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        d = {
            "id_": str(r.id_), "type_": str(r.type_),
            "arg_name": r.arg_name, "execution_id": str(r.execution_id),
        }
        if hasattr(r, "config_data") and r.config_data:
            d["value"] = str(r.config_data)[:100]
        elif hasattr(r, "value_json") and r.value_json is not None:
            v = r.value_json
            d["value"] = f"[{len(v)} items]" if isinstance(v, list) else str(v)[:100]
        else:
            d["value"] = ""
        data.append(d)
    return pd.DataFrame(data)


def get_edges_for_execution(exec_id: UUID) -> pd.DataFrame:
    """Query all edges for a given execution."""
    from tisserande.local_sync import edge

    rows = edge.find_by(execution_id=exec_id)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{"id_": r.id_, "from_id": str(r.from_id), "to_id": str(r.to_id)} for r in rows])


def _dump(row: object) -> dict:
    """Convert a Pydantic model to a dict with stringified UUIDs."""
    d = row.model_dump()  # type: ignore[attr-defined]
    for k, v in d.items():
        if isinstance(v, UUID):
            d[k] = str(v)
    return d
