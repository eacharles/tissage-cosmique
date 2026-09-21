from macon.local_async.base import LocalOperations

from .. import models
from ..db.tracer_config import TracerConfigTable
from ..db_oper.tracer_config import tracer_config as tracer_config_ops


class TracerConfigLocalOperations(
    LocalOperations[TracerConfigTable, models.TracerConfig, models.TracerConfigCreate],
):
    """Async session-managed operations for tracer configurations."""


tracer_config = TracerConfigLocalOperations(tracer_config_ops)
