from macon.local_sync.base import SyncOperations

from .. import models
from ..db.tracer_config import TracerConfigTable
from ..local_async.tracer_config import tracer_config as tracer_config_async


class TracerConfigSyncOperations(
    SyncOperations[TracerConfigTable, models.TracerConfig, models.TracerConfigCreate],
):
    """Synchronous operations for tracer configurations."""


tracer_config = TracerConfigSyncOperations(tracer_config_async)
