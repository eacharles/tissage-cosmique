from macon.local_sync.base import SyncOperations

from .. import models
from ..db.data_vector_config import DataVectorConfigTable
from ..local_async.data_vector_config import data_vector_config as dv_config_async


class DataVectorConfigSyncOperations(
    SyncOperations[DataVectorConfigTable, models.DataVectorConfig, models.DataVectorConfigCreate],
):
    """Synchronous operations for data vector configurations."""


data_vector_config = DataVectorConfigSyncOperations(dv_config_async)
