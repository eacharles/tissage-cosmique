from macon.local_async.base import LocalOperations

from .. import models
from ..db.data_vector_config import DataVectorConfigTable
from ..db_oper.data_vector_config import data_vector_config as dv_config_ops


class DataVectorConfigLocalOperations(
    LocalOperations[DataVectorConfigTable, models.DataVectorConfig, models.DataVectorConfigCreate],
):
    """Async session-managed operations for data vector configurations."""


data_vector_config = DataVectorConfigLocalOperations(dv_config_ops)
