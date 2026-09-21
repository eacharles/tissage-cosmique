from macon.db_oper.base import TableContext, TableOperations

from .. import models
from ..db.data_vector_config import DataVectorConfigTable


class DataVectorConfigOperations(
    TableOperations[DataVectorConfigTable, models.DataVectorConfig, models.DataVectorConfigCreate],
):
    """Table operations for data vector configurations."""


data_vector_config = DataVectorConfigOperations(TableContext.from_db_class(DataVectorConfigTable))
