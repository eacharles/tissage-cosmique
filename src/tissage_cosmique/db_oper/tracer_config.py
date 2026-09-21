from macon.db_oper.base import TableContext, TableOperations

from .. import models
from ..db.tracer_config import TracerConfigTable


class TracerConfigOperations(
    TableOperations[TracerConfigTable, models.TracerConfig, models.TracerConfigCreate],
):
    """Table operations for tracer configurations."""


tracer_config = TracerConfigOperations(TableContext.from_db_class(TracerConfigTable))
