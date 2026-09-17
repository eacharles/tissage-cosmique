from macon.db_oper.base import TableContext, TableOperations

from .. import models
from ..db.cosmology import CosmologyParamsTable


class CosmologyParamsOperations(
    TableOperations[CosmologyParamsTable, models.CosmologyParams, models.CosmologyParamsCreate],
):
    """Table operations for cosmological parameter sets."""


cosmology_params = CosmologyParamsOperations(TableContext.from_db_class(CosmologyParamsTable))
