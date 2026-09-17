from macon.local_async.base import LocalOperations

from .. import models
from ..db.cosmology import CosmologyParamsTable
from ..db_oper.cosmology import cosmology_params as cosmology_params_ops


class CosmologyParamsLocalOperations(
    LocalOperations[CosmologyParamsTable, models.CosmologyParams, models.CosmologyParamsCreate],
):
    """Async session-managed operations for cosmological parameter sets."""


cosmology_params = CosmologyParamsLocalOperations(cosmology_params_ops)
