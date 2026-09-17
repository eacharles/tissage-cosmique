from macon.local_sync.base import SyncOperations

from .. import models
from ..db.cosmology import CosmologyParamsTable
from ..local_async.cosmology import cosmology_params as cosmology_params_async


class CosmologyParamsSyncOperations(
    SyncOperations[CosmologyParamsTable, models.CosmologyParams, models.CosmologyParamsCreate],
):
    """Synchronous operations for cosmological parameter sets."""


cosmology_params = CosmologyParamsSyncOperations(cosmology_params_async)
