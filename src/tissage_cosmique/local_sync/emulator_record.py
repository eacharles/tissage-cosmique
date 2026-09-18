from macon.local_sync.base import SyncOperations

from .. import models
from ..db.emulator_record import EmulatorRecordTable
from ..local_async.emulator_record import emulator_record as emulator_record_async


class EmulatorRecordSyncOperations(
    SyncOperations[EmulatorRecordTable, models.EmulatorRecord, models.EmulatorRecordCreate],
):
    """Synchronous operations for emulator records."""


emulator_record = EmulatorRecordSyncOperations(emulator_record_async)
