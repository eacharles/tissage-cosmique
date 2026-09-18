from macon.local_async.base import LocalOperations

from .. import models
from ..db.emulator_record import EmulatorRecordTable
from ..db_oper.emulator_record import emulator_record as emulator_record_ops


class EmulatorRecordLocalOperations(
    LocalOperations[EmulatorRecordTable, models.EmulatorRecord, models.EmulatorRecordCreate],
):
    """Async session-managed operations for emulator records."""


emulator_record = EmulatorRecordLocalOperations(emulator_record_ops)
