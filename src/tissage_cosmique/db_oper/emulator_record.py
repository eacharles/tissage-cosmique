from macon.db_oper.base import TableContext, TableOperations

from .. import models
from ..db.emulator_record import EmulatorRecordTable


class EmulatorRecordOperations(
    TableOperations[EmulatorRecordTable, models.EmulatorRecord, models.EmulatorRecordCreate],
):
    """Table operations for emulator records."""


emulator_record = EmulatorRecordOperations(TableContext.from_db_class(EmulatorRecordTable))
