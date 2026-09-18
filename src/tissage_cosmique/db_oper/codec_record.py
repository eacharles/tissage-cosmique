from macon.db_oper.base import TableContext, TableOperations

from .. import models
from ..db.codec_record import CodecRecordTable


class CodecRecordOperations(
    TableOperations[CodecRecordTable, models.CodecRecord, models.CodecRecordCreate],
):
    """Table operations for codec records."""


codec_record = CodecRecordOperations(TableContext.from_db_class(CodecRecordTable))
