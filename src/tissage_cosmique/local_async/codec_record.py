from macon.local_async.base import LocalOperations

from .. import models
from ..db.codec_record import CodecRecordTable
from ..db_oper.codec_record import codec_record as codec_record_ops


class CodecRecordLocalOperations(
    LocalOperations[CodecRecordTable, models.CodecRecord, models.CodecRecordCreate],
):
    """Async session-managed operations for codec records."""


codec_record = CodecRecordLocalOperations(codec_record_ops)
