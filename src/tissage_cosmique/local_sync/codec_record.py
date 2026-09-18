from macon.local_sync.base import SyncOperations

from .. import models
from ..db.codec_record import CodecRecordTable
from ..local_async.codec_record import codec_record as codec_record_async


class CodecRecordSyncOperations(
    SyncOperations[CodecRecordTable, models.CodecRecord, models.CodecRecordCreate],
):
    """Synchronous operations for codec records."""


codec_record = CodecRecordSyncOperations(codec_record_async)
