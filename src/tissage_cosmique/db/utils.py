"""Database utility functions."""

import uuid


def uuid7() -> uuid.UUID:
    """Generate a UUID7 (time-ordered) as a standard uuid.UUID instance."""
    import uuid_utils

    return uuid.UUID(str(uuid_utils.uuid7()))
