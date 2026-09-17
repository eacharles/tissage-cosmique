"""Database base class and session management, re-exported from macon.

All tissage-cosmique ORM models inherit from this Base. Session management
(init_db, get_session, close_db) is provided by macon's async engine.
"""

from macon.db.base import Base
from macon.db.session import close_db, get_session, init_db

__all__ = ["Base", "init_db", "get_session", "close_db"]
