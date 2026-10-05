"""SQLite Database Connection and Session Management.

Provides the SQLAlchemy engine, session maker, base model, and dependency
for accessing the database throughout the FastAPI backend.
"""

from collections.abc import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.app.core.config import DATABASE_DIR, DATABASE_URL

# Ensure the database directory exists
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Create the SQLite SQLAlchemy Engine
# Note: check_same_thread=False is required for SQLite when accessed by multiple threads in FastAPI.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Set to True if detailed SQL query debugging is needed
)


# Enforce SQLite foreign key constraints
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints in SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Session factory for creating database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all declarative SQLAlchemy models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session per request.

    Yields:
        Session: Active SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables defined in SQLAlchemy models."""
    # Ensure database folder exists
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    # Import all models before calling create_all so that metadata is populated
    import backend.app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
