"""Unit and integration tests for SQLite Database and SQLAlchemy Models."""

import uuid
import pytest
from sqlalchemy import inspect
from backend.app.core.database import SessionLocal, init_db, engine, Base
from backend.app.models.scan_session import ScanSession
from backend.app.models.finding import Finding


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure database tables are initialized before any tests run."""
    init_db()


@pytest.fixture(scope="function")
def db_session():
    """Provide a clean transactional session for testing."""
    session = SessionLocal()
    yield session
    session.close()


def test_database_tables_created():
    """Verify that scan_sessions and findings tables exist in SQLite."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    assert "scan_sessions" in table_names
    assert "findings" in table_names


def test_create_scan_session_and_finding(db_session):
    """Test inserting and retrieving a scan session with associated findings."""
    test_session_id = f"test-sess-{uuid.uuid4().hex[:8]}"

    # Create a ScanSession
    session_record = ScanSession(
        session_id=test_session_id,
        risk_score=45,
        risk_level="MEDIUM",
        total_findings=1,
        protected=False,
    )
    db_session.add(session_record)
    db_session.commit()
    db_session.refresh(session_record)

    assert session_record.id is not None
    assert session_record.session_id == test_session_id

    # Create an associated Finding
    finding_record = Finding(
        finding_id="finding-test-01",
        session_id=test_session_id,
        category="PHONE_NUMBER",
        severity="HIGH",
        confidence=0.95,
        description="A phone number was detected in the lower right.",
        source="OCR",
        x=100.0,
        y=200.0,
        width=150.0,
        height=30.0,
        recommended_action="BLUR",
    )
    db_session.add(finding_record)
    db_session.commit()

    # Query back the scan session with findings
    queried = (
        db_session.query(ScanSession)
        .filter(ScanSession.session_id == test_session_id)
        .first()
    )
    assert queried is not None
    assert len(queried.findings) == 1
    assert queried.findings[0].category == "PHONE_NUMBER"
    assert queried.findings[0].recommended_action == "BLUR"

    # Cleanup: Delete the scan session and verify cascade delete
    db_session.delete(queried)
    db_session.commit()

    # Verify findings were deleted due to cascade
    orphaned_findings = (
        db_session.query(Finding)
        .filter(Finding.session_id == test_session_id)
        .all()
    )
    assert len(orphaned_findings) == 0
