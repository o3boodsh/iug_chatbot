"""
postgres_db.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PostgreSQL connection manager — EduPredict Database

قاعدة بيانات PostgreSQL منفصلة تماماً عن MongoDB.
تُستخدم للوصول إلى بيانات التوقعات الأكاديمية (edupredict_db).

يتم الوصول عبر SQLAlchemy (ORM) وpsycopg2 (raw queries).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://edupredict_user:G1Xsim5ArjBPnMtbgLr8wKCn8wBXeD0i"
    "@dpg-d8f9jfvavr4c73a2a2m0-a.oregon-postgres.render.com/edupredict_db",
)

# ── Module-level singletons ──────────────────────────────────────────────
_engine = None
_SessionLocal: Optional[sessionmaker] = None


# ── Internal helpers ─────────────────────────────────────────────────────

def _validate_env() -> None:
    if not DATABASE_URL:
        print("❌ [postgres_db] DATABASE_URL is not set in .env")
        sys.exit(1)


def _get_engine():
    """Lazy-init the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is not None:
        return _engine

    _validate_env()

    _engine = create_engine(
        DATABASE_URL,
        pool_size=5,           # max persistent connections
        max_overflow=10,       # extra connections when pool is full
        pool_timeout=30,       # seconds to wait for a connection
        pool_recycle=1800,     # recycle connections every 30 min
        pool_pre_ping=True,    # test connection before using it
        connect_args={
            "connect_timeout": 10,
            "application_name": "iug_chatbot",
        },
    )
    return _engine


def _get_session_factory() -> sessionmaker:
    """Return the session factory (singleton)."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


# ── Public API ───────────────────────────────────────────────────────────

@contextmanager
def get_session() -> Iterator[Session]:
    """
    Context manager that yields a SQLAlchemy session and handles
    commit / rollback / close automatically.

    Usage:
        with get_session() as session:
            rows = session.execute(text("SELECT * FROM students")).fetchall()
    """
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_query(sql: str, params: Optional[dict] = None) -> list[dict]:
    """
    Run a raw SQL SELECT and return results as a list of dicts.

    Args:
        sql:    Raw SQL string. Use :param_name for placeholders.
        params: Optional dict of bind parameters.

    Returns:
        List of row dicts (column → value).

    Example:
        rows = execute_query(
            "SELECT * FROM students WHERE faculty = :fac",
            {"fac": "Engineering"}
        )
    """
    with get_session() as session:
        result = session.execute(text(sql), params or {})
        keys = list(result.keys())
        return [dict(zip(keys, row)) for row in result.fetchall()]


def get_postgres_student(student_id: str) -> Optional[dict]:
    """Return a student row from the PostgreSQL students table."""
    rows = execute_query(
        """
        SELECT id_student, student_name, email, pin_hash
        FROM students
        WHERE id_student = :student_id
        LIMIT 1
        """,
        {"student_id": student_id},
    )
    return rows[0] if rows else None


def get_postgres_student_profile(student_id: str) -> Optional[dict]:
    """Return the student profile data needed by the chatbot."""
    student = get_postgres_student(student_id)
    if not student:
        return None

    enrollments = execute_query(
        """
        SELECT id, course_presentation_id, gender, region, highest_education,
               imd_band, age_band, num_of_prev_attempts, studied_credits,
               disability, final_result, date_registration, date_unregistration
        FROM enrollments
        WHERE id_student = :student_id
        ORDER BY created_at DESC
        LIMIT 5
        """,
        {"student_id": student_id},
    )
    predictions = execute_query(
        """
        SELECT p.day_of_course, p.risk_probability, p.risk_level, p.at_risk,
               p.recommended_action, p.created_at
        FROM predictions p
        JOIN enrollments e ON e.id = p.enrollment_id
        WHERE e.id_student = :student_id
        ORDER BY p.created_at DESC
        LIMIT 1
        """,
        {"student_id": student_id},
    )
    assessments = execute_query(
        """
        SELECT sa.id_assessment, sa.date_submitted, sa.score
        FROM student_assessments sa
        JOIN enrollments e ON e.id = sa.enrollment_id
        WHERE e.id_student = :student_id
        ORDER BY sa.date_submitted DESC
        LIMIT 5
        """,
        {"student_id": student_id},
    )

    return {
        "student": student,
        "enrollments": enrollments,
        "predictions": predictions,
        "assessments": assessments,
    }


def list_tables() -> list[str]:
    """Return all table names in the public schema."""
    inspector = inspect(_get_engine())
    return inspector.get_table_names(schema="public")


def describe_table(table_name: str) -> list[dict]:
    """
    Return column metadata for a given table.

    Returns a list of dicts with keys:
        name, type, nullable, default, primary_key
    """
    inspector = inspect(_get_engine())
    columns = inspector.get_columns(table_name, schema="public")
    pk_cols = set(
        inspector
        .get_pk_constraint(table_name, schema="public")
        .get("constrained_columns", [])
    )
    return [
        {
            "name":        col["name"],
            "type":        str(col["type"]),
            "nullable":    col["nullable"],
            "default":     col.get("default"),
            "primary_key": col["name"] in pk_cols,
        }
        for col in columns
    ]


def ping_postgres() -> bool:
    """
    Verify PostgreSQL is reachable.
    Returns True on success, False on failure.
    """
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
        return True
    except (OperationalError, SQLAlchemyError) as exc:
        print(f"❌ [postgres_db] ping failed: {exc}")
        return False


def close_postgres() -> None:
    """Dispose the engine pool on app shutdown."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
        print("🔌 [postgres_db] PostgreSQL connection pool closed.")
