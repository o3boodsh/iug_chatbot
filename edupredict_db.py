"""
edupredict_db.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PostgreSQL connection manager — EduPredict Database

Fetches student-specific data from PostgreSQL and converts
it into a structured dict ready to be injected as an
uploaded_file collection in the chatbot engine.

Tables used:
  students, course_presentations, enrollments,
  vle_sites, student_vle_events, assessments,
  student_assessments, academic_clocks, predictions,
  app_users, chat_sessions, admin_actions,
  system_settings, demo_enrollments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "EDUPREDICT_DATABASE_URL",
    "postgresql://edupredict_user:G1Xsim5ArjBPnMtbgLr8wKCn8wBXeD0i"
    "@dpg-d8f9jfvavr4c73a2a2m0-a.oregon-postgres.render.com/edupredict_db"
)

# ── Lazy import — psycopg2 loaded only when needed ───────────────────────
_psycopg2 = None


def _get_psycopg2():
    global _psycopg2
    if _psycopg2 is None:
        try:
            import psycopg2
            import psycopg2.extras
            _psycopg2 = psycopg2
        except ImportError:
            raise RuntimeError(
                "❌ psycopg2 غير مثبت — شغّل: pip install psycopg2-binary"
            )
    return _psycopg2


def _connect():
    """Open a new connection (caller is responsible for closing it)."""
    pg = _get_psycopg2()
    return pg.connect(DATABASE_URL, connect_timeout=10)


# ════════════════════════════════════════════════════════════════════════
#  PUBLIC HELPERS
# ════════════════════════════════════════════════════════════════════════

def ping_edupredict() -> bool:
    """Verify PostgreSQL is reachable. Returns True/False."""
    try:
        conn = _connect()
        conn.close()
        return True
    except Exception as exc:
        print(f"❌ [edupredict_db] ping failed: {exc}")
        return False


def fetch_student_by_id(student_id: str) -> Optional[dict]:
    """
    Return a student row from the `students` table, or None if not found.
    Also checks `app_users` for login credentials.
    """
    try:
        conn = _connect()
        pg   = _get_psycopg2()
        with conn.cursor(cursor_factory=pg.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM students WHERE id_student = %s LIMIT 1",
                (student_id,)
            )
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as exc:
        print(f"❌ [edupredict_db] fetch_student_by_id failed: {exc}")
        return None


def fetch_student_by_username(username: str) -> Optional[dict]:
    """
    Look up a student via app_users table (username → student link).
    Returns the app_user row merged with the student row, or None.
    """
    try:
        conn = _connect()
        pg   = _get_psycopg2()
        with conn.cursor(cursor_factory=pg.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM app_users WHERE username = %s LIMIT 1",
                (username,)
            )
            user_row = cur.fetchone()
            if not user_row:
                conn.close()
                return None

            student_id = user_row.get("student_id") or user_row.get("id_student")
            student_row = None
            if student_id:
                cur.execute(
                    "SELECT * FROM students WHERE id_student = %s LIMIT 1",
                    (student_id,)
                )
                student_row = cur.fetchone()

        conn.close()
        result = dict(user_row)
        if student_row:
            result.update(dict(student_row))
        return result
    except Exception as exc:
        print(f"❌ [edupredict_db] fetch_student_by_username failed: {exc}")
        return None


def fetch_student_full_profile(student_id) -> dict:
    """
    Gather all data for one student across every relevant table.

    Returns a single dict with keys:
      student_info, enrollments, assessments, vle_activity,
      predictions, academic_clock, chat_sessions
    """
    try:
        conn = _connect()
        pg   = _get_psycopg2()

        def q(sql, params=()):
            with conn.cursor(cursor_factory=pg.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]

        # ── Core student info ────────────────────────────────────────────
        students = q(
            "SELECT * FROM students WHERE id_student = %s",
            (student_id,)
        )
        student_info = students[0] if students else {}

        # ── Enrollments + course presentations ───────────────────────────
        enrollments = q(
            """
            SELECT e.*, cp.code_module, cp.code_presentation,
                   cp.length, cp.start_date
            FROM   enrollments e
            LEFT JOIN course_presentations cp
                   ON e.code_module = cp.code_module
                  AND e.code_presentation = cp.code_presentation
            WHERE  e.id_student = %s
            ORDER  BY cp.start_date DESC
            """,
            (student_id,)
        )

        # ── Assessments ──────────────────────────────────────────────────
        assessments = q(
            """
            SELECT sa.*, a.assessment_type, a.date AS due_date,
                   a.weight, a.code_module, a.code_presentation
            FROM   student_assessments sa
            LEFT JOIN assessments a ON sa.id_assessment = a.id_assessment
            WHERE  sa.id_student = %s
            ORDER  BY a.date DESC
            """,
            (student_id,)
        )

        # ── VLE activity summary (top 20 most-visited sites) ─────────────
        vle_activity = q(
            """
            SELECT sve.code_module, sve.code_presentation,
                   vs.activity_type, vs.site_name,
                   SUM(sve.sum_click) AS total_clicks,
                   COUNT(DISTINCT sve.date) AS active_days
            FROM   student_vle_events sve
            LEFT JOIN vle_sites vs ON sve.id_site = vs.id_site
            WHERE  sve.id_student = %s
            GROUP  BY sve.code_module, sve.code_presentation,
                      vs.activity_type, vs.site_name
            ORDER  BY total_clicks DESC
            LIMIT  20
            """,
            (student_id,)
        )

        # ── Predictions ──────────────────────────────────────────────────
        predictions = q(
            """
            SELECT * FROM predictions
            WHERE  id_student = %s
            ORDER  BY created_at DESC
            LIMIT  10
            """,
            (student_id,)
        )

        # ── Academic clock ───────────────────────────────────────────────
        academic_clock = q(
            "SELECT * FROM academic_clocks WHERE id_student = %s ORDER BY recorded_at DESC LIMIT 5",
            (student_id,)
        )

        # ── Demo enrollments (if any) ────────────────────────────────────
        demo_enrollments = q(
            "SELECT * FROM demo_enrollments WHERE id_student = %s",
            (student_id,)
        )

        conn.close()

        return {
            "student_info":     student_info,
            "enrollments":      enrollments,
            "assessments":      assessments,
            "vle_activity":     vle_activity,
            "predictions":      predictions,
            "academic_clock":   academic_clock,
            "demo_enrollments": demo_enrollments,
        }

    except Exception as exc:
        print(f"❌ [edupredict_db] fetch_student_full_profile failed: {exc}")
        return {}


def verify_student_pin(student_id: str, pin: str) -> Optional[dict]:
    """
    Verify login credentials from app_users.
    Accepts matching on: username, student_id, or id_student columns.
    Returns the app_user row on success, None on failure.
    """
    try:
        conn = _connect()
        pg   = _get_psycopg2()
        with conn.cursor(cursor_factory=pg.extras.RealDictCursor) as cur:
            # Try to find the user by student_id or username
            cur.execute(
                """
                SELECT * FROM app_users
                WHERE (student_id::text = %s OR username = %s)
                  AND password_hash = %s
                LIMIT 1
                """,
                (str(student_id), str(student_id), pin)
            )
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as exc:
        print(f"❌ [edupredict_db] verify_student_pin failed: {exc}")
        return None


def build_student_collection_name(student_id) -> str:
    """Return the uploaded-file collection name for a student's data."""
    return f"student_{student_id}_edupredict"


def profile_to_documents(profile: dict) -> list:
    """
    Convert the dict returned by fetch_student_full_profile()
    into a list of documents ready to be inserted into MongoDB
    as an uploaded_files collection.

    Each logical section becomes a separate document so the
    semantic chunker can index them independently.
    """
    docs = []

    # ── 1. Student basic info ────────────────────────────────────────────
    info = profile.get("student_info", {})
    if info:
        docs.append({
            "section": "معلومات الطالب الأساسية",
            "id_student":          info.get("id_student"),
            "gender":              info.get("gender"),
            "region":              info.get("region"),
            "highest_education":   info.get("highest_education"),
            "imd_band":            info.get("imd_band"),
            "age_band":            info.get("age_band"),
            "num_of_prev_attempts": info.get("num_of_prev_attempts"),
            "studied_credits":     info.get("studied_credits"),
            "disability":          info.get("disability"),
        })

    # ── 2. Enrollments ───────────────────────────────────────────────────
    for en in profile.get("enrollments", []):
        docs.append({
            "section":             "التسجيل في المساق",
            "id_student":          en.get("id_student"),
            "code_module":         en.get("code_module"),
            "code_presentation":   en.get("code_presentation"),
            "final_result":        en.get("final_result"),
            "date_registration":   str(en.get("date_registration", "")),
            "date_unregistration": str(en.get("date_unregistration", "")),
            "module_length":       en.get("length"),
            "module_start_date":   str(en.get("start_date", "")),
        })

    # ── 3. Assessments ───────────────────────────────────────────────────
    for a in profile.get("assessments", []):
        docs.append({
            "section":           "التقييمات والدرجات",
            "id_student":        a.get("id_student"),
            "id_assessment":     a.get("id_assessment"),
            "code_module":       a.get("code_module"),
            "code_presentation": a.get("code_presentation"),
            "assessment_type":   a.get("assessment_type"),
            "date_submitted":    str(a.get("date_submitted", "")),
            "due_date":          str(a.get("due_date", "")),
            "score":             a.get("score"),
            "weight":            a.get("weight"),
            "is_banked":         a.get("is_banked"),
        })

    # ── 4. VLE Activity ──────────────────────────────────────────────────
    for v in profile.get("vle_activity", []):
        docs.append({
            "section":           "نشاط المنصة الإلكترونية",
            "id_student":        info.get("id_student"),
            "code_module":       v.get("code_module"),
            "code_presentation": v.get("code_presentation"),
            "activity_type":     v.get("activity_type"),
            "site_name":         v.get("site_name"),
            "total_clicks":      v.get("total_clicks"),
            "active_days":       v.get("active_days"),
        })

    # ── 5. Predictions ───────────────────────────────────────────────────
    for p in profile.get("predictions", []):
        docs.append({
            "section":              "توقعات الأداء الأكاديمي",
            "id_student":           p.get("id_student"),
            "code_module":          p.get("code_module"),
            "code_presentation":    p.get("code_presentation"),
            "predicted_result":     p.get("predicted_result"),
            "confidence_score":     p.get("confidence_score"),
            "risk_level":           p.get("risk_level"),
            "recommendation":       p.get("recommendation"),
            "prediction_date":      str(p.get("created_at", "")),
        })

    # ── 6. Academic Clock ────────────────────────────────────────────────
    for ac in profile.get("academic_clock", []):
        docs.append({
            "section":           "الساعة الأكاديمية",
            "id_student":        ac.get("id_student"),
            "total_hours":       ac.get("total_hours"),
            "completed_modules": ac.get("completed_modules"),
            "in_progress":       ac.get("in_progress"),
            "recorded_at":       str(ac.get("recorded_at", "")),
        })

    # ── 7. Demo Enrollments ──────────────────────────────────────────────
    for de in profile.get("demo_enrollments", []):
        docs.append({
            "section":           "التسجيل التجريبي",
            "id_student":        de.get("id_student"),
            "code_module":       de.get("code_module"),
            "code_presentation": de.get("code_presentation"),
            "status":            de.get("status"),
        })

    return docs
