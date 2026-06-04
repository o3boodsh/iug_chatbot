"""
postgres_routes.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI routes for EduPredict PostgreSQL database.

أضف هذا الملف إلى مشروعك وسجّل الـ router في main.py:

    from postgres_routes import router as pg_router
    app.include_router(pg_router)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from postgres_db import (
    execute_query,
    list_tables,
    describe_table,
    ping_postgres,
)

router = APIRouter(prefix="/edupredict", tags=["EduPredict PostgreSQL"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RawQueryRequest(BaseModel):
    sql: str
    params: Optional[dict] = None


class RawQueryResponse(BaseModel):
    rows: List[dict]
    count: int


def _validate_table_name(table_name: str) -> None:
    if not table_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="اسم الجدول يحتوي على رموز غير مسموحة")


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
def pg_health():
    """التحقق من الاتصال بقاعدة بيانات PostgreSQL."""
    ok = ping_postgres()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL غير متاح حالياً")
    return {"status": "ok", "database": "edupredict_db"}


# ─── Schema exploration ────────────────────────────────────────────────────────

@router.get("/tables")
def get_tables():
    """قائمة جميع الجداول في قاعدة البيانات."""
    try:
        tables = list_tables()
        return {"tables": tables, "count": len(tables)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/tables/{table_name}/schema")
def get_table_schema(table_name: str):
    """عرض بنية (أعمدة) جدول معين."""
    try:
        columns = describe_table(table_name)
        return {"table": table_name, "columns": columns}
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"الجدول '{table_name}' غير موجود أو خطأ: {exc}",
        ) from exc


@router.get("/tables/{table_name}/rows")
def get_table_rows(
    table_name: str,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """
    استرجاع صفوف من جدول مع دعم الـ pagination.

    - limit:  عدد الصفوف (1–1000، افتراضي 50)
    - offset: ابدأ من الصف رقم X (افتراضي 0)
    """
    _validate_table_name(table_name)
    try:
        rows = execute_query(
            f'SELECT * FROM "{table_name}" LIMIT :lim OFFSET :off',
            {"lim": limit, "off": offset},
        )
        return {
            "table": table_name,
            "offset": offset,
            "limit": limit,
            "count": len(rows),
            "rows": rows,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── Raw query (careful — use only internally / with auth) ────────────────────

@router.post("/query", response_model=RawQueryResponse)
def raw_query(req: RawQueryRequest):
    """
    تنفيذ استعلام SQL مخصص (SELECT فقط).

    ⚠️  استخدم هذا الـ endpoint داخلياً فقط — لا تعرّضه للعموم.

    مثال:
        POST /edupredict/query
        { "sql": "SELECT * FROM students WHERE gpa > :g", "params": {"g": 3.5} }
    """
    sql_stripped = req.sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        raise HTTPException(
            status_code=400,
            detail="مسموح بـ SELECT فقط — لا يُسمح بـ INSERT/UPDATE/DELETE/DROP",
        )
    try:
        rows = execute_query(req.sql, req.params)
        return RawQueryResponse(rows=rows, count=len(rows))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
