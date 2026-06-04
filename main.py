from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os, hashlib, hmac, base64, json, time

from chatbot_core import IUGChatbot
from database import ping, close as db_close, get_collection, COL_RANKINGS

# استيراد مدير قاعدة بيانات الملفات المرفوعة
from uploaded_files_db import ping_uploaded, close_uploaded, list_uploaded_collections

# ← جديد: استيراد مدير قاعدة بيانات EduPredict (PostgreSQL)
from edupredict_db import (
    ping_edupredict,
    fetch_student_by_id,
    fetch_student_by_username,
    verify_student_pin,
    fetch_student_full_profile,
    build_student_collection_name,
    profile_to_documents,
)

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IUG Chatbot API",
    description="الجامعة الإسلامية بغزة — Fast Retrieval + Semantic Search + LLM + EduPredict (v6)",
    version="6.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Core Engine Instance ─────────────────────────────────────────────────────
chatbot = IUGChatbot()

# ─── Token helpers ────────────────────────────────────────────────────────────
_SECRET = os.getenv("TOKEN_SECRET", "iug-chatbot-secret-key-change-me")


def _make_token(student_id: str, source: str = "iug") -> str:
    """
    source: "iug"        → طالب من MongoDB (النظام الأصلي)
            "edupredict" → طالب من EduPredict PostgreSQL
    """
    payload = json.dumps({
        "sid":    student_id,
        "src":    source,
        "exp":    int(time.time()) + 86400,
    })
    b64 = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(_SECRET.encode(), b64.encode(), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _verify_token(token: str) -> Optional[dict]:
    """
    Returns {"sid": student_id, "src": source} on success, None on failure.
    Keeps backward-compat with old tokens (no "src" key → default "iug").
    """
    try:
        b64, sig = token.rsplit(".", 1)
        expected = hmac.new(_SECRET.encode(), b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(b64).decode())
        if payload["exp"] < int(time.time()):
            return None
        return {
            "sid": payload["sid"],
            "src": payload.get("src", "iug"),
        }
    except Exception:
        return None


# ─── Schemas ──────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    student_id: str
    pin: str


class LoginResponse(BaseModel):
    token:        str
    student_name: str
    message:      str
    source:       str   # "iug" | "edupredict"


class ChatRequest(BaseModel):
    question:   str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer:     str
    session_id: str
    top_chunks: List[str]


class HistoryResponse(BaseModel):
    session_id: str
    history:    list


class FileChatRequest(BaseModel):
    question:        str
    collection_name: str
    session_id:      Optional[str] = "default"


class FileChatResponse(BaseModel):
    answer:          str
    session_id:      str
    collection_name: str
    top_chunks:      List[str]
    source:          str


# ← جديد: schema لسؤال الطالب عن بياناته الأكاديمية (EduPredict)
class StudentChatRequest(BaseModel):
    question:   str
    session_id: str   # يجب أن يكون token (يحمل student_id)


class StudentChatResponse(BaseModel):
    answer:     str
    session_id: str
    top_chunks: List[str]
    source:     str


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    print("⏳ Verifying MongoDB connection…")
    if not ping():
        raise RuntimeError(
            "Cannot connect to MongoDB. "
            "Check MONGO_URI in .env and whitelist your IP in Atlas."
        )
    print("✅ MongoDB connected.")

    print("⏳ Verifying uploaded_files DB connection…")
    if not ping_uploaded():
        print("⚠️  uploaded_files DB not reachable — file upload feature disabled.")
    else:
        print("✅ uploaded_files DB connected.")

    # ← جديد: فحص اتصال EduPredict PostgreSQL
    print("⏳ Verifying EduPredict PostgreSQL connection…")
    if not ping_edupredict():
        print("⚠️  EduPredict DB not reachable — student academic data feature disabled.")
    else:
        print("✅ EduPredict PostgreSQL connected.")

    chatbot.initialize()


# ─── Shutdown ─────────────────────────────────────────────────────────────────
@app.on_event("shutdown")
def shutdown_event():
    db_close()
    close_uploaded()


# ════════════════════════════════════════════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """
    تسجيل الدخول الموحّد:
      1. يبحث أولاً في MongoDB (COL_RANKINGS) — النظام الأصلي.
      2. إن لم يجد → يبحث في EduPredict PostgreSQL.
    عند نجاح تسجيل دخول EduPredict → يُحمَّل profile الطالب تلقائياً.
    """
    # ── 1. محاولة MongoDB (النظام الأصلي) ─────────────────────────────────
    student = get_collection(COL_RANKINGS).find_one({"_id": req.student_id})
    if student:
        stored_pin = str(student.get("pin", ""))
        if not stored_pin or stored_pin != req.pin.strip():
            raise HTTPException(status_code=401, detail="PIN غير صحيح")
        token = _make_token(req.student_id, source="iug")
        return LoginResponse(
            token        = token,
            student_name = student.get("student_name", ""),
            message      = f"أهلاً {student.get('student_name', '')} 👋",
            source       = "iug",
        )

    # ── 2. محاولة EduPredict PostgreSQL ───────────────────────────────────
    ep_student = fetch_student_by_id(req.student_id)
    if not ep_student:
        raise HTTPException(status_code=401, detail="رقم الهوية غير موجود")

    # التحقق من PIN — يُجرَّب أولاً كـ password_hash مباشر
    # (يمكن تطوير هذا لاحقاً لدعم bcrypt أو SHA256)
    stored_pin = str(ep_student.get("pin", ep_student.get("password_hash", "")))
    if stored_pin != req.pin.strip():
        raise HTTPException(status_code=401, detail="PIN غير صحيح")

    # ── تحميل بيانات الطالب من EduPredict إلى uploaded_files ─────────────
    _load_edupredict_student(req.student_id)

    student_name = (
        ep_student.get("student_name")
        or ep_student.get("username")
        or str(req.student_id)
    )
    token = _make_token(req.student_id, source="edupredict")
    return LoginResponse(
        token        = token,
        student_name = student_name,
        message      = f"أهلاً {student_name} 👋",
        source       = "edupredict",
    )


# ── Helper: تحميل بيانات EduPredict للطالب ───────────────────────────────────
def _load_edupredict_student(student_id) -> bool:
    """
    جلب الـ profile كاملاً من PostgreSQL وتخزينه كـ uploaded_file.
    يُعيد True عند النجاح.
    """
    try:
        profile  = fetch_student_full_profile(student_id)
        if not profile:
            return False
        documents = profile_to_documents(profile)
        if not documents:
            return False
        col_name = build_student_collection_name(student_id)
        chatbot.upload_json_file(col_name, documents)
        print(f"✅ EduPredict profile loaded for student {student_id} → '{col_name}'")
        return True
    except Exception as exc:
        print(f"⚠️  Could not load EduPredict data for {student_id}: {exc}")
        return False


# ════════════════════════════════════════════════════════════════════════════════
#  CHAT ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    المحادثة العامة (بيانات الجامعة).
    إذا كان الطالب من EduPredict → تُحقَن بياناته الشخصية تلقائياً في الـ context.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً")

    resolved_id = req.session_id
    src = "iug"

    if req.session_id and "." in req.session_id:
        decoded = _verify_token(req.session_id)
        if decoded is None:
            raise HTTPException(
                status_code=401,
                detail="انتهت صلاحية الجلسة — سجّل دخولك مجدداً"
            )
        resolved_id = decoded["sid"]
        src         = decoded.get("src", "iug")

    try:
        result = chatbot.chat(req.question, resolved_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM غير متاح: {e}")

    return ChatResponse(
        answer     = result["answer"],
        session_id = req.session_id,
        top_chunks = result["top_chunks"],
    )


@app.post("/chat/student", response_model=StudentChatResponse)
def chat_with_student_data(req: StudentChatRequest):
    """
    ← جديد: محادثة مع البيانات الأكاديمية الشخصية للطالب (EduPredict).

    - يتطلب token صالح محتوياً على student_id.
    - يُجيب فقط من بيانات الطالب المحدد — لا يخلط مع بيانات الجامعة العامة.
    - إذا لم تكن البيانات محملة في الذاكرة → يحاول جلبها من PostgreSQL تلقائياً.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً")

    if "." not in req.session_id:
        raise HTTPException(status_code=401, detail="token غير صالح")

    decoded = _verify_token(req.session_id)
    if decoded is None:
        raise HTTPException(
            status_code=401,
            detail="انتهت صلاحية الجلسة — سجّل دخولك مجدداً"
        )

    student_id = decoded["sid"]
    col_name   = build_student_collection_name(student_id)

    # إذا لم تكن بيانات الطالب محملة → حملها الآن
    if col_name not in chatbot._uploaded_chunks:
        loaded = _load_edupredict_student(student_id)
        if not loaded:
            raise HTTPException(
                status_code=404,
                detail="لا تتوفر بيانات أكاديمية لحسابك. يرجى التواصل مع الدعم."
            )

    try:
        result = chatbot.chat_with_file(
            question        = req.question,
            collection_name = col_name,
            session_id      = student_id,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"خطأ في المعالجة: {e}")

    return StudentChatResponse(
        answer     = result["answer"],
        session_id = req.session_id,
        top_chunks = result["top_chunks"],
        source     = "edupredict",
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history_route(session_id: str):
    resolved_id = session_id
    if "." in session_id:
        decoded = _verify_token(session_id)
        if decoded:
            resolved_id = decoded["sid"]
    return HistoryResponse(
        session_id = session_id,
        history    = chatbot.get_history(resolved_id),
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    resolved_id = session_id
    if "." in session_id:
        decoded = _verify_token(session_id)
        if decoded:
            resolved_id = decoded["sid"]
    chatbot.clear_history(resolved_id)
    return {"message": f"تم مسح محادثة {session_id}"}


# ════════════════════════════════════════════════════════════════════════════════
#  UNIVERSITY DATA ROUTES (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "IUG Chatbot API v6 — Fast Retrieval + Semantic Search + LLM + EduPredict 🎓",
        "docs":    "/docs",
        "chunks":  len(chatbot.chunks) if chatbot.chunks else 0,
    }


@app.get("/health")
def health():
    from chatbot_core import EMBED_MODEL, GROQ_MODEL
    return {
        "status":               "ok",
        "data_loaded":          chatbot.data  is not None,
        "index_ready":          chatbot.index is not None,
        "chunks_count":         len(chatbot.chunks) if chatbot.chunks else 0,
        "embed_model":          EMBED_MODEL,
        "llm_model":            GROQ_MODEL,
        "uploaded_files_count": len(chatbot.get_uploaded_files_list()),
        "edupredict_db":        ping_edupredict(),
    }


@app.get("/faculties")
def list_faculties():
    return {"faculties": [
        {"name": f["name"], "programs_count": len(f["programs"])}
        for f in chatbot.data["faculties"].values()
    ]}


@app.get("/faculties/{faculty_name}/programs")
def get_faculty_programs(faculty_name: str):
    for f in chatbot.data["faculties"].values():
        if f["name"] == faculty_name:
            return {"faculty": faculty_name, "programs": f["programs"]}
    raise HTTPException(status_code=404, detail="الكلية غير موجودة")


@app.get("/grants")
def list_grants():
    return chatbot.data.get("grants", {})


@app.get("/enrollment-steps")
def get_enrollment_steps():
    return chatbot.data.get("enrollment_steps", {})


@app.get("/university-info")
def get_university_info():
    return chatbot.data.get("university", {})


@app.get("/chunks")
def list_all_chunks():
    return {"total": len(chatbot.chunks), "chunks": chatbot.chunks}


@app.get("/search")
def test_search(q: str, top_k: int = 5):
    results = chatbot.semantic_search(q, top_k=top_k, threshold=0.0)
    return {"query": q, "results": results}


# ════════════════════════════════════════════════════════════════════════════════
#  UPLOADED FILES ENDPOINTS (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/upload-file")
async def upload_file_from_ui(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="يُقبل فقط ملفات JSON (.json)")

    col_name = (
        collection_name.strip()
        if collection_name and collection_name.strip()
        else file.filename.replace(".json", "").replace(" ", "_")
    )

    try:
        content   = await file.read()
        json_data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"الملف ليس JSON صالحاً: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ في قراءة الملف: {e}")

    try:
        result = chatbot.upload_json_file(col_name, json_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في رفع الملف: {e}")

    chunks_count = len(chatbot._uploaded_chunks.get(col_name, []))

    return {
        "message": "✅ تم رفع الملف وفهرسته بنجاح",
        "details": {
            "collection_name": result["collection"],
            "inserted_docs":   result["inserted"],
            "chunks_count":    chunks_count,
            "filename":        file.filename,
        }
    }


@app.post("/chat/file", response_model=FileChatResponse)
def chat_with_file(req: FileChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً")
    if not req.collection_name.strip():
        raise HTTPException(status_code=400, detail="يجب تحديد اسم الملف (collection_name)")

    try:
        result = chatbot.chat_with_file(
            question        = req.question,
            collection_name = req.collection_name,
            session_id      = req.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"خطأ في المعالجة: {e}")

    return FileChatResponse(
        answer          = result["answer"],
        session_id      = req.session_id,
        collection_name = req.collection_name,
        top_chunks      = result["top_chunks"],
        source          = result["source"],
    )


@app.get("/files/list")
def list_uploaded_files():
    files = chatbot.get_uploaded_files_list()
    return {"total": len(files), "uploaded_files": files}


@app.delete("/files/{collection_name}")
def delete_uploaded_file(collection_name: str):
    success = chatbot.delete_uploaded_file(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"الملف '{collection_name}' غير موجود")
    return {"message": f"✅ تم حذف الملف '{collection_name}' بنجاح"}


@app.post("/files/{collection_name}/reload")
def reload_uploaded_file(collection_name: str):
    success = chatbot.reload_uploaded_file(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"فشل في إعادة تحميل '{collection_name}'")
    return {"message": f"✅ تم إعادة تحميل '{collection_name}' بنجاح"}


@app.get("/files/{collection_name}/chunks")
def get_file_chunks(collection_name: str):
    chunks = chatbot._uploaded_chunks.get(collection_name)
    if chunks is None:
        raise HTTPException(status_code=404, detail=f"الملف '{collection_name}' غير موجود")
    return {
        "collection":   collection_name,
        "chunks_count": len(chunks),
        "chunks":       chunks,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  EDUPREDICT-SPECIFIC ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/student/load-data")
def load_student_data(session_id: str):
    """
    ← جديد: تحميل بيانات الطالب يدوياً من EduPredict.
    مفيد إذا انتهت صلاحية الجلسة وأُعيد تسجيل الدخول.
    """
    if "." not in session_id:
        raise HTTPException(status_code=401, detail="token غير صالح")

    decoded = _verify_token(session_id)
    if decoded is None:
        raise HTTPException(status_code=401, detail="انتهت صلاحية الجلسة")

    student_id = decoded["sid"]
    success    = _load_edupredict_student(student_id)
    col_name   = build_student_collection_name(student_id)

    if not success:
        raise HTTPException(status_code=404, detail="لا تتوفر بيانات لهذا الطالب في EduPredict.")

    chunks_count = len(chatbot._uploaded_chunks.get(col_name, []))
    return {
        "message":       f"✅ تم تحميل بيانات الطالب {student_id}",
        "collection":    col_name,
        "chunks_count":  chunks_count,
    }


@app.get("/student/profile")
def get_student_profile(session_id: str):
    """
    ← جديد: إرجاع ملخص بيانات الطالب من EduPredict مباشرةً (بدون LLM).
    """
    if "." not in session_id:
        raise HTTPException(status_code=401, detail="token غير صالح")

    decoded = _verify_token(session_id)
    if decoded is None:
        raise HTTPException(status_code=401, detail="انتهت صلاحية الجلسة")

    student_id = decoded["sid"]
    profile    = fetch_student_full_profile(student_id)

    if not profile:
        raise HTTPException(status_code=404, detail="لا تتوفر بيانات لهذا الطالب.")

    # إزالة أي بيانات حساسة قبل الإرسال
    safe_profile = {
        "student_info":   profile.get("student_info", {}),
        "enrollments":    profile.get("enrollments", []),
        "assessments":    profile.get("assessments", []),
        "predictions":    profile.get("predictions", []),
        "academic_clock": profile.get("academic_clock", []),
    }
    return safe_profile
