from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os, hashlib, hmac, base64, json, time

from chatbot_core import IUGChatbot
from database import ping, close as db_close, get_collection, COL_RANKINGS

# ← إضافة: استيراد مدير قاعدة بيانات الملفات المرفوعة
from uploaded_files_db import ping_uploaded, close_uploaded, list_uploaded_collections

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IUG Chatbot API",
    description="الجامعة الإسلامية بغزة — Fast Retrieval + Semantic Search + LLM (v5)",
    version="5.0.0",
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

def _make_token(student_id: str) -> str:
    payload = json.dumps({"sid": student_id, "exp": int(time.time()) + 86400})
    b64     = base64.urlsafe_b64encode(payload.encode()).decode()
    sig     = hmac.new(_SECRET.encode(), b64.encode(), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"

def _verify_token(token: str) -> Optional[str]:
    try:
        b64, sig = token.rsplit(".", 1)
        expected = hmac.new(_SECRET.encode(), b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(b64).decode())
        if payload["exp"] < int(time.time()):
            return None
        return payload["sid"]
    except Exception:
        return None


# ─── Schemas ──────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    student_id: str
    pin: str

class LoginResponse(BaseModel):
    token: str
    student_name: str
    message: str

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    top_chunks: List[str]

class HistoryResponse(BaseModel):
    session_id: str
    history: list

# ← إضافة: Schema لمحادثة الملفات المرفوعة
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

    chatbot.initialize()


# ─── Shutdown ─────────────────────────────────────────────────────────────────
@app.on_event("shutdown")
def shutdown_event():
    db_close()
    close_uploaded()


# ─── Routes الأصلية (لم يتغير أي شيء فيها) ───────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "IUG Chatbot API v5 — Fast Retrieval + Semantic Search + LLM 🎓",
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
    }


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    student = get_collection(COL_RANKINGS).find_one({"_id": req.student_id})
    if not student:
        raise HTTPException(status_code=401, detail="رقم الهوية غير موجود")
    stored_pin = str(student.get("pin", ""))
    if not stored_pin or stored_pin != req.pin.strip():
        raise HTTPException(status_code=401, detail="PIN غير صحيح")
    token = _make_token(req.student_id)
    return LoginResponse(
        token        = token,
        student_name = student.get("student_name", ""),
        message      = f"أهلاً {student.get('student_name', '')} 👋",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً")

    resolved_id = req.session_id
    if req.session_id and "." in req.session_id:
        decoded = _verify_token(req.session_id)
        if decoded is None:
            raise HTTPException(status_code=401, detail="انتهت صلاحية الجلسة — سجّل دخولك مجدداً")
        resolved_id = decoded

    try:
        result = chatbot.chat(req.question, resolved_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM غير متاح: {e}")

    return ChatResponse(
        answer     = result["answer"],
        session_id = req.session_id,
        top_chunks = result["top_chunks"],
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history_route(session_id: str):
    return HistoryResponse(
        session_id = session_id,
        history    = chatbot.get_history(session_id),
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    chatbot.clear_history(session_id)
    return {"message": f"تم مسح محادثة {session_id}"}


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


# ═════════════════════════════════════════════════════════════════════════════
#  Uploaded Files Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/upload-file")
async def upload_file_from_ui(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
):
    """
    ← Endpoint المتوافق مع الواجهة (index.js يرسل إلى /upload-file)

    يقبل ملف JSON ويرفعه إلى قاعدة بيانات uploaded_files.
    يُعيد response بالشكل الذي تتوقعه الواجهة:
      { message, details: { collection_name, chunks_count, inserted_docs } }
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail="يُقبل فقط ملفات JSON (.json)"
        )

    # اسم الـ collection = collection_name إن وُجد، وإلا اسم الملف بدون .json
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
    """
    محادثة مبنية على ملف مرفوع محدد — منفصل تماماً عن /chat العادي.
    """
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
    """قائمة جميع الملفات المرفوعة مع عدد الـ chunks."""
    files = chatbot.get_uploaded_files_list()
    return {
        "total":          len(files),
        "uploaded_files": files,
    }


@app.delete("/files/{collection_name}")
def delete_uploaded_file(collection_name: str):
    """حذف ملف مرفوع من قاعدة البيانات والذاكرة."""
    success = chatbot.delete_uploaded_file(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"الملف '{collection_name}' غير موجود")
    return {"message": f"✅ تم حذف الملف '{collection_name}' بنجاح"}


@app.post("/files/{collection_name}/reload")
def reload_uploaded_file(collection_name: str):
    """إعادة تحميل ملف من MongoDB."""
    success = chatbot.reload_uploaded_file(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"فشل في إعادة تحميل '{collection_name}'")
    return {"message": f"✅ تم إعادة تحميل '{collection_name}' بنجاح"}


@app.get("/files/{collection_name}/chunks")
def get_file_chunks(collection_name: str):
    """عرض الـ chunks لملف مرفوع (للتشخيص)."""
    chunks = chatbot._uploaded_chunks.get(collection_name)
    if chunks is None:
        raise HTTPException(status_code=404, detail=f"الملف '{collection_name}' غير موجود")
    return {
        "collection":   collection_name,
        "chunks_count": len(chunks),
        "chunks":       chunks,
    }