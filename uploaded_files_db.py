"""
uploaded_files_db.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MongoDB connection manager — Uploaded Files Database

قاعدة بيانات منفصلة تماماً عن قاعدة البيانات الأصلية.
كل ملف JSON يُرفع يُخزَّن كـ Collection مستقلة داخل "uploaded_files".

لا يوجد أي تداخل مع قاعدة البيانات الأصلية (iug_chatbot).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
MONGO_URI           = os.getenv("MONGO_URI")
UPLOADED_DB_NAME    = "uploaded_files"   # قاعدة بيانات منفصلة تماماً

# ── Module-level singletons ──────────────────────────────────────────────
_client: MongoClient = None
_db: Database        = None


def _validate_env() -> None:
    if not MONGO_URI:
        print("❌ [uploaded_files_db] MONGO_URI is not set in .env")
        sys.exit(1)


def _get_uploaded_db() -> Database:
    global _client, _db
    if _db is not None:
        return _db
    _validate_env()
    _client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5_000,
        connectTimeoutMS=10_000,
        socketTimeoutMS=30_000,
        retryWrites=True,
        w="majority",
    )
    _db = _client[UPLOADED_DB_NAME]
    return _db


def get_uploaded_collection(collection_name: str) -> Collection:
    """إرجاع collection محددة من قاعدة بيانات الملفات المرفوعة."""
    return _get_uploaded_db()[collection_name]


def list_uploaded_collections() -> list:
    """إرجاع قائمة بأسماء جميع الـ collections (الملفات المرفوعة)."""
    db = _get_uploaded_db()
    return db.list_collection_names()


def drop_uploaded_collection(collection_name: str) -> bool:
    """حذف collection معينة (ملف مرفوع)."""
    db = _get_uploaded_db()
    db.drop_collection(collection_name)
    return True


def ping_uploaded() -> bool:
    """التحقق من الاتصال بقاعدة بيانات الملفات المرفوعة."""
    try:
        _get_uploaded_db().client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        print(f"❌ [uploaded_files_db] ping failed: {exc}")
        return False


def close_uploaded() -> None:
    """إغلاق اتصال MongoDB عند إيقاف التطبيق."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db     = None
        print("🔌 [uploaded_files_db] MongoDB connection closed.")