"""
database.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MongoDB connection manager — IUG Chatbot (Multi-Collection)

Collections:
  ┌─────────────────────┬───────────────────────────────────┐
  │ Collection          │ Content                           │
  ├─────────────────────┼───────────────────────────────────┤
  │ university_info     │ Basic university info + summary   │
  │ enrollment_steps    │ Registration steps + portal URL   │
  │ benefits            │ New student benefits list         │
  │ grants              │ All grant tiers + financial aids  │
  │ faculties           │ Faculty metadata (no programs)    │
  │ programs            │ All programs across all faculties │
  │ diploma             │ Educational diploma program       │
  └─────────────────────┴───────────────────────────────────┘
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
MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME     = os.getenv("MONGO_DB_NAME", "iug_chatbot")

# ── Collection names (single source of truth) ───────────────────────────
COL_UNIVERSITY  = "university_info"
COL_ENROLLMENT  = "enrollment_steps"
COL_BENEFITS    = "benefits"
COL_GRANTS      = "grants"
COL_FACULTIES   = "faculties"
COL_PROGRAMS    = "programs"
COL_DIPLOMA     = "diploma"
COL_RANKINGS    = "students_rankings"

ALL_COLLECTIONS = [
    COL_UNIVERSITY,
    COL_ENROLLMENT,
    COL_BENEFITS,
    COL_GRANTS,
    COL_FACULTIES,
    COL_PROGRAMS,
    COL_DIPLOMA,
    COL_RANKINGS,
]

# ── Module-level singletons ──────────────────────────────────────────────
_client: MongoClient = None
_db: Database        = None


def _validate_env() -> None:
    if not MONGO_URI:
        print("❌ [database] MONGO_URI is not set in .env")
        sys.exit(1)


def _get_db() -> Database:
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
    _db = _client[DB_NAME]
    return _db


def get_collection(name: str) -> Collection:
    """Return a specific collection by name."""
    return _get_db()[name]


def get_all_collections() -> dict[str, Collection]:
    """Return a dict of all collections keyed by their constant names."""
    db = _get_db()
    return {name: db[name] for name in ALL_COLLECTIONS}


def ping() -> bool:
    """Verify MongoDB is reachable. Returns True/False."""
    try:
        _get_db().client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        print(f"❌ [database] ping failed: {exc}")
        return False


def close() -> None:
    """Close the MongoClient on app shutdown."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db     = None
        print("🔌 [database] MongoDB connection closed.")