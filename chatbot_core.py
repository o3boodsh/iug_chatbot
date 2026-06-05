import os
import re
import time
from typing import List, Optional

import numpy as np
import requests
from pymongo.errors import PyMongoError

# ─── MongoDB ──────────────────────────────────────────────────────────────────
from database import (
    get_collection,
    COL_UNIVERSITY, COL_ENROLLMENT, COL_BENEFITS,
    COL_GRANTS, COL_FACULTIES, COL_PROGRAMS, COL_DIPLOMA,
    COL_RANKINGS,
)
from postgres_db import get_postgres_student_profile

from uploaded_files_db import (
    get_uploaded_collection,
    list_uploaded_collections,
)

# ─── Config ───────────────────────────────────────────────────────────────────
EMBED_MODEL   = "jina-embeddings-v3"
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
JINA_API_KEY  = os.getenv("JINA_API_KEY", "")
TOP_K         = 5
MAX_HISTORY   = 20
SIM_THRESHOLD = 0.25

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
أنت مساعد جامعي ذكي ومتخصص للجامعة الإسلامية بغزة.

فيما يلي المعلومات ذات الصلة بسؤال الطالب — مستخرجة من قاعدة بيانات الجامعة:
────────────────────────────────────────
{context}
────────────────────────────────────────

تعليمات صارمة يجب الالتزام بها:
1. أجب **فقط** بناءً على المعلومات الواردة أعلاه.
2. لا تخترع أي رقم أو معلومة غير موجودة في النص أعلاه.
3. إذا لم تجد الإجابة بوضوح في النص، أجب بما تعرفه بشكل عام
   واذكر أن المعلومة التفصيلية تحتاج تأكيد من الجامعة مباشرةً.
   لا تقل "لا أعلم" وتوقف — أضف سياقاً مفيداً دائماً.
4. أجب بالعربية فقط في جميع الأحوال.
5. نسّق إجابتك بشكل جميل: استخدم نقاط (•) للقوائم، وأرقاماً
   للخطوات، وأسطراً واضحة. اذكر الأرقام (سعر/معدل/مفتاح)
   بخط منفصل وبارز. الهدف: إجابة يقرأها الطالب بسهولة.
6. تعامل مع الطالب باحترام كأنك موظف في قسم القبول والتسجيل.
7. ⚠️ خصوصية صارمة: بيانات الترتيب والمعدل التراكمي خاصة بكل طالب.
   - إذا سألك الطالب عن معدل أو ترتيب طالب آخر بالاسم أو برقم الهوية → أجب فوراً: "عذراً، هذه البيانات خاصة ولا يمكن الاطلاع عليها."
   - لا تذكر أي معدل أو ترتيب لأي شخص غير الطالب الحالي تحت أي ظرف.
   - حتى لو وُجدت البيانات في السياق أعلاه، لا تُفصح عنها إذا كانت لطالب آخر.
8. إذا كان السؤال عاماً أو يحتمل أكثر من جانب، غطِّ
    أبرز الجوانب باختصار واسأل: "هل تريد تفاصيل عن جانب معين؟"
"""

# ─── System Prompt للملفات المرفوعة (منفصل تماماً) ───────────────────────────
UPLOADED_FILE_SYSTEM_PROMPT = """\
أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على محتوى الملف المُرفق فقط.

فيما يلي محتوى الملف المُرفق الذي يجب أن تُجيب منه حصراً:
────────────────────────────────────────
{context}
────────────────────────────────────────

تعليمات صارمة يجب الالتزام بها:
1. أجب **فقط** بناءً على المعلومات الواردة أعلاه.
2. لا تخترع أي رقم أو معلومة غير موجودة في النص أعلاه.
3. إذا لم تجد الإجابة بوضوح في النص، أجب بما تعرفه بشكل عام
   واذكر أن المعلومة التفصيلية تحتاج تأكيد من الجامعة مباشرةً.
   لا تقل "لا أعلم" وتوقف — أضف سياقاً مفيداً دائماً.
4. أجب بالعربية فقط في جميع الأحوال.
5. نسّق إجابتك بشكل جميل: استخدم نقاط (•) للقوائم، وأرقاماً
   للخطوات، وأسطراً واضحة. الهدف: إجابة يقرأها الطالب بسهولة.
6. تعامل مع الطالب باحترام كأنك موظف .
7. ⚠️ خصوصية صارمة: بيانات الترتيب والمعدل التراكمي خاصة بكل طالب.
   - إذا سألك الطالب عن معدل أو ترتيب طالب آخر بالاسم أو برقم الهوية → أجب فوراً: "عذراً، هذه البيانات خاصة ولا يمكن الاطلاع عليها."
   - لا تذكر أي معدل أو ترتيب لأي شخص غير الطالب الحالي تحت أي ظرف.
   - حتى لو وُجدت البيانات في السياق أعلاه، لا تُفصح عنها إذا كانت لطالب آخر.
8. إذا كان السؤال عاماً أو يحتمل أكثر من جانب، غطِّ
    أبرز الجوانب باختصار واسأل: "هل تريد تفاصيل عن جانب معين؟"
"""


class IUGChatbot:
    """
    Core engine for IUG Chatbot.
    Encapsulates all business logic: data loading, chunk building,
    semantic indexing, fast retrieval, session history, and LLM calls.
    """

    # ── كلمات مفتاحية للتصنيف السريع ─────────────────────────────────────────
    _KW_ENROLLMENT = [
        "كيف اسجل", "كيف أسجل", "طريقة التسجيل", "خطوات التسجيل",
        "خطوات الالتحاق", "كيفية الالتحاق", "الالتحاق بالجامعة",
        "الانتساب", "التسجيل في الجامعة", "رقم جامعي", "الرقم الجامعي",
        "بوابة التسجيل", "كيف التحق", "تسجيل المساقات",
        "خطوات", "التحاق",
    ]

    _KW_GRANTS = ["منحة", "منح", "منحه"]

    _KW_BENEFITS = [
        "مزايا طلبة جدد", "مزايا الطلبة", "تسهيلات الطلبة الجدد",
        "الاعفاءات", "إعفاءات", "الفصل الأول مجاني",
        "مجاني للطلبة الجدد", "مزايا ملتحق", "ما يحصل عليه الطالب الجديد",
    ]

    _KW_UNIV_INFO = [
        "موقع الجامعة", "ايميل الجامعة", "بريد الجامعة",
        "هاتف الجامعة", "رقم الجامعة", "واتساب الجامعة",
        "تواصل مع الجامعة", "بوابة الطلبة الجدد",
        "عن الجامعة", "تعريف الجامعة", "متى تاسست",
        "كم برنامج", "عنوان الجامعة", "اتواصل",
    ]

    # ═════════════════════════════════════════════════════════════════════════
    #  INIT / STARTUP
    # ═════════════════════════════════════════════════════════════════════════

    def __init__(self):
        self._data: dict = None
        self._chunks: List[str] = None
        self._index: np.ndarray = None
        self._sessions: dict = {}
        self._alias_index: list = None  # built lazily once

        self._uploaded_chunks: dict = {}   # {collection_name: [chunks]}

    def initialize(self):
        """Load data, build chunks, load embedder, build semantic index."""
        print("⏳ Loading university data …")
        self._data = self._load_data()
        self._chunks = self._build_chunks(self._data)
        print(f"✅ Built {len(self._chunks)} chunks.")

        print(f"⏳ Using Jina Embeddings API — model: '{EMBED_MODEL}' …")
        if not JINA_API_KEY:
            raise RuntimeError("❌ JINA_API_KEY غير موجود — أضفه في ملف .env")

        print("⏳ Building semantic index …")
        self._index = self._build_index(self._chunks)
        print(f"✅ Semantic index ready — shape: {self._index.shape}")

        self._load_all_uploaded_files()

    # ═════════════════════════════════════════════════════════════════════════
    #  PUBLIC PROPERTIES (read-only access for routes)
    # ═════════════════════════════════════════════════════════════════════════

    @property
    def data(self) -> dict:
        return self._data

    @property
    def chunks(self) -> List[str]:
        return self._chunks

    @property
    def index(self) -> np.ndarray:
        return self._index

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — DATA HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _load_data() -> dict:
        """
        Load university data from MongoDB — multi-collection schema.

        Queries 7 collections and reassembles them into the same dict
        structure the rest of the system expects (identical to the
        original university_data.json shape). Nothing above this method
        needs to change.

        Collections → JSON keys mapping:
          university_info   → data["university"] + data["programs_summary"]
          enrollment_steps  → data["enrollment_steps"]
          benefits          → data["new_student_benefits"]
          grants            → data["grants"]
          faculties         → data["faculties"] (metadata)
          programs          → data["faculties"][key]["programs"]  (merged in)
          diploma           → data["educational_diploma"]
        """
        try:
            # ── 1. university_info ────────────────────────────────
            u_doc = get_collection(COL_UNIVERSITY).find_one({"_id": "university_info"})
            if not u_doc:
                raise RuntimeError(
                    "❌ 'university_info' document missing in MongoDB.\n"
                    "Run: python seed_database.py"
                )
            university = {
                k: u_doc[k]
                for k in (
                    "name", "name_en", "short_name", "aliases",
                    "website", "email", "phone", "whatsapp_support",
                    "new_students_portal", "academic_year",
                    "location", "description",
                )
                if k in u_doc
            }
            programs_summary = u_doc.get("programs_summary", {})

            # ── 2. enrollment_steps ───────────────────────────────
            e_doc = get_collection(COL_ENROLLMENT).find_one({"_id": "enrollment"})
            enrollment_steps = {
                "title":      e_doc.get("title", "") if e_doc else "",
                "portal_url": e_doc.get("portal_url", "") if e_doc else "",
                "steps":      e_doc.get("steps", []) if e_doc else [],
            }

            # ── 3. benefits ───────────────────────────────────────
            b_doc = get_collection(COL_BENEFITS).find_one({"_id": "new_student_benefits"})
            new_student_benefits = {
                "title":    b_doc.get("title", "") if b_doc else "",
                "benefits": b_doc.get("benefits", []) if b_doc else [],
            }

            # ── 4. grants ─────────────────────────────────────────
            # Rebuild the original nested grants dict keyed by tier_key
            grants: dict = {}
            for g_doc in get_collection(COL_GRANTS).find({}):
                tier_key = g_doc.pop("_id")
                g_doc.pop("seeded_at", None)
                grants[tier_key] = g_doc

            # ── 5. faculties + programs (merged) ──────────────────
            # Load all programs once and group by faculty_key
            programs_by_faculty: dict = {}
            for p_doc in get_collection(COL_PROGRAMS).find({}):
                fac_key = p_doc.pop("faculty_key")
                p_doc.pop("faculty_name", None)
                p_doc.pop("seeded_at",   None)
                # Restore original _id → id mapping
                orig_id = p_doc.pop("_id").split("_", 1)[-1]
                p_doc["id"] = orig_id
                programs_by_faculty.setdefault(fac_key, []).append(p_doc)

            # Load faculty metadata and attach programs
            faculties: dict = {}
            for f_doc in get_collection(COL_FACULTIES).find({}):
                fac_key = f_doc.pop("_id")
                f_doc.pop("programs_count", None)
                f_doc.pop("seeded_at",     None)
                f_doc["programs"] = programs_by_faculty.get(fac_key, [])
                faculties[fac_key] = f_doc

            # ── 6. diploma ────────────────────────────────────────
            d_doc = get_collection(COL_DIPLOMA).find_one({"_id": "educational_diploma"})
            educational_diploma = {}
            if d_doc:
                d_doc.pop("seeded_at", None)
                d_doc.pop("_id",       None)
                educational_diploma = d_doc

            # ── 7. students_rankings ──────────────────────────────
            rankings_docs = list(get_collection(COL_RANKINGS).find({}))
            for r in rankings_docs:
                r.pop("seeded_at", None)
            students_rankings = rankings_docs

        except PyMongoError as exc:
            raise RuntimeError(
                f"❌ MongoDB query failed: {exc}\n"
                "Check MONGO_URI in .env and Atlas network access."
            ) from exc

        # ── Reassemble into original JSON shape ───────────────────
        data = {
            "university":            university,
            "programs_summary":      programs_summary,
            "enrollment_steps":      enrollment_steps,
            "new_student_benefits":  new_student_benefits,
            "grants":                grants,
            "educational_diploma":   educational_diploma,
            "faculties":             faculties,
            "students_rankings":     students_rankings,
        }

        print(
            f"✅ Data loaded from MongoDB — "
            f"{len(faculties)} faculties, "
            f"{sum(len(v) for v in programs_by_faculty.values())} programs, "
            f"{len(students_rankings)} ranked students"
        )
        return data

    @staticmethod
    def _fmt_admission(r) -> str:
        if not r:
            return "غير محدد"
        if isinstance(r, str):
            return r
        if "general" in r:
            return r["general"]
        parts = []
        if "scientific" in r: parts.append(f"علمي {r['scientific']}")
        if "literary"   in r: parts.append(f"أدبي {r['literary']}")
        if "male"       in r: parts.append(f"طلاب {r['male']}")
        if "female"     in r: parts.append(f"طالبات {r['female']}")
        note   = r.get("note", "")
        result = " / ".join(parts)
        return f"{result} ({note})" if note else result

    @staticmethod
    def _fmt_key(k) -> str:
        if not k:
            return "غير متوفر"
        if isinstance(k, (int, float)):
            return str(int(k))
        if isinstance(k, str):
            return k
        if isinstance(k, dict):
            male   = k.get("male", "")
            female = k.get("female", "")
            note   = k.get("note", "")
            if male and female:
                base = f"{male} للطلاب / {female} للطالبات"
                return f"{base} — {note}" if note else base
            return note or "غير متوفر"
        return "غير متوفر"

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — CHUNK BUILDER
    # ═════════════════════════════════════════════════════════════════════════

    def _build_chunks(self, data: dict) -> List[str]:
        chunks: List[str] = []

        u = data.get("university", {})
        chunks.append(
            f"معلومات الجامعة الإسلامية بغزة:\n"
            f"الاسم: {u.get('name')} ({u.get('name_en')})\n"
            f"{u.get('description', '')}\n"
            f"الموقع: {u.get('website')} | البريد: {u.get('email')}\n"
            f"الهاتف: {u.get('phone')} | واتساب: {u.get('whatsapp_support')}\n"
            f"بوابة الطلبة الجدد: {u.get('new_students_portal')}\n"
            f"العام الدراسي: {u.get('academic_year')}"
        )

        enroll     = data.get("enrollment_steps", {})
        steps_text = "\n".join(
            f"  الخطوة {s['step']}: {s['title']} — {s['description']}"
            for s in enroll.get("steps", [])
        )
        chunks.append(
            f"{enroll.get('title', 'خطوات التسجيل والالتحاق بالجامعة')}:\n"
            f"{steps_text}\n"
            f"رابط البوابة: {enroll.get('portal_url', '')}"
        )

        benefits = data.get("new_student_benefits", {})
        b_text   = "\n".join(f"  • {b}" for b in benefits.get("benefits", []))
        chunks.append(f"{benefits.get('title', 'مزايا الطلبة الجدد')}:\n{b_text}")

        for tier_key, tier in data.get("grants", {}).items():
            if tier_key == "other_financial_facilities":
                items = "\n".join(
                    f"  • {i['name']}: {i.get('description', '')}"
                    for i in tier.get("items", [])
                )
                chunks.append(f"تسهيلات مالية أخرى:\n{items}")
            elif isinstance(tier, dict) and "grants" in tier:
                pct  = tier.get("percentage", "")
                desc = tier.get("description", "")
                g_lines = []
                for g in tier["grants"]:
                    name    = g["name"]                if isinstance(g, dict) else g
                    note    = g.get("note", "")         if isinstance(g, dict) else ""
                    aliases = ", ".join(g.get("aliases", [])) if isinstance(g, dict) else ""
                    line    = f"  • {name}"
                    if note:    line += f" ({note})"
                    if aliases: line += f" [يُعرف أيضاً بـ: {aliases}]"
                    g_lines.append(line)
                chunks.append(
                    f"منح دراسية بنسبة {pct} ({desc}):\n" + "\n".join(g_lines)
                )

        for fac in data.get("faculties", {}).values():
            fname      = fac["name"]
            aliases    = ", ".join(fac.get("aliases", []))
            tags       = ", ".join(fac.get("tags", []))
            desc       = fac.get("description", "")
            free_note  = "الفصل الأول مجاني للطلبة الجدد." if fac.get("first_semester_free") else ""
            inst_note  = "تقسيط الرسوم متاح." if fac.get("installment_available") else ""
            prog_names = "، ".join(p["name"] for p in fac.get("programs", []))

            chunks.append(
                f"كلية {fname} (مرادفات: {aliases}):\n"
                f"{desc}\n"
                f"الكلمات المفتاحية: {tags}\n"
                f"{free_note} {inst_note}\n"
                f"البرامج المتاحة في الكلية: {prog_names}"
            )

        for fac in data.get("faculties", {}).values():
            fname = fac["name"]
            for prog in fac.get("programs", []):
                pname   = prog["name"]
                aliases = ", ".join(prog.get("aliases", []))
                tags    = ", ".join(prog.get("tags", []))
                price   = prog.get("credit_hour_price", "؟")
                rate    = self._fmt_admission(prog.get("admission_rate"))
                key     = self._fmt_key(prog.get("coordination_key"))
                gender  = prog.get("gender_restriction", "")
                dur     = prog.get("duration_years", "")
                specs   = "، ".join(prog.get("specializations", []))
                g_list  = "\n".join(f"    - {g}" for g in prog.get("grants", []))

                lines = [
                    f"برنامج: {pname}",
                    f"الكلية: كلية {fname}",
                    f"أسماء أخرى / مرادفات: {aliases}" if aliases else None,
                    f"الكلمات المفتاحية: {tags}"        if tags    else None,
                    f"سعر الساعة المعتمدة: {price} دينار",
                    f"معدل القبول: {rate}",
                    f"مفتاح التنسيق: {key}",
                    f"القيد: {gender}"                  if gender  else None,
                    f"مدة الدراسة: {dur} سنوات"         if dur     else None,
                    f"تخصصات فرعية: {specs}"            if specs   else None,
                    f"منح خاصة بهذا البرنامج:\n{g_list}" if g_list else None,
                ]
                chunks.append("\n".join(l for l in lines if l))

        d = data.get("educational_diploma", {})
        if d:
            chunks.append(
                f"دبلوم التأهيل التربوي:\n"
                f"{d.get('description', '')}\n"
                f"مفتاح التنسيق: {d.get('coordination_key')}"
            )

        # ── Rankings (privacy-aware) ───────────────────────────────
        for student in data.get("students_rankings", []):
            sid      = student.get("student_id", "")
            name     = student.get("student_name", "")
            gpa      = student.get("gpa", "")
            rank     = student.get("rank", "")
            allowed  = student.get("privacy", {}).get("allowed_users", [])
            chunks.append(
                f"[RANKING|student_id={sid}|allowed={','.join(allowed)}]\n"
                f"الطالب: {name}\n"
                f"رقم الهوية: {sid}\n"
                f"المعدل التراكمي: {gpa}\n"
                f"الترتيب: {rank}"
            )

        return chunks

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION 3 — SEMANTIC INDEX
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _jina_embed(texts: List[str]) -> np.ndarray:
        """استدعاء Jina Embeddings API وإرجاع مصفوفة numpy."""
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        data = {
            "model": EMBED_MODEL,
            "input": texts,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        embeddings = [item["embedding"] for item in resp.json()["data"]]
        return np.array(embeddings, dtype=np.float32)

    @staticmethod
    def _call_groq(headers: dict, payload: dict, max_retries: int = 4) -> str:
        """
        استدعاء Groq API مع Exponential Backoff للتعامل مع خطأ 429 (Rate Limit).

        الانتظار: 2s → 4s → 8s → 16s (قابل للتمديد عبر max_retries).
        يُعيد نص الإجابة مباشرة عند النجاح، ويرفع RuntimeError عند الفشل.
        """
        url = "https://api.groq.com/openai/v1/chat/completions"

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)

                # ── 429: Rate Limit → انتظر وأعد المحاولة ──────────────────
                if resp.status_code == 429:
                    # Groq يُرسل أحياناً Retry-After (بالثواني)
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else (2 ** attempt)
                    print(
                        f"⚠️  Groq 429 — المحاولة {attempt}/{max_retries}، "
                        f"انتظار {wait:.1f}s …"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    "❌ تعذّر الاتصال بـ Groq API — تحقق من الاتصال بالإنترنت."
                )
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"⏱️  Groq Timeout — المحاولة {attempt}/{max_retries}، انتظار {wait}s …")
                    time.sleep(wait)
                    continue
                raise RuntimeError("❌ Groq API استغرق وقتاً طويلاً — حاول مرة أخرى.")
            except Exception as exc:
                raise RuntimeError(f"❌ خطأ في Groq: {exc}")

        raise RuntimeError(
            "❌ Groq API: تجاوزنا الحد المسموح به من الطلبات (429). "
            "حاول بعد لحظات أو تحقق من خطة Groq الخاصة بك."
        )

    @staticmethod
    def _build_index(chunks: List[str]) -> np.ndarray:
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"   Embedding batch {i // batch_size + 1} ({len(batch)} chunks) …")
            embeddings = IUGChatbot._jina_embed(batch)
            all_embeddings.append(embeddings)
        result = np.vstack(all_embeddings) if all_embeddings else np.array([], dtype=np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return result / norms

    def semantic_search(
        self,
        question: str,
        top_k: int = TOP_K,
        threshold: float = SIM_THRESHOLD,
    ) -> List[str]:
        q_arr = self._jina_embed([question])
        # normalize
        norm  = np.linalg.norm(q_arr)
        q_vec = (q_arr / norm if norm != 0 else q_arr).T

        scores = (self._index @ q_vec).flatten()
        ranked = np.argsort(scores)[::-1]

        results = []
        for idx in ranked[:top_k]:
            if float(scores[idx]) >= threshold:
                results.append(self._chunks[int(idx)])

        if not results:
            results.append(self._chunks[int(ranked[0])])

        return results

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION 4 — UPLOADED FILES (منفصل تماماً عن النظام الأصلي)
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _flatten_json_to_text(obj, prefix: str = "") -> List[str]:
        """
        تحويل JSON (بأي هيكل) إلى قائمة نصوص قابلة للفهرسة.
        يعمل مع: dict, list, قيم بسيطة — بشكل تعاودي.
        """
        lines = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.extend(IUGChatbot._flatten_json_to_text(value, full_key))
                else:
                    lines.append(f"{full_key}: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                full_key = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    lines.extend(IUGChatbot._flatten_json_to_text(item, full_key))
                else:
                    lines.append(f"{prefix}: {item}")
        else:
            lines.append(f"{prefix}: {obj}")
        return lines

    @staticmethod
    def _build_uploaded_chunks(docs: List[dict], collection_name: str) -> List[str]:
        """
        تحويل documents من MongoDB إلى chunks نصية.
        كل document يُحوَّل إلى chunk واحد أو أكثر.
        """
        chunks = []
        for doc in docs:
            doc.pop("_id", None)  # حذف _id من MongoDB — لا يفيد في البحث
            doc.pop("__file_meta__", None)  # حذف الـ metadata الداخلية

            # تسطيح الـ JSON إلى سطور نصية
            flat_lines = IUGChatbot._flatten_json_to_text(doc)
            if flat_lines:
                chunk_text = f"[ملف: {collection_name}]\n" + "\n".join(flat_lines)
                chunks.append(chunk_text)

        return chunks

    def _load_all_uploaded_files(self):
        """
        تحميل جميع الملفات المرفوعة من MongoDB عند بدء التشغيل.
        """
        try:
            collections = list_uploaded_collections()
            if not collections:
                print("ℹ️  No uploaded files found in MongoDB.")
                return

            for col_name in collections:
                self._load_uploaded_collection(col_name)

            print(f"✅ Loaded {len(self._uploaded_chunks)} uploaded file(s).")
        except Exception as exc:
            print(f"⚠️  Could not load uploaded files: {exc}")

    def _load_uploaded_collection(self, collection_name: str):
        """
        بناء chunks لـ collection واحدة من الملفات المرفوعة.
        يُستدعى عند التحميل الأولي وعند رفع ملف جديد أو إعادة التحميل.
        """
        col = get_uploaded_collection(collection_name)
        docs = list(col.find({}))

        if not docs:
            return

        chunks = self._build_uploaded_chunks(docs, collection_name)
        if not chunks:
            return

        self._uploaded_chunks[collection_name] = chunks
        print(f"   ✅ Loaded uploaded file '{collection_name}' ({len(chunks)} chunks).")

    def upload_json_file(self, collection_name: str, json_data: list) -> dict:
        """
        رفع ملف JSON إلى MongoDB (قاعدة بيانات uploaded_files) وتحضيره للمحادثة.
        
        - collection_name: اسم الـ collection (اسم الملف)
        - json_data: محتوى الملف كـ list of dicts أو dict واحد
        
        يُعيد: {"inserted": int, "collection": str}
        """
        col = get_uploaded_collection(collection_name)

        # إذا كان المحتوى dict واحد → حوّله إلى list
        if isinstance(json_data, dict):
            json_data = [json_data]

        if not isinstance(json_data, list):
            raise ValueError("محتوى الملف يجب أن يكون JSON object أو array.")

        # تنظيف البيانات من أي _id موجود لتفادي تضارب MongoDB
        cleaned = []
        for item in json_data:
            if isinstance(item, dict):
                item.pop("_id", None)
                cleaned.append(item)
            else:
                # إذا كان العنصر قيمة بسيطة → ضعها في dict
                cleaned.append({"value": item})

        # حذف الـ collection القديمة إن وُجدت (استبدال كامل عند إعادة الرفع)
        col.drop()

        # إدراج البيانات الجديدة
        if cleaned:
            col.insert_many(cleaned)

        self._load_uploaded_collection(collection_name)

        return {"inserted": len(cleaned), "collection": collection_name}

    def chat_with_file(
        self,
        question: str,
        collection_name: str,
        session_id: str,
    ) -> dict:
        """
        محادثة مبنية على ملف مرفوع محدد — لا تخلط مع البيانات الأصلية إطلاقاً.
        
        يُستدعى من endpoint /chat/file بدلاً من /chat العادي.
        """
        if collection_name not in self._uploaded_chunks:
            return {
                "answer":     f"الملف '{collection_name}' غير موجود. يرجى رفع الملف أولاً.",
                "top_chunks": [],
                "source":     "uploaded_file",
            }

        # استخدم كل بيانات الملف المرفوع كسياق، بدون حصرها بأفضل TOP_K نتائج.
        relevant_chunks = self._uploaded_chunks[collection_name]

        if not relevant_chunks:
            return {
                "answer":     "لا تتوفر هذه المعلومة في الملف المُرفق.",
                "top_chunks": [],
                "source":     "uploaded_file",
            }

        context = "\n\n---\n\n".join(relevant_chunks)

        # بناء الـ prompt — يستخدم UPLOADED_FILE_SYSTEM_PROMPT فقط
        history      = self.get_history(session_id)
        history_text = self.fmt_history(history)
        system       = UPLOADED_FILE_SYSTEM_PROMPT.format(context=context)
        user_message = f"{history_text}السؤال: {question}"

        if not GROQ_API_KEY:
            raise RuntimeError("❌ GROQ_API_KEY غير موجود — أضفه في ملف .env")

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.05,
        }

        answer = self._call_groq(headers, payload)
        self.push_history(session_id, question, answer)

        return {
            "answer":     answer,
            "top_chunks": relevant_chunks,
            "source":     "uploaded_file",
        }

    def get_uploaded_files_list(self) -> List[dict]:
        """إرجاع قائمة الملفات المرفوعة من MongoDB مباشرةً (مصدر الحقيقة)."""
        from uploaded_files_db import list_uploaded_collections, get_uploaded_collection
        try:
            collections = list_uploaded_collections()
            result = []
            for col_name in collections:
                chunks_in_memory = self._uploaded_chunks.get(col_name, [])
                count = len(chunks_in_memory) if chunks_in_memory else get_uploaded_collection(col_name).count_documents({})
                result.append({
                    "collection": col_name,
                    "chunks_count": count,
                })
            return result
        except Exception:
            # fallback للـ memory لو MongoDB تعطّل
            return [
                {"collection": name, "chunks_count": len(chunks)}
                for name, chunks in self._uploaded_chunks.items()
            ]

    def reload_uploaded_file(self, collection_name: str) -> bool:
        """إعادة تحميل ملف مرفوع من MongoDB (مفيد بعد تعديل البيانات يدوياً)."""
        try:
            self._load_uploaded_collection(collection_name)
            return True
        except Exception:
            return False

    def delete_uploaded_file(self, collection_name: str) -> bool:
        """حذف ملف مرفوع من قاعدة البيانات والذاكرة."""
        from uploaded_files_db import drop_uploaded_collection
        drop_uploaded_collection(collection_name)
        self._uploaded_chunks.pop(collection_name, None)
        print(f"🗑️  Deleted uploaded file '{collection_name}' from DB and memory.")
        return True

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION 5 — SESSION HISTORY
    # ═════════════════════════════════════════════════════════════════════════

    def get_history(self, sid: str) -> list:
        return self._sessions.setdefault(sid, [])

    def push_history(self, sid: str, user: str, assistant: str):
        h = self.get_history(sid)
        h.append({"user": user, "assistant": assistant})
        if len(h) > MAX_HISTORY:
            self._sessions[sid] = h[-MAX_HISTORY:]

    def clear_history(self, sid: str):
        self._sessions.pop(sid, None)

    @staticmethod
    def fmt_history(history: list) -> str:
        if not history:
            return ""
        turns = "\n".join(
            f"الطالب: {t['user']}\nالمساعد: {t['assistant']}"
            for t in history[-6:]
        )
        return f"سجل المحادثة السابقة:\n{turns}\n\n"

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION 6 — FAST RETRIEVAL LAYER
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_fr(text: str) -> str:
        """تطبيع بسيط: أحرف صغيرة + توحيد الألف + إزالة التشكيل."""
        text = text.lower().strip()
        text = re.sub(r"[أإآ]", "ا", text)
        text = re.sub(r"[\u064B-\u065F]", "", text)
        return text

    @classmethod
    def _contains_any(cls, text: str, keywords: list) -> bool:
        """هل النص يحتوي على أي من الكلمات المفتاحية بعد التطبيع؟"""
        t = cls._normalize_fr(text)
        return any(cls._normalize_fr(kw) in t for kw in keywords)

    @staticmethod
    def _fmt_admission_fr(rate) -> str:
        """تحويل admission_rate بأي شكل إلى نص عربي."""
        if not rate:
            return "غير محدد"
        if isinstance(rate, str):
            return rate
        if "general" in rate:
            return rate["general"]
        parts = []
        if "scientific" in rate: parts.append(f"علمي {rate['scientific']}")
        if "literary"   in rate: parts.append(f"أدبي {rate['literary']}")
        if "male"       in rate: parts.append(f"ذكور {rate['male']}")
        if "female"     in rate: parts.append(f"إناث {rate['female']}")
        note   = rate.get("note", "")
        result = " / ".join(parts)
        return f"{result} ({note})" if note else result

    @staticmethod
    def _fmt_key_fr(key) -> str:
        """تحويل coordination_key بأي شكل إلى نص عربي."""
        if not key:
            return "غير متوفر"
        if isinstance(key, (int, float)):
            return str(int(key))
        if isinstance(key, str):
            return key
        if isinstance(key, dict):
            male   = key.get("male", "")
            female = key.get("female", "")
            note   = key.get("note", "")
            if male and female:
                base = f"{male} للطلاب / {female} للطالبات"
                return f"{base} — {note}" if note else base
            return note or "غير متوفر"
        return "غير متوفر"

    def _build_alias_index(self, data: dict) -> list:
        """
        يبني قائمة (pattern_normalized, entry_dict) لكل برنامج وكلية مع aliases.
        يُرتَّب تنازلياً بطول الـ pattern → الأطول يُفحص أولاً لتفادي التضارب.
        """
        index = []

        for fac in data.get("faculties", {}).values():
            fname = fac["name"]

            fac_entry = {
                "type":     "faculty",
                "faculty":  fname,
                "programs": fac.get("programs", []),
            }
            for alias in [fname] + fac.get("aliases", []):
                index.append((self._normalize_fr(alias), fac_entry))

            for prog in fac.get("programs", []):
                prog_entry = {
                    "type":    "program",
                    "name":    prog["name"],
                    "faculty": fname,
                    "data":    prog,
                }
                for alias in [prog["name"]] + prog.get("aliases", []):
                    index.append((self._normalize_fr(alias), prog_entry))

        index.sort(key=lambda x: len(x[0]), reverse=True)
        return index

    def _search_alias_index(self, question: str) -> Optional[dict]:
        """يبحث عن أول تطابق في فهرس الأسماء ويُعيد الـ entry أو None."""
        if self._alias_index is None:
            self._alias_index = self._build_alias_index(self._data)

        q = self._normalize_fr(question)
        for pattern, entry in self._alias_index:
            if pattern in q:
                return entry
        return None

    def _fr_handle_enrollment(self, data: dict) -> str:
        enroll = data.get("enrollment_steps", {})
        steps  = enroll.get("steps", [])
        if not steps:
            return "للاستفسار عن خطوات التسجيل يرجى التواصل مع الجامعة مباشرة."
        lines = [f"📋 {enroll.get('title', 'خطوات الالتحاق بالجامعة الإسلامية بغزة')}:\n"]
        for s in steps:
            lines.append(f"🔸 الخطوة {s['step']}: {s['title']}")
            lines.append(f"   {s['description']}\n")
        portal = enroll.get("portal_url", "")
        if portal:
            lines.append(f"🌐 رابط البوابة: {portal}")
        return "\n".join(lines)

    def _fr_handle_grants(self, question: str, data: dict) -> str:
        q           = self._normalize_fr(question)
        grants_data = data.get("grants", {})

        tier_map = {
            "hundred_percent":  "100%",
            "35_to_70_percent": "35% - 70%",
            "15_to_30_percent": "15% - 30%",
            "25_to_35_percent": "25% - 35%",
        }

        for tier_key, pct in tier_map.items():
            tier = grants_data.get(tier_key, {})
            for g in tier.get("grants", []):
                names = ([g["name"]] + g.get("aliases", [])) if isinstance(g, dict) else [g]
                for gname in names:
                    if self._normalize_fr(gname) in q:
                        note   = g.get("note", "") if isinstance(g, dict) else ""
                        result = f"منحة «{g['name'] if isinstance(g, dict) else g}» نسبتها {pct}"
                        if note:
                            result += f"\nملاحظة: {note}"
                        return result

        if self._contains_any(question, ["كل", "جميع", "قائمة", "اذكر", "ما هي", "انواع", "اريد اعرف"]):
            lines = ["🎓 المنح الدراسية المتاحة في الجامعة الإسلامية بغزة:\n"]
            for tier_key, pct in tier_map.items():
                tier        = grants_data.get(tier_key, {})
                grants_list = tier.get("grants", [])
                if grants_list:
                    lines.append(f"🔹 منح بنسبة {pct}:")
                    for g in grants_list:
                        gname = g["name"] if isinstance(g, dict) else g
                        note  = g.get("note", "") if isinstance(g, dict) else ""
                        line  = f"   • {gname}"
                        if note: line += f" ({note})"
                        lines.append(line)
                    lines.append("")
            other = grants_data.get("other_financial_facilities", {})
            items = other.get("items", [])
            if items:
                lines.append("🔹 تسهيلات مالية إضافية:")
                for item in items:
                    lines.append(f"   • {item['name']}: {item.get('description', '')}")
            return "\n".join(lines)

        return (
            "تتوفر في الجامعة الإسلامية بغزة منح دراسية متعددة:\n"
            "  • منح بنسبة 100% (إعفاء كامل)\n"
            "  • منح بنسبة 35% - 70%\n"
            "  • منح بنسبة 25% - 35%\n"
            "  • منح بنسبة 15% - 30%\n"
            "يمكنني تزويدك بتفاصيل أي منحة محددة إذا ذكرت اسمها."
        )

    def _fr_handle_benefits(self, data: dict) -> Optional[str]:
        bd       = data.get("new_student_benefits", {})
        benefits = bd.get("benefits", [])
        if not benefits:
            return None
        lines = [f"🎁 {bd.get('title', 'مزايا الطلبة الجدد')}:\n"]
        for b in benefits:
            lines.append(f"  ✅ {b}")
        return "\n".join(lines)

    def _fr_handle_univ_info(self, question: str, data: dict) -> Optional[str]:
        u = data.get("university", {})
        if self._contains_any(question, ["موقع", "ويب", "site", "web", "رابط", "لينك"]):
            return f"🌐 الموقع الرسمي للجامعة: {u.get('website')}"
        if self._contains_any(question, ["ايميل", "بريد", "email", "mail"]):
            return f"📧 البريد الإلكتروني: {u.get('email')}"
        if self._contains_any(question, ["واتساب", "whatsapp", "واتس"]):
            return f"💬 واتساب الدعم: {u.get('whatsapp_support')}"
        if self._contains_any(question, ["هاتف", "تلفون", "رقم", "phone", "اتصال", "جوال"]):
            return f"📞 هاتف الجامعة: {u.get('phone')}"
        if self._contains_any(question, ["بوابة", "portal", "newstd"]):
            return f"🔗 بوابة الطلبة الجدد: {u.get('new_students_portal')}"
        if self._contains_any(question, ["عن الجامعة", "تعريف", "تاسست", "كم برنامج", "عنوان"]):
            return (
                f"🏛️  الجامعة الإسلامية بغزة (IUG)\n"
                f"{u.get('description', '')}\n"
                f"🌐 {u.get('website')}  |  📧 {u.get('email')}  |  📞 {u.get('phone')}"
            )
        return None

    def _fr_handle_program(self, question: str, prog: dict, faculty_name: str) -> str:
        name  = prog["name"]
        price = prog.get("credit_hour_price", "غير متوفر")
        rate  = self._fmt_admission_fr(prog.get("admission_rate"))
        key   = self._fmt_key_fr(prog.get("coordination_key"))

        if self._contains_any(question, ["سعر", "ساعة", "رسوم", "تكلفة", "تكلفه", "ساعه", "كم تكلف", "كم الرسوم"]):
            return f"💰 سعر الساعة المعتمدة في {name}: {price} دينار"

        if self._contains_any(question, ["معدل", "قبول", "نسبة", "نسبه", "مقبول", "يقبل"]):
            return f"📊 معدل القبول في {name}: {rate}"

        if self._contains_any(question, ["مفتاح", "تنسيق"]):
            return f"🔑 مفتاح التنسيق لـ{name}: {key}"

        if self._contains_any(question, ["منحة", "منح"]):
            prog_grants = prog.get("grants", [])
            if prog_grants:
                lines = [f"🎓 المنح الخاصة بـ{name}:"]
                for g in prog_grants:
                    lines.append(f"  • {g}")
                return "\n".join(lines)
            return f"لا توجد منح خاصة مُدرجة لـ{name}، يمكنك الاستفسار عن المنح العامة للجامعة."

        lines = [
            f"📚 برنامج: {name}",
            f"🏛️  الكلية: {faculty_name}",
            f"💰 سعر الساعة المعتمدة: {price} دينار",
            f"📊 معدل القبول: {rate}",
            f"🔑 مفتاح التنسيق: {key}",
        ]
        gender = prog.get("gender_restriction", "")
        if gender:
            lines.append(f"⚠️  القيد: {gender}")

        specs = prog.get("specializations", [])
        if specs:
            lines.append(f"📌 تخصصات فرعية: {', '.join(specs)}")

        prog_grants = prog.get("grants", [])
        if prog_grants:
            lines.append("🎓 منح خاصة:")
            for g in prog_grants:
                lines.append(f"   • {g}")

        return "\n".join(lines)

    def _fr_handle_faculty(self, question: str, fac_entry: dict) -> str:
        fname    = fac_entry["faculty"]
        programs = fac_entry["programs"]

        if self._contains_any(question, ["سعر", "ساعة", "رسوم", "تكلفة", "ساعه"]):
            lines = [f"💰 أسعار الساعة في كلية {fname}:\n"]
            for p in programs:
                lines.append(f"  • {p['name']}: {p.get('credit_hour_price', '؟')} دينار")
            return "\n".join(lines)

        if self._contains_any(question, ["معدل", "قبول", "نسبة", "نسبه"]):
            lines = [f"📊 معدلات القبول في كلية {fname}:\n"]
            for p in programs:
                lines.append(f"  • {p['name']}: {self._fmt_admission_fr(p.get('admission_rate'))}")
            return "\n".join(lines)

        if self._contains_any(question, ["مفتاح", "تنسيق"]):
            lines = [f"🔑 مفاتيح التنسيق في كلية {fname}:\n"]
            for p in programs:
                lines.append(f"  • {p['name']}: {self._fmt_key_fr(p.get('coordination_key'))}")
            return "\n".join(lines)

        lines = [f"🏛️  كلية {fname} — البرامج المتاحة ({len(programs)} برنامج):\n"]
        for p in programs:
            price = p.get("credit_hour_price", "؟")
            rate  = self._fmt_admission_fr(p.get("admission_rate"))
            lines.append(f"  • {p['name']}  |  الساعة: {price} دينار  |  القبول: {rate}")
        return "\n".join(lines)

    @staticmethod
    def _build_postgres_student_context(profile: Optional[dict]) -> str:
        if not profile:
            return ""

        student = profile.get("student") or {}
        lines = [
            "بيانات الطالب الحالي من EduPredict PostgreSQL (سري — للطالب نفسه فقط):",
            f"الاسم: {student.get('student_name') or 'غير متوفر'}",
            f"رقم الطالب: {student.get('id_student') or 'غير متوفر'}",
        ]
        if student.get("email"):
            lines.append(f"البريد الإلكتروني: {student['email']}")

        enrollments = profile.get("enrollments") or []
        if enrollments:
            lines.append("\nالتسجيلات الأكاديمية:")
            for enrollment in enrollments:
                lines.extend([
                    f"- رقم التسجيل: {enrollment.get('id')}",
                    f"  المقرر/العرض: {enrollment.get('course_presentation_id')}",
                    f"  النتيجة النهائية: {enrollment.get('final_result') or 'غير متوفرة'}",
                    f"  الساعات المدروسة: {enrollment.get('studied_credits') or 'غير متوفرة'}",
                    f"  المنطقة: {enrollment.get('region') or 'غير متوفرة'}",
                    f"  الفئة العمرية: {enrollment.get('age_band') or 'غير متوفرة'}",
                    f"  المؤهل الأعلى: {enrollment.get('highest_education') or 'غير متوفر'}",
                ])

        predictions = profile.get("predictions") or []
        if predictions:
            lines.append("\nآخر توقعات المخاطر الأكاديمية:")
            for prediction in predictions:
                lines.extend([
                    f"- اليوم الدراسي: {prediction.get('day_of_course')}",
                    f"  مستوى الخطر: {prediction.get('risk_level')}",
                    f"  احتمالية الخطر: {prediction.get('risk_probability')}",
                    f"  الإجراء المقترح: {prediction.get('recommended_action') or 'غير متوفر'}",
                ])

        assessments = profile.get("assessments") or []
        if assessments:
            lines.append("\nآخر التقييمات:")
            for assessment in assessments:
                lines.append(
                    f"- تقييم {assessment.get('id_assessment')}: "
                    f"العلامة {assessment.get('score') or 'غير متوفرة'}، "
                    f"تاريخ التسليم {assessment.get('date_submitted')}"
                )

        return "\n".join(lines)

    def fast_retrieval(self, question: str) -> Optional[str]:
        if not question or not question.strip():
            return None

        if self._contains_any(question, self._KW_ENROLLMENT):
            return self._fr_handle_enrollment(self._data)

        if self._contains_any(question, self._KW_GRANTS):
            return self._fr_handle_grants(question, self._data)

        if self._contains_any(question, self._KW_BENEFITS):
            result = self._fr_handle_benefits(self._data)
            if result:
                return result

        if self._contains_any(question, self._KW_UNIV_INFO):
            result = self._fr_handle_univ_info(question, self._data)
            if result:
                return result

        match = self._search_alias_index(question)
        if match:
            if match["type"] == "program":
                return self._fr_handle_program(question, match["data"], match["faculty"])
            if match["type"] == "faculty":
                return self._fr_handle_faculty(question, match)

        return None

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION 8 — CHAT (main orchestration)
    # ═════════════════════════════════════════════════════════════════════════

    def chat(self, question: str, session_id: str) -> dict:
        """
        Full chat pipeline:
        1. Semantic search → context
        2. LLM call
        Returns dict with answer and top_chunks.
        """
        # ── Step 1: semantic retrieval ───────────────────────────────────────
        relevant_chunks = self.semantic_search(
            question  = question,
            top_k     = TOP_K,
            threshold = SIM_THRESHOLD,
        )

        # ── Step 1b: فصل chunks العامة عن الـ rankings ──────────────────────
        general_chunks = [c for c in relevant_chunks if not c.startswith("[RANKING|")]

        # ── Step 1c: كشف نية السؤال — هل يسأل عن ranking شخص آخر؟ ──────────
        ranking_keywords = ["معدل", "ترتيب", "gpa", "معدله", "ترتيبه", "معدلها", "ترتيبها"]
        asking_about_ranking = any(kw in question for kw in ranking_keywords)

        rankings_data   = self._data.get("students_rankings", [])
        current_student = next(
            (s for s in rankings_data if s.get("student_id") == session_id), None
        )
        postgres_profile = None if current_student else get_postgres_student_profile(session_id)
        if asking_about_ranking:
            other_names = [
                s["student_name"].split()[0]
                for s in rankings_data
                if s.get("student_id") != session_id
            ]
            mentions_other = any(name in question for name in other_names)
            if mentions_other:
                blocked_answer = "عذراً، بيانات الترتيب والمعدلات خاصة بكل طالب ولا يمكن الاطلاع عليها."
                self.push_history(session_id, question, blocked_answer)
                return {"answer": blocked_answer, "top_chunks": []}

        # ── Step 1d: حقن بيانات الطالب الحالي مباشرة في الـ context ──────────
        student_context_chunk = ""
        if current_student:
            student_context_chunk = (
                f"بيانات الطالب الحالي (سري — للطالب نفسه فقط):\n"
                f"الاسم: {current_student['student_name']}\n"
                f"رقم الهوية: {current_student['student_id']}\n"
                f"المعدل التراكمي: {current_student['gpa']}\n"
                f"الترتيب على الدفعة: {current_student['rank']}"
            )
            context = "\n\n---\n\n".join(
                ([student_context_chunk] if student_context_chunk else []) + general_chunks
            )
        elif postgres_profile:
            student_context_chunk = self._build_postgres_student_context(postgres_profile)
            context = "\n\n---\n\n".join([student_context_chunk] + general_chunks)
        else:
            context = "\n\n---\n\n".join(general_chunks)

        # ── Step 1e: إضافة chunks الملفات المرفوعة إلى الـ context ─────────────
        all_uploaded_chunks = [
            chunk
            for chunks in self._uploaded_chunks.values()
            for chunk in chunks
        ]
        if all_uploaded_chunks:
            uploaded_context = "\n\n---\n\n".join(all_uploaded_chunks)
            context = context + "\n\n---\n\n[معلومات إضافية من ملفات مرفوعة]\n" + uploaded_context

        # ── Step 2: build prompt ─────────────────────────────────────────────
        identity_note = ""
        if current_student:
            identity_note = (
                f"\n\nالطالب الذي يحادثك الآن: {current_student['student_name']} "
                f"(رقم الهوية: {current_student['student_id']}). "
                f"أجبه عن بياناته مباشرة دون تحفظ، ولا تكشف بيانات أي طالب آخر."
            )
        elif postgres_profile:
            student = postgres_profile["student"]
            identity_note = (
                f"\n\nالطالب الذي يحادثك الآن: {student.get('student_name')} "
                f"(رقم الطالب: {student.get('id_student')}). "
                f"أجبه عن بياناته الشخصية والأكاديمية المتوفرة في السياق مباشرة، "
                f"ولا تطلب منه رقم الطالب مرة أخرى."
            )

        history      = self.get_history(session_id)
        history_text = self.fmt_history(history)
        system       = SYSTEM_PROMPT_TEMPLATE.format(context=context) + identity_note
        user_message = f"{history_text}السؤال: {question}"

        # ── Step 3: call LLM (Groq API) ──────────────────────────────────────
        if not GROQ_API_KEY:
            raise RuntimeError(
                "❌ GROQ_API_KEY غير موجود — أضفه في ملف .env"
            )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
            "temperature": 0.05,
        }

        answer = self._call_groq(headers, payload)
        self.push_history(session_id, question, answer)

        return {
            "answer":     answer,
            "top_chunks": general_chunks,
        }