"""
chatbot_core_edupredict_patch.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
أضف هذه الدوال داخل class IUGChatbot في chatbot_core.py
(بعد دالة delete_uploaded_file مباشرةً — السطر ~831)

لا تعدّل أي كود موجود — فقط أضف هذا الـ block.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ═════════════════════════════════════════════════════════════════════════
#  SECTION 4b — EDUPREDICT STUDENT DATA (PostgreSQL → uploaded_file)
# ═════════════════════════════════════════════════════════════════════════

def load_student_edupredict_data(self, student_id) -> bool:
    """
    جلب بيانات الطالب من PostgreSQL (EduPredict) وحفظها
    كـ uploaded_file collection منفصلة في MongoDB وبناء index لها.

    يُستدعى مباشرةً بعد تسجيل دخول الطالب.

    Returns True on success, False if no data found or error.
    """
    from edupredict_db import (
        fetch_student_full_profile,
        build_student_collection_name,
        profile_to_documents,
    )

    try:
        print(f"⏳ Fetching EduPredict data for student {student_id} …")
        profile = fetch_student_full_profile(student_id)

        if not profile or not profile.get("student_info"):
            print(f"⚠️  No EduPredict data found for student {student_id}.")
            return False

        documents = profile_to_documents(profile)
        if not documents:
            print(f"⚠️  profile_to_documents returned empty list for {student_id}.")
            return False

        col_name = build_student_collection_name(student_id)

        # upload_json_file handles MongoDB insert + index build
        result = self.upload_json_file(col_name, documents)
        print(
            f"✅ EduPredict data loaded for student {student_id}: "
            f"{result['inserted']} docs → collection '{col_name}'"
        )
        return True

    except Exception as exc:
        print(f"❌ load_student_edupredict_data failed for {student_id}: {exc}")
        return False


def chat_with_student_data(
    self,
    question: str,
    student_id,
    session_id: str,
) -> dict:
    """
    محادثة مع بيانات الطالب الشخصية المحملة من EduPredict.

    - تبحث في collection الطالب فقط (student_{id}_edupredict).
    - إذا لم تجد البيانات محملة → تحاول تحميلها أولاً.
    - تعيد dict مطابقاً لـ chat_with_file().
    """
    from edupredict_db import build_student_collection_name

    col_name = build_student_collection_name(student_id)

    # إذا لم تكن البيانات محملة → حاول تحميلها الآن
    if col_name not in self._uploaded_chunks:
        loaded = self.load_student_edupredict_data(student_id)
        if not loaded:
            return {
                "answer": (
                    "لا تتوفر بيانات أكاديمية لحسابك في النظام حالياً. "
                    "يرجى التواصل مع الدعم الفني."
                ),
                "top_chunks": [],
                "source": "edupredict",
            }

    return self.chat_with_file(
        question        = question,
        collection_name = col_name,
        session_id      = session_id,
    )


def get_student_collection_name(self, student_id) -> str:
    """Helper: اسم الـ collection الخاص ببيانات الطالب."""
    from edupredict_db import build_student_collection_name
    return build_student_collection_name(student_id)


def is_student_data_loaded(self, student_id) -> bool:
    """هل بيانات الطالب محملة في الذاكرة؟"""
    from edupredict_db import build_student_collection_name
    return build_student_collection_name(student_id) in self._uploaded_chunks
