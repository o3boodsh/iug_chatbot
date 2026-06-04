# التعديلات المطلوبة — ربط EduPredict PostgreSQL بالشات بوت

## الملفات الجديدة

| الملف | الوصف |
|-------|-------|
| `edupredict_db.py` | مدير اتصال PostgreSQL + دوال جلب بيانات الطالب |
| `main.py` | نسخة محدّثة من main.py (v6) |
| `chatbot_core_edupredict_patch.py` | دوال إضافية — أضفها لـ `IUGChatbot` (اختيارية) |

---

## 1. ثبّت المتطلبات

```bash
pip install psycopg2-binary
```

---

## 2. أضف متغير البيئة في `.env`

```env
# القيمة الافتراضية مضمّنة في edupredict_db.py — يمكن تجاوزها هنا
EDUPREDICT_DATABASE_URL=postgresql://edupredict_user:G1Xsim5ArjBPnMtbgLr8wKCn8wBXeD0i@dpg-d8f9jfvavr4c73a2a2m0-a.oregon-postgres.render.com/edupredict_db
```

---

## 3. انسخ الملفات إلى مجلد المشروع

```
chatbot/
├── chatbot_core.py        ← بدون تعديل
├── database.py            ← بدون تعديل
├── uploaded_files_db.py   ← بدون تعديل
├── edupredict_db.py       ← جديد ✅
└── main.py                ← استبدل بالنسخة الجديدة ✅
```

---

## 4. كيف يعمل النظام

### تسجيل الدخول `/auth/login`

```
student_id + pin
      │
      ├─→ MongoDB (النظام الأصلي IUG)
      │         source = "iug"
      │
      └─→ PostgreSQL EduPredict (إذا لم يُوجد في MongoDB)
                source = "edupredict"
                ↓
         يُحمَّل profile الطالب تلقائياً كـ uploaded_file
         collection: student_{id}_edupredict
```

### المحادثة العامة `/chat`
- تعمل كالمعتاد مع بيانات الجامعة
- تستخدم الـ token لمعرفة هوية الطالب

### المحادثة الشخصية `/chat/student` ← جديد
```json
POST /chat/student
{
  "question": "ما نتيجتي في مساق AAA",
  "session_id": "<JWT_TOKEN>"
}
```
- تُجيب **فقط** من بيانات الطالب الشخصية
- تتحقق من الـ token → تستخرج student_id → تبحث في collection الطالب
- إذا لم تكن البيانات محملة → تجلبها من PostgreSQL تلقائياً

---

## 5. Endpoints الجديدة

| Method | Path | الوصف |
|--------|------|-------|
| `POST` | `/chat/student` | محادثة مع البيانات الأكاديمية الشخصية |
| `POST` | `/student/load-data?session_id=TOKEN` | تحميل بيانات الطالب يدوياً |
| `GET`  | `/student/profile?session_id=TOKEN` | عرض ملف الطالب الكامل (JSON) |

---

## 6. بنية collection الطالب في MongoDB (uploaded_files)

```
collection: student_12345_edupredict
documents:
  - section: "معلومات الطالب الأساسية"
  - section: "التسجيل في المساق"  (واحد لكل مساق)
  - section: "التقييمات والدرجات"  (واحد لكل تقييم)
  - section: "نشاط المنصة الإلكترونية"
  - section: "توقعات الأداء الأكاديمي"
  - section: "الساعة الأكاديمية"
```

---

## 7. PIN في EduPredict

حالياً يقارن PIN مع `password_hash` أو `pin` مباشرةً.
إذا كانت كلمات المرور مشفّرة (bcrypt)، عدّل دالة `login()` في `main.py`:

```python
import bcrypt
# بدل:
if stored_pin != req.pin.strip():
# استخدم:
if not bcrypt.checkpw(req.pin.encode(), stored_pin.encode()):
```

---

## 8. خصوصية البيانات

- كل طالب يرى **فقط** collection الخاصة به (`student_{id}_edupredict`)
- الـ token يحمل student_id ومُوقَّع — لا يمكن التلاعب به
- الـ LLM يتلقى تعليمات صارمة بعدم مشاركة بيانات طالب مع آخر
