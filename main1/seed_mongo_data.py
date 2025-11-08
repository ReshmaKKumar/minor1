from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from datetime import datetime
import certifi
from bson import ObjectId

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://dbflaskuser:flask123@cluster0.tvpcrxu.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "attendanceDB"
# ---------------------

# ✅ Step 1: Connect to MongoDB Atlas
ca = certifi.where()
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=ca
)
print("Connecting to MongoDB Atlas...")

# ✅ Step 2: Select database and collections
db = client[DB_NAME]
teachers = db["teachers"]
subjects = db["subjects"]
students = db["students"]
attendance = db["attendance"]
sessions = db["sessions"]
snapshots = db["snapshots"]
print(f"Connected to Database: {DB_NAME}")

# --- Data Seeding ---

# ✅ Step 3: Create or find default teacher (admin)
teacher_doc = teachers.find_one({"username": "admin"})
if not teacher_doc:
    teacher_doc = {
        "_id": ObjectId(),
        "teacher_id": "T001",
        "username": "admin",
        "email": "admin@school.com",
        "password_hash": generate_password_hash("admin123"),
        "created_at": datetime.utcnow()
    }
    teachers.insert_one(teacher_doc)
    print("✅ Added default teacher: username=admin, password=admin123")
else:
    print("ℹ️ Default teacher already exists.")

teacher_id = teacher_doc["_id"]

# ✅ Step 4: Insert subjects and assign admin as teacher for DBMS & Python
sample_subjects = [
    {"name": "Database Management Systems", "code": "CS501", "teacher_id": teacher_id, "created_at": datetime.utcnow()},
    {"name": "Python Programming", "code": "CS502", "teacher_id": teacher_id, "created_at": datetime.utcnow()},
    {"name": "Operating Systems", "code": "CS503", "teacher_id": None, "created_at": datetime.utcnow()},
    {"name": "Computer Networks", "code": "CS504", "teacher_id": None, "created_at": datetime.utcnow()}
]

subject_ids = {}
for subj in sample_subjects:
    existing = subjects.find_one({"code": subj["code"]})
    if not existing:
        result = subjects.insert_one(subj)
        subject_ids[subj["code"]] = result.inserted_id
        print(f"✅ Added subject: {subj['name']}")
    else:
        subject_ids[subj["code"]] = existing["_id"]
        # Ensure teacher_id is set correctly
        if not existing.get("teacher_id") and subj["teacher_id"]:
            subjects.update_one({"_id": existing["_id"]}, {"$set": {"teacher_id": subj["teacher_id"]}})
            print(f"🔄 Linked admin as teacher for: {subj['name']}")
        else:
            print(f"ℹ️ Subject already exists: {subj['name']}")

# ✅ Step 5: Get necessary subject IDs for student enrollment
try:
    dbms_id = subject_ids["CS501"]
    python_id = subject_ids["CS502"]
except KeyError:
    print("❌ ERROR: Could not find required subject IDs (CS501 or CS502). Check Step 4.")
    client.close()
    exit()

# ✅ Step 6: Insert two sample students and enroll them in DBMS & Python
sample_students = [
    {
        "name": "Sanidhya",
        "roll_number": "23CS001",
        "email": "sanidhya@example.com",
        "password_hash": generate_password_hash("sani123"),
        "face_encoding": None,
        "subjects": [dbms_id, python_id], # Use ObjectIds directly
        "created_at": datetime.utcnow()
    },
    {
        "name": "Pranamya",
        "roll_number": "23CS002",
        "email": "pranamya@example.com",
        "password_hash": generate_password_hash("pranu123"),
        "face_encoding": None,
        "subjects": [dbms_id, python_id], # Use ObjectIds directly
        "created_at": datetime.utcnow()
    }
]

for stu in sample_students:
    existing = students.find_one({"roll_number": stu["roll_number"]})
    if not existing:
        students.insert_one(stu)
        print(f"✅ Added student: {stu['name']} (enrolled in DBMS & Python)")
    else:
        # Ensure subjects field contains both ObjectIds
        students.update_one(
            {"_id": existing["_id"]},
            {"$addToSet": {"subjects": {"$each": [dbms_id, python_id]}}}
        )
        print(f"🔄 Updated existing student: {stu['name']} (added DBMS & Python if missing)")

# --- Data Cleanup (The fix for "Unknown Subject") ---

# ✅ Step 7: Fix subject ObjectId consistency for all students
print("\n🔧 Verifying student subject ObjectIds and fixing old string IDs...")

for student in students.find():
    fixed_ids = []
    has_changed = False
    
    # Iterate through current subjects (which might be strings or ObjectIds)
    for sid in student.get("subjects", []):
        sid_obj = None
        
        try:
            # 1. Convert to proper ObjectId if stored as string
            if isinstance(sid, str):
                sid_obj = ObjectId(sid)
                if sid_obj != sid: # Check if conversion happened
                    has_changed = True
            else:
                sid_obj = sid

            # 2. Verify the subject actually exists in subjects collection
            if subjects.find_one({"_id": sid_obj}):
                fixed_ids.append(sid_obj)
            else:
                # Subject no longer exists, skip it and flag change
                has_changed = True 

        except Exception:
            # Invalid string format, skip and flag change
            has_changed = True
            continue

    # Update only if the list structure or content has changed
    if has_changed or fixed_ids != student.get("subjects", []):
        students.update_one({"_id": student["_id"]}, {"$set": {"subjects": fixed_ids}})
        print(f"✅ Fixed subject references for {student.get('name')}")
    else:
        print(f"✔️ {student.get('name')} already has correct ObjectIds or no changes needed.")

# ✅ Step 8: Ensure empty collections exist (for future use)
for col in [attendance, sessions, snapshots]:
    # Check if the collection exists, creating it implicitly if needed (or just printing confirmation)
    if col.estimated_document_count() == 0:
        print(f"🆕 Collection exists and is empty: {col.name}")
    else:
        print(f"ℹ️ Collection {col.name} has {col.estimated_document_count()} documents.")


print("\n🎉 All collections seeded and verified successfully!")
print("Next step: Run your Flask application (`python app.py`) and log in.")

client.close()