import os
from bson import ObjectId
from flask import session, redirect, url_for, render_template

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session, make_response
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

from bson import ObjectId
import cv2
import numpy as np
import json
import requests
from dotenv import load_dotenv
import base64
import io
import time
import csv
from datetime import datetime as _dt
import sys
import traceback
import threading
import atexit 

# --- ENVIRONMENT & APP SETUP ---
load_dotenv()

# 🔹 Step 1: Create Flask app
app = Flask(__name__)

# 🔹 Step 2: Configure MongoDB - SECURE CONNECTION
# --- SECURE MONGODB ATLAS CONNECTION (Direct PyMongo Client) ---
from pymongo import MongoClient
import certifi

ca = certifi.where()
# WARNING: Replace this connection string with a secure environment variable (os.environ.get('MONGO_URI'))
client = MongoClient(
    "mongodb+srv://dbflaskuser:flask123@cluster0.tvpcrxu.mongodb.net/?retryWrites=true&w=majority",
    tls=True,
    tlsCAFile=ca
)
db = client["attendanceDB"]

students_collection = db["students"]
# 🟢 CORRECTION: Use the plural collection name 'subjects' for clarity and consistency
subjects_collection = db["subjects"] 
attendance_collection = db["attendance"]
teachers_collection = db["teachers"]
sessions_collection = db["sessions"]
snapshots_collection = db["snapshots"]

print("✅ Secure MongoDB Atlas connection established successfully.")


# 🔹 App Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# --- Mongo-backed Flask-Login User wrapper ---
class MongoUser(UserMixin):
    def __init__(self, doc):
        self._doc = doc

    def get_id(self):
        return str(self._doc.get('_id'))

    @property
    def id(self):
        return str(self._doc.get('_id'))

    @property
    def username(self):
        return self._doc.get('username')

    @property
    def email(self):
        return self._doc.get('email')

    @staticmethod
    def get(user_id):
        try:
            doc = teachers_collection.find_one({'_id': ObjectId(user_id)})
            if doc:
                return MongoUser(doc)
        except Exception:
            return None
        return None

# 🔹 Face Recognition Library Check
try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except Exception:
    FACE_RECOG_AVAILABLE = False
    print("Warning: face_recognition library not found. Recognition features will be disabled.")

# Minimal Camera Worker Implementation (Mock for background recognition)
class MinimalCamWorker:
    def __init__(self):
        self.running = False
        self.last_recognitions = [] 
        self.active_subject = None
        self.active_session = None
        self.attendance_callback = None
        self.video_capture = None
        self.encodings_loaded = False
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.running = True
            if not self.video_capture:
                # 0 typically refers to the default camera
                self.video_capture = cv2.VideoCapture(0)
            if FACE_RECOG_AVAILABLE and not self.encodings_loaded:
                self.load_encodings()
            print("MinimalCamWorker started (Mocking background thread).")

    def get_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            success, frame = self.video_capture.read()
            if success:
                # Simple face detection for visual feedback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Simple annotation
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Active Cam", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                return buffer.tobytes()
        return None

    def load_encodings(self):
        # Triggered when new face data is uploaded
        self.encodings_loaded = True
        print("Mock: Encodings reloaded for worker.")
        
    def stop(self):
        if self.video_capture:
            self.video_capture.release()
        self.running = False
        self.active_subject = None
        self.active_session = None
        print("MinimalCamWorker stopped.")

# Initialize the worker
cam_worker = MinimalCamWorker()
print("Using MinimalCamWorker mock for camera functions.")

# 🔹 OpenPyXL Check
try:
    import openpyxl
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False

# --- FACE DETECTION SYSTEM ---
class FaceDetectionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_students = []
        self.load_known_students()
    
    def load_known_students(self):
        """Load student information from database"""
        docs = students_collection.find()
        out = []
        for s in docs:
            out.append({'id': str(s.get('_id')), 'name': s.get('name'), 'roll_number': s.get('roll_number')})
        self.known_students = out
    
    def detect_faces(self, frame):
        """Detect faces in the given frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_locations = []
        # Convert OpenCV format (x, y, w, h) to face_recognition format (top, right, bottom, left)
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
        
        return face_locations

# Initialize face system later in app context
face_system = None


# --- FLASK-LOGIN USER LOADER ---
@login_manager.user_loader
def load_user(user_id):
    return MongoUser.get(user_id)


# --- ROUTES ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('choose'))

@app.route('/choose')
def choose():
    """Common landing page where user selects Teacher or Student flow."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('choose.html')

## Authentication Routes

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        teacher_doc = teachers_collection.find_one({'username': username})
        if teacher_doc and check_password_hash(teacher_doc.get('password_hash', ''), password):
            user = MongoUser(teacher_doc)
            login_user(user)
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if teachers_collection.find_one({'username': username}):
            flash('Username already exists')
            return render_template('register.html')

        if teachers_collection.find_one({'email': email}):
            flash('Email already exists')
            return render_template('register.html')

        teacher_doc = {
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.utcnow()
        }

        teachers_collection.insert_one(teacher_doc)
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        session.clear()
        flash('Successfully logged out', 'success')
    except Exception as e:
        flash('An error occurred during logout', 'error')
        print(f"Logout error: {str(e)}")
    return redirect(url_for('login'))

## Student Auth Routes

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    # Prevent cross-login confusion by logging out teacher session if one exists
    if current_user.is_authenticated:
        logout_user()
        session.clear()
        flash("Logged out teacher session. You can now log in as a student.")
        return redirect(url_for('student_login'))

    # Normal student session check
    if session.get('student_user_id'):
        return redirect(url_for('student_dashboard')) 

    if request.method == 'POST':
        roll_number = request.form.get('roll_number', '').strip()
        password = request.form.get('password', '')
        student_doc = students_collection.find_one({'roll_number': roll_number})
        if not student_doc or not student_doc.get('password_hash'):
            flash('Invalid roll number or password')
            return render_template('student_login.html')

        if not check_password_hash(student_doc.get('password_hash', ''), password):
            flash('Invalid roll number or password')
            return render_template('student_login.html')

        session['student_user_id'] = str(student_doc.get('_id'))
        flash('Login successful')
        return redirect(url_for('student_dashboard'))

    return render_template('student_login.html')

@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    # If already logged in as a student, redirect to dashboard
    if session.get('student_user_id'):
        return redirect(url_for('student_dashboard'))

    # --- Fetch subjects for the dropdown ---
    subjects_cursor = subjects_collection.find()
    subjects = []
    for s in subjects_cursor:
        subjects.append({
            'id': str(s.get('_id')),
            'name': s.get('name'),
            'code': s.get('code')
        })

    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        roll_number = (request.form.get('roll_number') or '').strip()
        email = (request.form.get('email') or '').strip()
        password = (request.form.get('password') or '').strip()
        selected_subject_ids = request.form.getlist('subject_id')

        # --- Validation ---
        if not name or not roll_number or not email or not password:
            flash("All fields are required.")
            return render_template('student_register.html', subjects=subjects)

        if not selected_subject_ids or len(selected_subject_ids) == 0:
            flash("At least one valid subject must be selected.")
            return render_template('student_register.html', subjects=subjects)

        # Convert valid subject IDs to ObjectIds
        subject_oids = []
        for sid in selected_subject_ids:
            try:
                subject_oids.append(ObjectId(sid))
            except Exception:
                continue

        if len(subject_oids) == 0:
            flash("Invalid subject selection. Please try again.")
            return render_template('student_register.html', subjects=subjects)

        # --- Duplicate checks ---
        if students_collection.find_one({'email': email}):
            flash("Email already registered.")
            return render_template('student_register.html', subjects=subjects)

        if students_collection.find_one({'roll_number': roll_number}):
            flash("Account already exists for this roll number.")
            return render_template('student_register.html', subjects=subjects)

        # --- Create student document ---
        student_doc = {
            'name': name,
            'roll_number': roll_number,
            'email': email,
            'password_hash': generate_password_hash(password),
            'face_encoding': None,
            'subjects': subject_oids,  # store ObjectIds for linking
            'created_at': datetime.utcnow()
        }

        try:
            result = students_collection.insert_one(student_doc)
            if result.inserted_id:
                flash("Registration successful! Please login.")
                return redirect(url_for('student_login'))
            else:
                flash("Registration failed. Please try again.")
        except Exception as e:
            # Added better logging for server-side error
            app.logger.error(f"Error inserting student: {e}\n{traceback.format_exc()}")
            flash("An error occurred while registering the student. Please try again.")

        # On any failure, re-render with subjects list
        return render_template('student_register.html', subjects=subjects)

    # GET → render form
    return render_template('student_register.html', subjects=subjects)

@app.route('/student/logout')
def student_logout():
    session.pop('student_user_id', None)
    return redirect(url_for('student_login'))

@app.route('/student/dashboard')
def student_dashboard():
    student_id = session.get('student_user_id')
    if not student_id:
        return redirect(url_for('student_login'))
    
    try:
        student_doc = students_collection.find_one({'_id': ObjectId(student_id)})
    except Exception:
        student_doc = None

    if not student_doc:
        flash('Student record not found')
        return redirect(url_for('student_login'))

    subject_stats = []
    subject_ids = student_doc.get('subjects') or []
    
    # --- CORRECTION 2: Handle string/ObjectId list items and ensure correct lookup ---
    for subj_id in subject_ids:
        subj_oid = None
        if isinstance(subj_id, ObjectId):
            subj_oid = subj_id
        else:
            try:
                # Backward compatibility for old documents storing strings
                subj_oid = ObjectId(subj_id)
            except Exception:
                continue

        if not subj_oid:
            continue
            
        subj_doc = subjects_collection.find_one({'_id': subj_oid})
        
        # If subject is not found (e.g., deleted), skip
        if not subj_doc:
            continue 

        total_records = attendance_collection.count_documents({'student_id': student_doc['_id'], 'subject_id': subj_oid})
        present_count = attendance_collection.count_documents({'student_id': student_doc['_id'], 'subject_id': subj_oid, 'status': 'present'})
        
        if total_records > 0:
            percent = round((present_count / total_records) * 100, 1)
        else:
            percent = None

        today_iso = datetime.now().date().isoformat()
        today_record = attendance_collection.find_one({'student_id': student_doc['_id'], 'subject_id': subj_oid, 'date': today_iso})
        today_status = today_record.get('status') if today_record else 'N/A'

        subject_stats.append({
            'subject': {'id': str(subj_doc.get('_id')), 'name': subj_doc.get('name')},
            'total': total_records,
            'present': present_count,
            'percent': percent,
            'today_status': today_status
        })

    return render_template('student_dashboard.html', student=student_doc, subject_stats=subject_stats)

@app.route('/student/<student_id>/upload_face', methods=['GET', 'POST'])
def upload_face(student_id):
    """Page to capture multiple face images and save encodings for a student."""
    try:
        student_doc = students_collection.find_one({'_id': ObjectId(student_id)})
        if not student_doc:
            return redirect(url_for('student_login'))
    except Exception:
        return redirect(url_for('student_login'))

    is_teacher = current_user.is_authenticated
    is_student_owner = session.get('student_user_id') == student_id
    
    if not (is_teacher or is_student_owner):
        flash('Authorization required to upload face data.')
        if is_teacher:
             return redirect(url_for('dashboard')) 
        else:
            return redirect(url_for('student_login')) 

    if request.method == 'POST':
        if not FACE_RECOG_AVAILABLE:
            return jsonify({'success': False, 'message': 'face_recognition library not installed on server'}), 500
            
        data = request.get_json() or {}
        img_data = data.get('image')
        if not img_data:
            return jsonify({'success': False, 'message': 'No image data provided'}), 400

        header, encoded = img_data.split(',', 1)
        try:
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            encs = face_recognition.face_encodings(rgb)
            if not encs:
                return jsonify({'success': False, 'message': 'No face found in image. Please ensure your face is clearly visible.'}), 400

            enc = encs[0].tolist()
            existing = student_doc.get('face_encoding') or []
            if isinstance(existing, str):
                try:
                    existing = json.loads(existing)
                except Exception:
                    existing = []

            existing.append(enc)
            students_collection.update_one({'_id': student_doc['_id']}, {'$set': {'face_encoding': existing}})
            
            try:
                cam_worker.load_encodings() 
            except Exception:
                pass
            
            return jsonify({'success': True, 'count': len(existing)})
        except Exception as e:
            app.logger.error(f"Error in upload_face: {e}\n{traceback.format_exc()}")
            return jsonify({'success': False, 'message': f'Server error processing image: {str(e)}'}), 500

    # GET -> render page
    count = 0
    existing = student_doc.get('face_encoding')
    if existing:
        try:
            count = len(existing) if isinstance(existing, list) else len(json.loads(existing)) 
        except Exception:
            count = 0

    return render_template('upload_face.html', student=student_doc, count=count, face_recog_available=FACE_RECOG_AVAILABLE)

## Teacher Routes

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        try:
            subjects = list(subjects_collection.find({'teacher_id': ObjectId(current_user.id)}))
        except Exception:
            subjects = []

        if not subjects:
            subjects = list(subjects_collection.find()) 

        subjects_data = [{'id': str(s.get('_id')), 'name': s.get('name'), 'code': s.get('code'), 'teacher_id': str(s.get('teacher_id')) if s.get('teacher_id') else None} for s in subjects]
        return render_template('dashboard.html', subjects=subjects, subjects_json=subjects_data, subjects_count=len(subjects))
    except Exception as e:
        app.logger.error(f'Error loading dashboard for Teacher {current_user.id}: {str(e)}\n{traceback.format_exc()}')
        flash(f'An error occurred loading the dashboard: {str(e)}')
        return render_template('dashboard.html', subjects=[], subjects_json=[], subjects_count=0)


@app.route('/subjects')
@login_required
def subjects():
    subs = list(subjects_collection.find({'teacher_id': ObjectId(current_user.id)}))
    if not subs:
        subs = list(subjects_collection.find())
    return render_template('subjects.html', subjects=subs)

@app.route('/add_subject', methods=['POST'])
@login_required
def add_subject():
    name = request.form['name']
    code = request.form['code']
    if subjects_collection.find_one({'code': code}):
        flash('Subject code already exists')
        return redirect(url_for('subjects'))

    subj = {
        'name': name,
        'code': code,
        'teacher_id': ObjectId(current_user.id),
        'created_at': datetime.utcnow()
    }
    subjects_collection.insert_one(subj)
    flash('Subject added successfully!')
    return redirect(url_for('subjects'))

@app.route('/delete_subject', methods=['POST'])
@login_required
def delete_subject():
    subject_id = request.form.get('subject_id')
    if not subject_id:
        flash('No subject specified')
        return redirect(url_for('subjects'))

    try:
        sid_obj = ObjectId(subject_id)
        subj = subjects_collection.find_one({'_id': sid_obj})
        if not subj:
            flash('Subject not found')
            return redirect(url_for('subjects'))

        if str(subj.get('teacher_id')) != current_user.id:
            flash('You are not authorized to delete this subject')
            return redirect(url_for('subjects'))

        subjects_collection.delete_one({'_id': subj.get('_id')})

        # Remove subject ID (as string and ObjectId) from all students' subject lists
        students_collection.update_many(
            {'subjects': {'$in': [subject_id, sid_obj]}},
            {'$pull': {'subjects': {'$in': [subject_id, sid_obj]}}}
        )

        # Delete all attendance records for this subject
        attendance_collection.delete_many({'subject_id': sid_obj})

        flash('Subject deleted successfully')
    except Exception as e:
        flash(f'Failed to delete subject: {str(e)}')

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({'success': True})

    return redirect(url_for('subjects'))

@app.route('/students/<subject_id>')
@login_required
def students(subject_id):
    try:
        subj = subjects_collection.find_one({'_id': ObjectId(subject_id)})
    except Exception:
        subj = None

    if not subj:
        flash('Subject not found')
        return redirect(url_for('dashboard'))

    if str(subj.get('teacher_id')) != current_user.id:
        flash('You are not authorized to view students for this subject.')
        return redirect(url_for('dashboard'))

    # Query students who have either the string ID (old) or the ObjectId (new/migrated)
    try:
        subj_obj = ObjectId(subject_id)
        query = {'subjects': {'$in': [subject_id, subj_obj]}}
    except:
        query = {'subjects': subject_id}

    students_cursor = students_collection.find(query)
    students_list = list(students_cursor)
    return render_template('students.html', subject={'id': str(subj.get('_id')), 'name': subj.get('name')}, students=students_list)

@app.route('/add_student', methods=['POST'])
@login_required
def add_student():
    name = request.form['name']
    roll_number = request.form['roll_number']
    subject_id = request.form['subject_id']
    
    try:
        subj_obj = ObjectId(subject_id)
        subj = subjects_collection.find_one({'_id': subj_obj})
    except Exception:
        subj = None

    if not subj:
        flash('Subject not found.')
        return redirect(url_for('subjects'))

    student_doc = students_collection.find_one({'roll_number': roll_number})
    if student_doc:
        # Check if student is already in the subject list (checking for both string and ObjectId formats)
        is_already_registered = False
        for s_id in (student_doc.get('subjects') or []):
            if str(s_id) == subject_id:
                is_already_registered = True
                break
        
        if not is_already_registered:
            # Add the ObjectId to the list
            students_collection.update_one({'_id': student_doc['_id']}, {'$addToSet': {'subjects': subj_obj}})
            flash(f'Existing student updated and added to subject.')
        else:
            flash(f'Student **{name}** is already registered for this subject.')
    else:
        new_doc = {
            'name': name,
            'roll_number': roll_number,
            'face_encoding': [],
            'email': None,
            'password_hash': None,
            'subjects': [subj_obj], # <-- Use ObjectId here
            'created_at': datetime.utcnow()
        }
        students_collection.insert_one(new_doc)
        flash('Student added successfully!')

    return redirect(url_for('students', subject_id=subject_id))

@app.route('/attendance/<subject_id>')
@login_required
def attendance(subject_id):
    try:
        subj_obj = ObjectId(subject_id)
        subj = subjects_collection.find_one({'_id': subj_obj})
    except Exception:
        subj = None
        subj_obj = None

    if not subj:
        flash('Subject not found')
        return redirect(url_for('dashboard'))

    # Query students who have either the string ID (old) or the ObjectId (new/migrated)
    students_list = list(students_collection.find({'subjects': {'$in': [subject_id, subj_obj]}}))

    today_iso = datetime.now().date().isoformat()
    attendance_records = list(attendance_collection.find({'subject_id': subj_obj, 'date': today_iso}))
    attendance_status = {str(r.get('student_id')): r.get('status') for r in attendance_records}

    active_session = sessions_collection.find_one({'subject_id': subj_obj, 'is_active': True})

    return render_template('attendance.html', subject={'id': str(subj.get('_id')), 'name': subj.get('name')}, students=students_list,
                            attendance_status=attendance_status, today=datetime.now().date(),
                            active_session=active_session)

@app.route('/start_attendance', methods=['POST'])
@login_required
def start_attendance():
    subject_id = request.form.get('subject_id')
    try:
        subj_obj = ObjectId(subject_id)
        existing = sessions_collection.find_one({'subject_id': subj_obj, 'is_active': True})
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid Subject ID format'})

    if existing:
        return jsonify({'success': False, 'message': 'Attendance session already active', 'session_id': str(existing.get('_id'))})

    session_doc = {
        'subject_id': subj_obj,
        'teacher_id': ObjectId(current_user.id),
        'start_time': datetime.utcnow(),
        'is_active': True,
        'created_at': datetime.utcnow()
    }

    res = sessions_collection.insert_one(session_doc)

    try:
        cam_worker.active_subject = subject_id # Stored as string for cam_worker ease of use
        cam_worker.active_session = str(res.inserted_id)
        cam_worker.attendance_callback = _attendance_callback
        cam_worker.start()
    except Exception as e:
        app.logger.warning(f"Failed to set cam_worker active session: {e}")

    return jsonify({'success': True, 'session_id': str(res.inserted_id)})

@app.route('/stop_attendance', methods=['POST'])
@login_required
def stop_attendance():
    session_id = request.form['session_id']
    try:
        sessions_collection.update_one({'_id': ObjectId(session_id)}, {'$set': {'is_active': False, 'end_time': datetime.utcnow()}})
        
        try:
            cam_worker.stop()
            cam_worker.active_subject = None
            cam_worker.active_session = None
            cam_worker.attendance_callback = None
            cam_worker.last_recognitions = []
        except Exception:
            pass
        return jsonify({'success': True})
    except Exception:
        return jsonify({'success': False, 'message': 'Session not found'})

@app.route('/mark_present', methods=['POST'])
@login_required
def mark_present():
    student_id = request.form['student_id']
    subject_id = request.form['subject_id']
    today_iso = datetime.now().date().isoformat()
    try:
        sid_obj = ObjectId(student_id)
        sub_obj = ObjectId(subject_id)
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID format'})

    attendance_collection.update_one(
        {'student_id': sid_obj, 'subject_id': sub_obj, 'date': today_iso},
        {'$set': {'status': 'present', 'timestamp': datetime.utcnow()}},
        upsert=True
    )

    return jsonify({'success': True})

# --- Video Streaming and Recognition ---

@app.route('/video_feed')
@login_required
def video_feed():
    return render_template('video_feed.html')

@app.route('/video')
@login_required
def video():
    def generate_frames():
        global face_system
        if face_system is None:
            try:
                face_system = FaceDetectionSystem()
            except Exception as e:
                print(f"Error initializing FaceDetectionSystem in video stream: {e}")
                
        if cam_worker.running:
            flash("Error: Camera is being used by the background attendance worker. Stop the session to view raw feed.")
            return

        cap = cv2.VideoCapture(0)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            if face_system:
                face_locations = face_system.detect_faces(frame)
                
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, "Face Detected", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "No Face System", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        cap.release()
    
    if cam_worker.running:
          flash("The camera is currently in use by an active attendance session. Please stop the session first.")
          return redirect(url_for('dashboard'))
          
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_worker')
@login_required
def stream_worker():
    """Stream MJPEG frames served by the background CameraWorker."""
    try:
        # Note: Worker should be started via /start_attendance
        pass
    except Exception as e:
        app.logger.error(f"Error starting cam_worker: {e}")

    def gen():
        while True:
            frame = cam_worker.get_frame() 
            if frame:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(1)
                continue 
            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_recognize')
@login_required
def stream_recognize():
    """Stream MJPEG frames annotated with recognition results from cam_worker."""
    if not cam_worker.active_session or not cam_worker.running:
        return Response("Camera worker is not active for recognition.", status=400)

    def gen():
        while True:
            frame = cam_worker.get_frame() 
            annotated = frame
            
            try:
                if frame and cam_worker.last_recognitions:
                    nparr = np.frombuffer(frame, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    for rec in cam_worker.last_recognitions:
                        x1, y1, x2, y2 = rec['box'] 
                        color = (0, 255, 0) if rec.get('live') else (0, 0, 255) 
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        label = rec.get('name') or ('Unknown')
                        cv2.rectangle(img, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)
                        cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        
                    ret2, jpg = cv2.imencode('.jpg', img)
                    if ret2:
                        annotated = jpg.tobytes()
            except Exception as e:
                app.logger.error(f"Annotation error: {e}")
                pass

            if annotated:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + annotated + b'\r\n')
            else:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
                time.sleep(1)
            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recognitions_json')
@login_required
def recognitions_json():
    if not cam_worker.active_session:
        return jsonify({'recognitions': [], 'message': 'No active attendance session.'})
    try:
        return jsonify({'recognitions': cam_worker.last_recognitions})
    except Exception:
        return jsonify({'recognitions': []})


# Attendance callback 
def _attendance_callback(rec, subject_id, session_id):
    try:
        sid = rec.get('student_id')
        if not sid or not subject_id:
            return

        today_iso = datetime.now().date().isoformat()
        try:
            sid_obj = ObjectId(sid)
            sub_obj = ObjectId(subject_id)
            ses_obj = ObjectId(session_id)
        except Exception:
            # Should not happen if data is clean, but guard against it
            return

        attendance_collection.update_one(
            {'student_id': sid_obj, 'subject_id': sub_obj, 'date': today_iso},
            {'$set': {'status': 'present', 'timestamp': datetime.utcnow(), 'session_id': ses_obj}},
            upsert=True
        )

        # Snapshot saving logic
        snap = rec.get('snapshot')
        if snap:
            try:
                import os
                snaps_dir = os.path.join(os.getcwd(), 'snapshots')
                os.makedirs(snaps_dir, exist_ok=True)
                fname = f"snap_{sid}_{session_id}_{int(time.time())}.jpg"
                with open(os.path.join(snaps_dir, fname), 'wb') as f:
                    f.write(snap)
                
                try:
                    snapshots_collection.insert_one({'student_id': sid_obj, 'session_id': ses_obj, 'filename': fname, 'created_at': datetime.utcnow()})
                except Exception:
                    pass
            except Exception as e:
                app.logger.error(f"Error saving snapshot: {e}")

    except Exception as e:
        app.logger.error(f"Error in attendance callback: {e}\n{traceback.format_exc()}")


@app.route('/worker/set_active', methods=['POST'])
@login_required
def worker_set_active():
    subject_id = request.form.get('subject_id')
    session_id = request.form.get('session_id')
    try:
        cam_worker.active_subject = subject_id if subject_id else None 
        cam_worker.active_session = session_id if session_id else None 
        cam_worker.attendance_callback = _attendance_callback
        cam_worker.start() 
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/export_attendance')
@login_required
def export_attendance():
    """Export attendance as CSV or Excel."""
    subject_id = request.args.get('subject_id')
    date_str = request.args.get('date')
    fmt = request.args.get('format', 'csv').lower()

    try:
        if date_str:
            date_obj = _dt.strptime(date_str, '%Y-%m-%d').date()
        else:
            date_obj = datetime.now().date()
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid date format; use YYYY-MM-DD'}), 400

    date_iso = date_obj.isoformat()
    query = {'date': date_iso}
    
    subj_obj = None
    if subject_id:
        try:
            subj_obj = ObjectId(subject_id)
            query['subject_id'] = subj_obj
        except Exception:
            pass

    rows = list(attendance_collection.find(query))

    # Helper function for Mongo lookup
    def find_doc(collection, oid_or_str):
        if isinstance(oid_or_str, ObjectId):
            return collection.find_one({'_id': oid_or_str})
        elif oid_or_str:
            try:
                return collection.find_one({'_id': ObjectId(oid_or_str)})
            except Exception:
                return None
        return None

    if fmt == 'csv' or not HAVE_OPENPYXL:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['student_id', 'student_name', 'roll_number', 'subject_id', 'subject_name', 'date', 'status', 'timestamp', 'snapshot_file'])

        import os
        snaps_dir = os.path.join(os.getcwd(), 'snapshots')

        for r in rows:
            try:
                sid = r.get('student_id')
                subid = r.get('subject_id')
                
                stu = find_doc(students_collection, sid)
                sub = find_doc(subjects_collection, subid)

                snapshot_file = ''
                if os.path.isdir(snaps_dir) and sid:
                    snapshot_meta = snapshots_collection.find_one({'student_id': sid, 'session_id': r.get('session_id')})
                    if snapshot_meta:
                        snapshot_file = snapshot_meta.get('filename')

                writer.writerow([
                    str(sid) if sid else '',
                    stu.get('name') if stu else '',
                    stu.get('roll_number') if stu else '',
                    str(subid) if subid else '',
                    sub.get('name') if sub else '',
                    r.get('date'),
                    r.get('status'),
                    r.get('timestamp').isoformat() if r.get('timestamp') else '',
                    snapshot_file
                ])
            except Exception:
                app.logger.error(f"Error processing row for attendance: {traceback.format_exc()}")
                continue

        output.seek(0)
        return Response(output.getvalue(), mimetype='text/csv', headers={
            'Content-Disposition': f'attachment; filename=attendance_{date_obj.isoformat()}.csv'
        })

    if fmt == 'xlsx':
        if not HAVE_OPENPYXL:
            return jsonify({'success': False, 'message': 'openpyxl not installed; install to enable xlsx export'}), 400

        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(['student_id', 'student_name', 'roll_number', 'subject_id', 'subject_name', 'date', 'status', 'timestamp', 'snapshot_file'])

        import os
        snaps_dir = os.path.join(os.getcwd(), 'snapshots')

        for r in rows:
            try:
                sid = r.get('student_id')
                subid = r.get('subject_id')
                
                stu = find_doc(students_collection, sid)
                sub = find_doc(subjects_collection, subid)
                
                snapshot_file = ''
                if os.path.isdir(snaps_dir) and sid:
                    snapshot_meta = snapshots_collection.find_one({'student_id': sid, 'session_id': r.get('session_id')})
                    if snapshot_meta:
                        snapshot_file = snapshot_meta.get('filename')

                ws.append([
                    str(sid) if sid else '',
                    stu.get('name') if stu else '',
                    stu.get('roll_number') if stu else '',
                    str(subid) if subid else '',
                    sub.get('name') if sub else '',
                    r.get('date'),
                    r.get('status'),
                    r.get('timestamp').isoformat() if r.get('timestamp') else '',
                    snapshot_file
                ])
            except Exception:
                app.logger.error(f"Error processing row for student in XLSX: {traceback.format_exc()}")
                continue

        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return Response(bio.read(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={
            'Content-Disposition': f'attachment; filename=attendance_{date_obj.isoformat()}.xlsx'
        })

    return jsonify({'success': False, 'message': 'Unsupported format'}), 400

@app.route('/absence_list/<subject_id>')
@login_required
def absence_list(subject_id):
    try:
        subj_obj = ObjectId(subject_id)
        subj = subjects_collection.find_one({'_id': subj_obj})
    except Exception:
        subj = None

    if not subj:
        flash('Subject not found')
        return redirect(url_for('dashboard'))

    today_iso = datetime.now().date().isoformat()
    
    # Query students who have either the string ID (old) or the ObjectId (new/migrated)
    all_students_for_subject = list(students_collection.find({'subjects': {'$in': [subject_id, subj_obj]}}))

    present_records_today = list(attendance_collection.find({'subject_id': subj_obj, 'date': today_iso, 'status': 'present'}))
    present_ids = {r.get('student_id') for r in present_records_today}

    absent_students = []
    for s in all_students_for_subject:
        student_id_obj = s.get('_id')
        if student_id_obj not in present_ids:
             absent_students.append(s)

    return render_template('absence_list.html', subject={'id': str(subj.get('_id')), 'name': subj.get('name')}, 
                             absent_students=absent_students, today=datetime.now().date())

# --- Error Handlers and Debug Routes ---

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_message="Page not found (404)"), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"500 Internal Error: {error}\n{traceback.format_exc()}")
    return render_template('error.html', error_message="Internal server error (500)"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled Exception: {e}\n{traceback.format_exc()}")
    return render_template('error.html', error_message=f"Error: {str(e)}"), 500

@app.route('/__debug_subjects')
def debug_subjects():
    try:
        subs = list(subjects_collection.find())
        out = []
        for s in subs:
            out.append({'id': str(s.get('_id')), 'name': s.get('name'), 'code': s.get('code'), 'teacher_id': str(s.get('teacher_id')) if s.get('teacher_id') else None, 'created_at': s.get('created_at').isoformat() if s.get('created_at') else None})
        return jsonify({'count': len(out), 'subjects': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/__whoami')
def debug_whoami():
    try:
        is_authenticated = bool(getattr(current_user, 'is_authenticated', False))
        is_student = 'student_user_id' in session
        return jsonify({
            'is_teacher_authenticated': is_authenticated,
            'teacher_id': getattr(current_user, 'id', None) if is_authenticated else None,
            'teacher_username': getattr(current_user, 'username', None) if is_authenticated else None,
            'is_student_authenticated': is_student,
            'student_id': session.get('student_user_id', None)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/__debug_dashboard')
def debug_dashboard():
    try:
        teacher_id = getattr(current_user, 'id', None)
        subs_by_teacher = list(subjects_collection.find({'teacher_id': ObjectId(teacher_id) })) if teacher_id else []
        subjects_by_teacher = [ {'id': str(s.get('_id')), 'name': s.get('name'), 'teacher_id': str(s.get('teacher_id'))} for s in subs_by_teacher ]
        fallback_docs = list(subjects_collection.find())
        fallback = [ {'id': str(s.get('_id')), 'name': s.get('name'), 'teacher_id': str(s.get('teacher_id'))} for s in fallback_docs ]
        final = subjects_by_teacher if subjects_by_teacher else fallback
        return jsonify({ 'teacher_id': teacher_id, 'subjects_by_teacher': subjects_by_teacher, 'fallback_subjects': fallback, 'final_subjects': final })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/__render_test')
def render_test():
    subs = list(subjects_collection.find())
    return render_template('dashboard.html', subjects=subs)

@app.route('/api/subjects')
@login_required
def api_subjects():
    """Return subjects for the logged-in teacher as JSON."""
    try:
        subs = list(subjects_collection.find({'teacher_id': ObjectId(current_user.id)}))
        if not subs:
            subs = list(subjects_collection.find())
        out = [{'id': str(s.get('_id')), 'name': s.get('name'), 'code': s.get('code'), 'teacher_id': str(s.get('teacher_id')) if s.get('teacher_id') else None} for s in subs]
        return jsonify({'count': len(out), 'subjects': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- MAIN RUN BLOCK ---
if __name__ == '__main__':
    with app.app_context():
        # Ensure MongoDB indexes for important fields
        try:
            students_collection.create_index('roll_number', unique=True)
            students_collection.create_index('email', unique=True, sparse=True)
            subjects_collection.create_index('code', unique=True)
            teachers_collection.create_index('username', unique=True)
            teachers_collection.create_index('email', unique=True)
            # Compound index for fast attendance lookups
            attendance_collection.create_index([('student_id', 1), ('subject_id', 1), ('date', 1)])
            attendance_collection.create_index([('subject_id', 1), ('date', 1)])
            sessions_collection.create_index([('subject_id', 1), ('is_active', 1)])
        except Exception as e:
            app.logger.warning(f'Could not create MongoDB indexes: {e}')
        
        # Attempt to initialize face detection system
        try:
            face_system = FaceDetectionSystem()
            # Check if cascade is loaded
            if face_system.face_cascade.empty():
                raise Exception("Haar Cascade Classifier failed to load.")
            print("Face Detection System (OpenCV) initialized successfully.")
        except Exception as e:
            app.logger.warning(f"Warning: Could not initialize FaceDetectionSystem (Haar Cascade): {e}")
            
        # Create default teacher if none exists
        try:
            if teachers_collection.count_documents({}) == 0:
                teachers_collection.insert_one({
                    'username': 'admin',
                    'email': 'admin@school.com',
                    'password_hash': generate_password_hash('admin123'),
                    'created_at': datetime.utcnow()
                })
                print("Default teacher created: username=admin, password=admin123")
        except Exception as e:
            app.logger.warning(f"Could not ensure default teacher in MongoDB: {e}")
    
    # Stop the minimal worker on shutdown
    atexit.register(cam_worker.stop) 
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)